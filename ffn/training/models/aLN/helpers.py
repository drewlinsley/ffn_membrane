import tensorflow as tf
import numpy as np
from ops import initialization


def label_backward(
        labels_pre,
        labels_post,
        caps_match,
        labels_filter,
        decay_const,
        b,
        strides = [1, 1, 1, 1],
        dtype = tf.float32,
        labels_gate_scale = None,
        labels_gate_bias = None,
        caps1_norm = None,
        labels_update_mode = 'pushpull', #'average' or 'add'
        sensitivity_const=None,
        pushpull_scale=None,
        pushpull_bias=None,
        cap_labels=False,
        fixed_labels_mask=None):
    '''
    :param labels_pre:
        B, H, W, C where C = fc
    :param labels_post:
        B, H', W', C' where C' = fc'
    :param tolerance_const:
    :param sensitivity_const:
    :return:
    '''
    # Propagate label
    labels_pre_candidate = propagate_labels(labels_post, labels_pre, caps_match, labels_filter, b, mode='')

    # Apply label update
    if labels_update_mode=='average':
        sensitivity_const_positive = tf.nn.relu(sensitivity_const)
        new_lablel = tf.multiply(labels_pre + labels_pre_candidate*sensitivity_const_positive, 1/(1 + sensitivity_const_positive)) \
                     - tf.nn.relu(decay_const)
        ## propagation between bars is equally strong between lower and higher ends
    elif labels_update_mode=='add':
        new_lablel = labels_pre + labels_pre_candidate \
                     - tf.nn.relu(decay_const)
    elif labels_update_mode=='pushpull':
        print('label_backward: pushpull update')
        update_gate = tf.nn.sigmoid(tf.square(labels_pre - labels_pre_candidate)*pushpull_scale + pushpull_bias)
        new_lablel = labels_pre_candidate*update_gate + labels_pre*(1-update_gate) \
                     - tf.nn.relu(decay_const)
    if cap_labels:
        new_lablel = tf.minimum(tf.maximum(new_lablel, -6.), 6.)

    # All-or-zero-gating with caps norm
    if labels_gate_scale is not None:
        print('label_backward: gating on')
        labels_gate = tf.multiply(caps1_norm, labels_gate_scale) + labels_gate_bias
        new_lablel *= tf.sigmoid(labels_gate)
    if fixed_labels_mask is None:
        return new_lablel
    else:
        return tf.multiply(fixed_labels_mask, labels_pre) + \
               tf.multiply(1 - fixed_labels_mask, new_lablel)
    # TODO: some kind of optional capping-by-norm??



def propagate_labels(labels_post,
                     labels_pre,
                     caps_match,
                     labels_filter,
                     b,
                     mode='weight_and_convolve', strides=[1, 1, 1, 1]):
    if mode=='weight_and_convolve': # CONVENTIONAL
        labels_intermediate = tf.tile(labels_post, [1, 1, 1, b])                           # B, H', W', C' where C' = b(fc')
        labels_intermediate_weighted = tf.multiply(labels_intermediate, caps_match)
        labels_pre_candidate = tf.nn.conv2d_transpose(labels_intermediate_weighted, labels_filter,
                                                      output_shape=labels_pre.get_shape(), strides=strides)
    else:
        raise NotImplementedError('propagate_labels: undefined mode ' + mode)
    return labels_pre_candidate




def cap_labels(labels, dtype):
    top = tf.ones_like(labels, dtype=dtype)*6.
    bottom = top*(-1.)
    return tf.minimum(tf.maximum(labels, bottom), top)



def get_caps_match(
        caps_pre,
        caps_post,
        caps_filter,
        tolerance_const,
        cc_post,
        b,
        eps,
        strides = [1, 1, 1, 1]):
    if caps_post.get_shape().as_list()[3] % cc_post > 0:
        raise ValueError('labels_backward: caps_post C dim is not divisible by cc_post')
    caps_intermediate1 = tf.nn.conv2d(caps_pre, caps_filter, strides=strides, padding='SAME')
    caps_intermediate2 = tf.tile(caps_post, [1, 1, 1, b]) 					# B, H', W', C'' where C'' = b(fc'(cc))
    caps_match, caps1_norm, caps2_norm = \
        capscompare(caps_intermediate1, caps_intermediate2, cc_post, b, tolerance_const, eps) # B, H', W', C' where C' = b(fc')
    return caps_intermediate1, caps_match, caps1_norm, caps2_norm



def feature_forward(
        caps_pre,
        labels_pre,
        labels_post,
        caps_filter,
        labels_filter,
        b,
        cc_post,
        strides = [1, 1, 1, 1],
        dtype=tf.float32):
    '''
    :param caps_pre:
        B, H, W, C where C = fc(cc)
    :param labels_pre:
        B, H, W, C where C = fc
    :param labels_post:
        B, H', W', C' where C' = fc'
    :param b:
        integer
    :return caps_intermediate:
        B, H', W', C'' where C'' = b(fc'(cc))
    :return caps_post:
        B, H', W', C' where C' = fc'(cc)
    '''
    label_post_shape = labels_post.get_shape().as_list()
    if label_post_shape[-1] % b > 0:
        raise ValueError('feature_forward: labels_post C dim is not divisible by b')
    # compute label match
    labels_intermediate1 = tf.nn.conv2d(labels_pre, labels_filter, strides=strides, padding='SAME')
    labels_intermediate2 = tf.tile(labels_post, [1, 1, 1, b])
    labels_match = labelcompare(labels_intermediate1, labels_intermediate2)         # B, H', W', C' where C' = b(fc')
    if cc_post>1:
        labels_match_replicated = tf.reshape(
                                    tf.tile(tf.expand_dims(labels_match, axis=4), [1, 1, 1, 1, cc_post]),
                                    labels_match.get_shape().as_list()[:3] + [-1])
    elif cc_post==1:
        labels_match_replicated = labels_match
    else:
        raise ValueError('feature_forward: cc should be at least 1')
    # feedforward capsule update
    caps_intermediate = tf.nn.conv2d(caps_pre, caps_filter, strides=strides,
                                     padding='SAME')  # B, H', W', C'' where C'' = b(fc'(cc))
    caps_intermediate_weighted = tf.multiply(labels_match_replicated, caps_intermediate)
    caps_intermediate_weighted_reshaped = tf.reshape(caps_intermediate_weighted, [-1, -1, -1, b, -1])
    return caps_intermediate, \
           tf.reduce_sum(caps_intermediate_weighted_reshaped, axis=[3], keep_dims=None)


def capscompare(
        caps1,
        caps2,
        cc,
        b,
        tolerance_const,
        eps):
    '''

    :param caps1 and caps2:
        B, H, W, C where C = b(fc(cc))
    :param tolerance_const:
        1, 1, 1, C where C = fc
    :return:
        B, H, W, C' where C = b(fc)
    '''
    caps_shape = caps1.get_shape().as_list()
    if cc >1:
        if caps_shape[3] % cc*b >0:
            raise ValueError('capscompare: caps C dim is not divisible by cc*b')
        else:
            if tolerance_const is None:
                out = capsdot(caps1, caps2, cc)
            else:
                print('aLNh: using tolerance_const')
                out = capsdot(caps1, caps2, cc) # B, H, W, C where C = b(fc)
                caps1_norm = caps2scalar_map(caps1, cc, eps)
                caps2_norm = caps2scalar_map(caps2, cc, eps)
                norm_combined = tf.sqrt(tf.multiply(caps1_norm, caps2_norm) + eps) #just to be safe
                tolerance_scaled = tf.multiply(tf.tile(tolerance_const, [1, 1, 1, b]), norm_combined)
                out += tolerance_scaled
            #out += tf.tile(tolerance_const, [1, 1, 1, b])
        return out, caps1_norm, caps2_norm
    elif cc==1:
        if tolerance_const is None:
            out = tf.exp(-tf.square(caps1 - caps2))
        else:
            print('aLNh: using tolerance_const')
            out = tf.exp(-tf.square(caps1 - caps2)/(tf.square(tolerance_const) + eps))
        return out, caps1, caps2
    else:
        raise ValueError('capscompare: cc should be at least 1')


def labelcompare(
        labels1,
        labels2):
    '''
    :param labels1 and labels2:
        B, H, W, C where C = b(fc)
    :param caps_dims:
    :param strides:
    :param padding:
    :param data_format:
    :param name:
    :return:
    '''
    '''Equivalent to 'delta' in draft'''
    return tf.multiply(tf.sigmoid(labels1), tf.sigmoid(labels2))



def capsdot(
        caps1,
        caps2,
        cc):
    '''
    :param caps1 and caps2:
        B, H, W, C where C = b(fc(cc))
    :return:
        B, H, W, C' where C' = b(fc)
    '''
    if cc >1:
        caps1_reshaped = tf.reshape(caps1, caps1.get_shape().as_list()[:3] + [-1, cc])
        caps2_reshaped = tf.reshape(caps2, caps2.get_shape().as_list()[:3] + [-1, cc])
        return tf.reduce_sum(tf.multiply(caps1_reshaped, caps2_reshaped), axis=4)
    else:
        raise ValueError('capsdot: cc should be greater than 1')



def caps2scalar_map(
        caps,
        cc,
        eps):
    '''
    :param caps:
        B, H, W, C where C = b(fc(cc))
    :return:
        B, H, W, C' where C' = b(fc)
    '''
    caps_shape_in = caps.get_shape().as_list()
    if cc > 1:
        if caps_shape_in[3] % (cc) > 0:
            raise ValueError('caps2scalar: caps C dim is not divisible by cc')
        else:
            caps_reshaped = tf.reshape(caps, caps_shape_in[:3] + [-1, cc])
        return tf.sqrt(tf.reduce_sum(tf.square(caps_reshaped), axis=4) + eps) # just to be safe
    elif cc==1:
        return caps
    else:
        raise ValueError('caps2scalar_map: cc should be at least 1')
    # return tf.norm(tf.reshape(caps, caps_shape_in[:3] + [-1, cc]), ord='euclidean', axis=4, keep_dims=None)



def caps2scalar_filter(
        caps_filter,
        cc):
    '''
    :param caps_filter:
        H, W, iC, oC where C = b(fc(cc))
    :return:
        H, W, iC', oC' where C' = b(fc)
    '''
    # TODO: Math is not rigorous. Approximating w/ Frobenious norm for now.
    #   A linear transformer W is described as k[A] where k is a scalar scaling coefficient and A is a length-preserving matrix.
    #   A is length-preserving iif A is orthogonal, or inverse(A) = transpose(A)
    caps_shape_in = caps_filter.get_shape().as_list()
    if cc > 1:
        if caps_shape_in[3] % (cc) > 0:
            raise ValueError('caps2scalar_filter: caps C dim is not divisible by cc')
        else:
            return tf.norm(tf.reshape(caps_filter, caps_shape_in[:2] + [caps_shape_in[2]/cc, cc] + [caps_shape_in[3]/cc, cc]), ord='euclidean', axis=(3,5), keep_dims=None)
    elif cc==1:
        return caps_filter
    else:
        raise ValueError('caps2scalar_filter: cc should be at least 1')

