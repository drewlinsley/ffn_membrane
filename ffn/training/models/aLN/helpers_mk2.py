import tensorflow as tf
import numpy as np
from ops import initialization

def get_mask(scalar_map, cc,
             scalar_to_caps=False, confidence=1.0):
    if scalar_to_caps:
        return scalar2caps_map(tf.nn.sigmoid(scalar_map*confidence), cc)
    else:
        return tf.nn.sigmoid(scalar_map * confidence)



def get_conv_and_mask(map,
                      filters,
                      square=True,
                      batchnorm_id=None, batchnorm_var_scope=None, batchnorm_reuse=False):
    # with square=True, making is more likely to detect local moment (instability)
    conv_output = tf.nn.conv2d(map, filters, strides=[1, 1, 1, 1], padding='SAME')
    if batchnorm_id is not None:
        with tf.variable_scope(batchnorm_var_scope + '_' + 'bn' + '_' + batchnorm_id, reuse=batchnorm_reuse) as scope:
            conv_output = tf.contrib.layers.batch_norm(
                inputs=conv_output,
                scale=True,
                center=True,  # PROBABLy FALSE
                fused=True,
                renorm=False,
                param_initializers={
                    'moving_mean': tf.constant_initializer(0.),
                    'moving_variance': tf.constant_initializer(1.),
                    'gamma': tf.constant_initializer(0.1)
                },
                updates_collections=None,
                scope=scope,
                reuse=False,
                is_training=True)
    if square:
        return tf.nn.sigmoid(tf.square(conv_output))
    else:
        return tf.nn.sigmoid(tf.nn.relu(conv_output))


def mix_maps(map_pos, map_neg, mix_gate):
    return map_pos*mix_gate + map_neg*(1-mix_gate)



def scalar2caps_map(
        scalar_map,
        cc):
    scalar_map_shape = scalar_map.get_shape().as_list()
    return tf.reshape(tf.tile(tf.expand_dims(scalar_map, axis=-1), [1, 1, 1, 1, cc]),
                      scalar_map_shape[:3] + [scalar_map_shape[-1] * cc])



def tile_by_bb(
        map,
        bb):
    return tf.tile(map, [1, 1, 1, bb])



def bb_to_batch(
        map,
        bb,
        batch_to_bb=False):
    map_shape = map.get_shape().as_list()
    if not batch_to_bb:
        map = tf.reshape(map, map_shape[:3] + [bb, map_shape[3]/bb])
        map = tf.transpose(map, perm=[3, 0, 1, 2, 4])
        map = tf.reshape(map, [bb*map_shape[0]] + map_shape[1:3] + [map_shape[3]/bb])
    else:
        map = tf.reshape(map, [bb, map_shape[0]/bb] + map_shape[1:])
        map = tf.transpose(map, perm=[1, 2, 3, 0, 4])
        map = tf.reshape(map, [map_shape[0]/bb] + map_shape[1:3] + [map_shape[3]*bb])
    return map



def softmax_over_bb(
        map,
        bb,
        competition_bias=None,
        scale_afterwards=True):
    if competition_bias is None:
        competition_bias = tf.constant(1., shape=[1, 1, 1, bb])
    competition_bias_shape = competition_bias.get_shape().as_list()
    map_shape = map.get_shape().as_list()
    if map_shape[-1] % bb >0:
        raise ValueError('map ch dim is not divisible by bb')

    map_reshaped = tf.reshape(map, map_shape[:3] + [bb, -1]) * \
                   tf.reshape(tf.nn.relu(competition_bias), competition_bias_shape[:3] + [bb, -1])
    map_reshaped_sm = tf.nn.softmax(map_reshaped, dim=3)
    if scale_afterwards:
        return tf.reshape(map_reshaped_sm*map_reshaped,
                          map_shape)
    else:
        return tf.reshape(map_reshaped_sm,
                          map_shape)



def sum_over_bb(
        map,
        bb):
    map_shape = map.get_shape().as_list()
    map_reshaped = tf.reshape(map, map_shape[:3] + [bb, -1])
    return tf.reduce_sum(map_reshaped, axis=3, keep_dims=False)



def bundle_switched_transpose_conv(
        map_post, map_pre,
        filter,
        bb,
        strides=[1, 1, 1, 1],
        batchnorm_id=None, batchnorm_var_scope=None, batchnorm_reuse=False):
    filter_shape = filter.get_shape().as_list()
    if filter_shape[-1] % bb >0:
        raise ValueError('filter out ch dim is not divisible by bb')
    filter_reshaped = tf.reshape(filter, filter_shape[:3] + [bb, filter_shape[3]/bb])
    filter_block_switched = tf.transpose(filter_reshaped, perm=[0, 1, 3, 2, 4])
    filter_block_switched = tf.reshape(filter_block_switched, filter_shape[:2] + [filter_shape[2]*bb, filter_shape[3]/bb])

    out = tf.nn.conv2d_transpose(map_post, filter_block_switched,
                                  output_shape=map_pre.get_shape(), strides=strides)
    if batchnorm_id is not None:
        with tf.variable_scope(batchnorm_var_scope + '_' + 'bn' + '_' + batchnorm_id, reuse=batchnorm_reuse) as scope:
            out = tf.contrib.layers.batch_norm(
                inputs=out,
                scale=True,
                center=True, # PROBABLy FALSE
                fused=True,
                renorm=False,
                param_initializers={
                    'moving_mean': tf.constant_initializer(0.),
                    'moving_variance': tf.constant_initializer(1.),
                    'gamma': tf.constant_initializer(0.1)
                    },
                updates_collections=None,
                scope=scope,
                reuse=False,
                is_training=True)
    return out



def caps_compare(
        caps1, caps2,
        cc):
    '''
    simple dot product for now
    '''
    caps_shape = caps1.get_shape().as_list()
    if caps_shape[3] % cc > 0:
        raise ValueError('capscompare: caps C dim is not divisible by cc')
    caps1_reshaped = tf.reshape(caps1, caps1.get_shape().as_list()[:3] + [-1, cc])
    caps2_reshaped = tf.reshape(caps2, caps2.get_shape().as_list()[:3] + [-1, cc])
    return tf.reduce_sum(tf.multiply(caps1_reshaped, caps2_reshaped), axis=4)



def generic_map_combine(
        map1, map2, map3,
        filter,
        bb,
        batchnorm_id=None, batchnorm_var_scope=None, batchnorm_reuse=False):
    # bb to batch
    map1 = bb_to_batch(map1, bb, batch_to_bb=False)
    map2 = bb_to_batch(map2, bb, batch_to_bb=False)
    if map3 is not None:
        map3 = bb_to_batch(map3, bb, batch_to_bb=False)

    # Check shape
    filter_shape = filter.get_shape().as_list()
    if map3 is not None:
        map_chs = map1.get_shape().as_list()[3] + map2.get_shape().as_list()[3] + map3.get_shape().as_list()[3]
    else:
        map_chs = map1.get_shape().as_list()[3] + map2.get_shape().as_list()[3]

    if filter_shape[2] != map_chs:
        raise ValueError('generic_map_combine: maps C dim does not match filter')

    # Concatenate
    if map3 is not None:
        concatenated = tf.concat([map1, map2, map3], axis=3)
    else:
        concatenated = tf.concat([map1, map2], axis=3)
    out = tf.nn.conv2d(concatenated, filter, strides=[1, 1, 1, 1], padding='SAME')

    # batch to bb
    out = bb_to_batch(out, bb, batch_to_bb=True)

    # Batchnorm
    if batchnorm_id is not None:
        with tf.variable_scope(batchnorm_var_scope + '_' + 'bn' + '_' + batchnorm_id, reuse=batchnorm_reuse) as scope:
            out = tf.contrib.layers.batch_norm(
                inputs=out,
                scale=True,
                center=True,  # PROBABLy FALSE
                fused=True,
                renorm=False,
                param_initializers={
                    'moving_mean': tf.constant_initializer(0.),
                    'moving_variance': tf.constant_initializer(1.),
                    'gamma': tf.constant_initializer(0.1)
                },
                updates_collections=None,
                scope=scope,
                reuse=False,
                is_training=True)
    return out



def label_backward(caps_pre, labels_pre, caps_post, labels_post,
                   caps_filter, compare_filter, labels_filter,
                   labels_deletegate_filter, labels_writegate_filter,
                   bundle_competition_bias, fig_ground_competition_bias, update_gain,
                   bb, cc_pre, cc_post,
                   batchnorm_var_scope=None):
    '''
    high-level function for label update
    '''
    # Separate fig and background caps & predict caps_pre with fig and ground caps_post
    fig_mask_caps = get_mask(labels_post, cc_post, scalar_to_caps=True)
    caps_post_fig = caps_post*fig_mask_caps
    caps_post_gnd = caps_pre*(1-fig_mask_caps)

    # predict caps_pre with fig and ground caps_post
    caps_pre_tiled = tile_by_bb(caps_pre, bb)
    caps_pre_predict_fig = bundle_switched_transpose_conv(caps_post_fig, caps_pre_tiled, caps_filter, bb, strides=[1, 1, 1, 1],
                                                          batchnorm_id='conv_caps_fig', batchnorm_var_scope=batchnorm_var_scope, batchnorm_reuse=False)
    caps_pre_predict_gnd = bundle_switched_transpose_conv(caps_post_gnd, caps_pre_tiled, caps_filter, bb, strides=[1, 1, 1, 1],
                                                          batchnorm_id='conv_caps_gnd', batchnorm_var_scope=batchnorm_var_scope, batchnorm_reuse=False)

    # compare caps
    caps_compared_fig = caps_compare(caps_pre_tiled, caps_pre_predict_fig, cc_pre)
    caps_compared_fig = tf.nn.relu(generic_map_combine(caps_compared_fig,
                                                       caps2scalar_map(caps_pre_tiled, cc_pre, 0.00001), caps2scalar_map(caps_pre_predict_fig, cc_pre, 0.00001),
                                                       compare_filter,
                                                       bb,
                                                       batchnorm_id='compare_fig', batchnorm_var_scope=batchnorm_var_scope, batchnorm_reuse=False))
    caps_compared_gnd = caps_compare(caps_pre_tiled, caps_pre_predict_gnd, cc_pre)
    caps_compared_gnd = tf.nn.relu(generic_map_combine(caps_compared_gnd,
                                                       caps2scalar_map(caps_pre_tiled, cc_pre, 0.00001), caps2scalar_map(caps_pre_predict_gnd, cc_pre, 0.00001),
                                                       compare_filter,
                                                       bb,
                                                       batchnorm_id='compare_gnd', batchnorm_var_scope=batchnorm_var_scope, batchnorm_reuse=False))

    # Separate fig and background caps & predict caps_pre with fig and ground caps_post
    fig_mask_labels = get_mask(labels_post, cc_post, scalar_to_caps=False) ######## save memory by using fig_mask_caps (and folding)
    labels_post_fig = labels_post*fig_mask_labels
    labels_post_gnd = labels_post*(1-fig_mask_labels)

    # predict labels_pre with fig and ground labels_post
    labels_pre_tiled = tile_by_bb(labels_pre, bb)
    labels_pre_predict_fig = tf.nn.relu(bundle_switched_transpose_conv(labels_post_fig, labels_pre_tiled, labels_filter, bb, strides=[1, 1, 1, 1],
                                                                       batchnorm_id='conv_labl_fig', batchnorm_var_scope=batchnorm_var_scope, batchnorm_reuse=False))
    labels_pre_predict_gnd = -tf.nn.relu(-bundle_switched_transpose_conv(labels_post_gnd, labels_pre_tiled, labels_filter, bb, strides=[1, 1, 1, 1],
                                                                         batchnorm_id='conv_labl_gnd', batchnorm_var_scope=batchnorm_var_scope, batchnorm_reuse=False))

    # biased WTA pruning of caps compared
    # BIAS COMPETITION WITH LABELS
    fig_ground_competition_bias = tf.nn.relu(fig_ground_competition_bias)
    bundle_competition_bias = tf.nn.relu(bundle_competition_bias)
    competition_bias = tf.concat([bundle_competition_bias*fig_ground_competition_bias[:,:,:,0],
                                  bundle_competition_bias*fig_ground_competition_bias[:,:,:,1]], axis=3)
    caps_compared = tf.concat([caps_compared_fig, caps_compared_gnd], axis=3)
    caps_compared_WTA = softmax_over_bb(caps_compared, bb*2, competition_bias=competition_bias, scale_afterwards=True)


    # weight labels_preds with caps_compared and combine
    labels_pre_predict = tf.concat([labels_pre_predict_fig, labels_pre_predict_gnd], axis=3)
    labels_pre_predict *= caps_compared_WTA
    labels_pre_candidate = tf.nn.tanh(sum_over_bb(labels_pre_predict, bb * 2) * tf.nn.sigmoid(update_gain))

    # STRATEGY 1: DEL with labels_pre and WRITE with caps_pre
    # delete_gate = get_conv_and_mask(labels_pre, labels_deletegate_filter, square=False,
    #                              batchnorm_id='conv_label_delete', batchnorm_var_scope=batchnorm_var_scope, batchnorm_reuse=False)
    # write_gate = get_conv_and_mask(caps2scalar_map(caps_pre, cc_pre, 0.00001), labels_writegate_filter, square=False,
    #                              batchnorm_id='conv_label_write', batchnorm_var_scope=batchnorm_var_scope, batchnorm_reuse=False)
    # return labels_pre*(1-delete_gate) + tf.nn.tanh(labels_pre_candidate)*(write_gate)

    # STRATEGY 2: MIX with cc_pre and labels_pre
    concatenated = tf.concat([tf.nn.tanh(labels_pre), caps2scalar_map(caps_pre, cc_pre, 0.00001)], axis=3)
    mix_gate = get_conv_and_mask(concatenated, labels_deletegate_filter, square=False,
                                 batchnorm_id='conv_label_mix', batchnorm_var_scope=batchnorm_var_scope, batchnorm_reuse=False)
    return labels_pre*(1-mix_gate) + tf.nn.tanh(labels_pre_candidate)*(mix_gate)

    # STRATEGY 3: MIX with labels_pre
    # mix_gate = get_conv_and_mask(labels_pre, labels_deletegate_filter, square=False,
    #                              batchnorm_id='conv_label_mix', batchnorm_var_scope=batchnorm_var_scope, batchnorm_reuse=False)
    # return labels_pre*(1-mix_gate) + tf.nn.tanh(labels_pre_candidate)*(mix_gate)

#### MISC



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
        cc,
        eps=0.00001):
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
            filter_reshaped = tf.reshape(caps_filter, caps_shape_in[:2] + [caps_shape_in[2]/cc, cc] + [caps_shape_in[3]/cc, cc])
            return tf.sqrt(tf.reduce_sum(tf.square(filter_reshaped), axis=[3,5], keep_dims=False) + eps)
    elif cc==1:
        return caps_filter
    else:
        raise ValueError('caps2scalar_filter: cc should be at least 1')
