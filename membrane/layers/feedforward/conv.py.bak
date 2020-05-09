import numpy as np
import tensorflow as tf
from membrane.layers.feedforward import normalization, pooling


def conv_layer(
        bottom,
        name,
        num_filters=None,
        kernel_size=None,
        stride=[1, 1, 1, 1],
        padding='SAME',
        trainable=True,
        use_bias=True,
        dtype=tf.float32,
        reuse=False,
        aux={}):
    """2D convolutional layer with pretrained weights."""
    in_ch = int(bottom.get_shape()[-1])
    if 'transpose_inds' in aux.keys():
        transpose_inds = aux['transpose_inds']
    else:
        transpose_inds = False
    if 'pretrained' in aux.keys():
        kernel_initializer = np.load(aux['pretrained']).item()
        key = aux['pretrained_key']
        if key == 'weights':
            key = kernel_initializer.keys()[0]
        kernel_initializer, preloaded_bias = kernel_initializer[key]
        if not len(preloaded_bias) and use_bias:
            bias = tf.get_variable(
                name='%s_conv_bias' % name,
                initializer=tf.zeros_initializer(),
                shape=[1, 1, 1, kernel_initializer.shape[-1]],
                trainable=trainable)
        if transpose_inds:
            kernel_initializer = kernel_initializer.transpose(transpose_inds)
        kernel_size = kernel_initializer.shape[0]
        pretrained = True
    else:
        assert num_filters is not None, 'Describe your filters'
        assert kernel_size is not None, 'Describe your kernel_size'
        if 'initializer' in aux.keys():
            kernel_initializer = aux['initializer']
        else:
            # kernel_initializer = tf.variance_scaling_initializer()
            kernel_initializer = [
                [kernel_size, kernel_size, in_ch, num_filters],
                tf.contrib.layers.xavier_initializer(uniform=False)]
        pretrained = False
    if pretrained:
        filters = tf.get_variable(
            name='%s_pretrained' % name,
            initializer=kernel_initializer,
            dtype=dtype,
            trainable=trainable)
    else:
        filters = tf.get_variable(
            name='%s_initialized' % name,
            shape=kernel_initializer[0],
            initializer=kernel_initializer[1],
            trainable=trainable)
        if use_bias:
            bias = tf.get_variable(
                name='%s_bias' % name,
                initializer=tf.zeros([1, 1, 1, num_filters]),
                trainable=trainable)
    activity = tf.nn.conv2d(
        bottom,
        filters,
        strides=stride,
        padding='SAME')
    if use_bias:
        activity += bias
    if 'nonlinearity' in aux.keys():
        if aux['nonlinearity'] == 'square':
            activity = tf.pow(activity, 2)
        elif aux['nonlinearity'] == 'relu':
            activity = tf.nn.relu(activity)
        elif aux['nonlinearity'] == 'elu':
            activity = tf.nn.elu(activity)
        else:
            raise NotImplementedError(aux['nonlinearity'])
    return activity


def conv3d_layer(
        bottom,
        name,
        num_filters=None,
        kernel_size=None,
        stride=[1, 1, 1],
        padding='SAME',
        trainable=True,
        dtype=tf.float32,
        use_bias=True,
        reuse=False,
        aux={}):
    """3D convolutional layer."""
    activity = tf.layers.conv3d(
        inputs=bottom,
        filters=num_filters,
        kernel_size=kernel_size,
        strides=stride,
        # dtype=dtype,
        padding=padding,
        use_bias=use_bias)
    if 'nonlinearity' in aux.keys():
        if aux['nonlinearity'] == 'square':
            activity = tf.pow(activity, 2)
        elif aux['nonlinearity'] == 'relu':
            activity = tf.nn.relu(activity)
        elif aux['nonlinearity'] == 'elu':
            activity = tf.nn.elu(activity)
        else:
            raise NotImplementedError(aux['nonlinearity'])
    return activity


def conv3d_transpose_layer(
        bottom,
        name,
        num_filters=None,
        kernel_size=None,
        stride=[1, 1, 1],
        padding='SAME',
        trainable=True,
        use_bias=True,
        reuse=False,
        aux={}):
    """3D convolutional layer."""
    activity = tf.layers.conv3d_transpose(
        inputs=bottom,
        filters=num_filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        use_bias=use_bias)
    if 'nonlinearity' in aux.keys():
        if aux['nonlinearity'] == 'square':
            activity = tf.pow(activity, 2)
        elif aux['nonlinearity'] == 'relu':
            activity = tf.nn.relu(activity)
        elif aux['nonlinearity'] == 'elu':
            activity = tf.nn.elu(activity)
        else:
            raise NotImplementedError(aux['nonlinearity'])
    return activity


def down_block(
        layer_name,
        bottom,
        reuse,
        kernel_size,
        num_filters,
        training,
        use_bias=False,
        include_pool=True):
    """Forward block for seung model."""
    with tf.variable_scope('%s_block' % layer_name, reuse=reuse):
        with tf.variable_scope('%s_layer_1' % layer_name, reuse=reuse):
            x = conv3d_layer(
                bottom=bottom,
                name='%s_1' % layer_name,
                stride=[1, 1, 1],
                padding='SAME',
                kernel_size=kernel_size[0],
                num_filters=num_filters,
                trainable=training,
                use_bias=use_bias)
            x = normalization.batch(
                bottom=x,
                name='%s_bn_1' % layer_name,
                training=training)
            x = tf.nn.elu(x)
            skip = tf.identity(x)

        with tf.variable_scope('%s_layer_2' % layer_name, reuse=reuse):
            x = conv3d_layer(
                bottom=x,
                name='%s_2' % layer_name,
                stride=[1, 1, 1],
                padding='SAME',
                kernel_size=kernel_size[1],
                num_filters=num_filters,
                trainable=training,
                use_bias=use_bias)
            x = normalization.batch(
                bottom=x,
                name='%s_bn_2' % layer_name,
                training=training)
            x = tf.nn.elu(x)

        with tf.variable_scope('%s_layer_3' % layer_name, reuse=reuse):
            x = conv3d_layer(
                bottom=x,
                name='%s_3' % layer_name,
                stride=[1, 1, 1],
                padding='SAME',
                kernel_size=kernel_size[2],
                num_filters=num_filters,
                trainable=training,
                use_bias=use_bias)
            x = tf.nn.elu(x)
            x = x + skip
            x = normalization.batch(
                bottom=x,
                name='%s_bn_3' % layer_name,
                training=training)

        if include_pool:
            with tf.variable_scope('%s_pool' % layer_name, reuse=reuse):
                x = pooling.max_pool3d(
                    bottom=x,
                    name='%s_pool' % layer_name,
                    k=[1, 2, 2],
                    s=[1, 2, 2])
    return x


def up_block(
        layer_name,
        bottom,
        skip_activity,
        reuse,
        kernel_size,
        num_filters,
        training,
        stride=[1, 2, 2],
        use_bias=False):
    """Forward block for seung model."""
    with tf.variable_scope('%s_block' % layer_name, reuse=reuse):
        with tf.variable_scope('%s_layer_1' % layer_name, reuse=reuse):
            x = conv3d_transpose_layer(
                bottom=bottom,
                name='%s_1' % layer_name,
                stride=stride,
                padding='SAME',
                num_filters=num_filters,
                kernel_size=kernel_size,
                trainable=training,
                use_bias=use_bias)
            x = x + skip_activity  # Rethink if this is valid
            x = normalization.batch(
                bottom=x,
                name='%s_bn_1' % layer_name,
                training=training)
            x = tf.nn.elu(x)
    return x
