import tensorflow as tf


def max_pool(
        bottom,
        name,
        k=[1, 2, 2, 1],
        s=[1, 2, 2, 1],
        p='SAME',
        data_format='NHWC'):
    """Local max pooling."""
    return tf.nn.max_pool(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        data_format=data_format,
        name=name)


def global_pool(
        bottom,
        name,
        aux,
        data_format='NHWC'):
    """Global avg pool. TODO: move to pool class."""
    if 'reduction_indices' in aux:
        reduction_indices = aux['reduction_indices']
    else:
        if data_format == 'NHWC':
            reduction_indices = [1, 2]
        elif data_format == 'NCHW':
            reduction_indices = [2, 3]
    if 'pool_type' in aux:
        pool_type = aux['pool_type']
    else:
        pool_type = 'max'
    if pool_type == 'average' or pool_type == 'mean':
        activity = tf.reduce_mean(
            bottom,
            reduction_indices=reduction_indices)
    elif pool_type == 'max':
        activity = tf.reduce_max(
            bottom,
            reduction_indices=reduction_indices)
    else:
        raise NotImplementedError('Cannot understand %s' % pool_type)
    return activity
