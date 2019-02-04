import tensorflow as tf


def batch(
        bottom,
        name,
        scale=True,
        center=True,
        fused=True,
        renorm=False,
        data_format='NHWC',
        reuse=False,
        renorm_decay=0.99,
        decay=0.999,
        training=True):
    if data_format == 'NHWC' or data_format == 'channels_last':
        axis = -1
    elif data_format == 'NCHW' or data_format == 'channels_first':
        axis = 1
    else:
        raise NotImplementedError(data_format)
    return tf.layers.batch_normalization(
        inputs=bottom,
        name=name,
        scale=scale,
        center=center,
        fused=fused,
        renorm=renorm,
        reuse=reuse,
        axis=axis,
        momentum=decay,
        renorm_momentum=renorm_decay,
        training=training)
