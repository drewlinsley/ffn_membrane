import warnings
import numpy as np
import tensorflow as tf
import initialization
import gradients
from pooling import max_pool

"""
HGRU MODEL. CVPR VERSION.
CAN SWITCH BTW 2D and 3D
WITH EXTRA CUSTOMIZABILITY
"""

class hGRU(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            var_scope,
            fsiz,
            num_channels,
            use_3d=False,
            symmetric_conv_weights=True,
            bistream_weights='shared',
            in_place_integration=True,
            soft_coefficients=False,
            h1_nl=tf.nn.tanh,
            h2_nl=tf.nn.tanh,
            gate_nl=tf.nn.sigmoid,
            bn_reuse=False,
            train=True,
            train_bn=True,
            bn_decay=0.95,
            dtype=tf.bfloat16):

        # Structural
        self.train = train
        self.train_bn = train_bn
        self.bn_decay = bn_decay
        self.var_scope = var_scope
        self.bn_reuse=bn_reuse
        self.symmetric_weights= symmetric_conv_weights
        if (bistream_weights is not 'independent') & (bistream_weights is not 'shared') & (bistream_weights is not 'symmetric'):
            raise ValueError('bistreaam_weights should be independent, shared or symmetric')
        self.bistream_weights = bistream_weights
        self.in_place_integration = in_place_integration
        self.num_channels=num_channels
        self.gate_nl=gate_nl
        self.dtype = dtype

        # RF Hyperparams
        self.use_3d = use_3d
        self.h1_nl = h1_nl
        self.h2_nl = h2_nl
        self.soft_coefficients = soft_coefficients
        self.fsiz = fsiz

        print('>>>>>>>>>>>>>>>>>>>>>>IS_TRAINING: ' + str(self.train))

    def prepare_tensors(self):
        local_shape = [1, 1] if not self.use_3d else [1, 1, 1]
        self.bn_param_initializer = {
                            'moving_mean': tf.constant_initializer(0., dtype=self.dtype),
                            'moving_variance': tf.constant_initializer(1., dtype=self.dtype),
                            'beta': tf.constant_initializer(0., dtype=self.dtype),
                            'gamma': tf.constant_initializer(0.1, dtype=self.dtype)
        }
        with tf.variable_scope(self.var_scope):
            tf.get_variable(
                name='h1_w',
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.fsiz + [self.num_channels, self.num_channels],
                    dtype=self.dtype,
                    uniform=True),
                trainable=self.train)
            tf.get_variable(
                name='g1_w',
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=local_shape + [self.num_channels, self.num_channels],
                    dtype=self.dtype,
                    uniform=True),
                trainable=self.train)
            tf.get_variable(
                name='g1_b',
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=[1] + local_shape + [self.num_channels],
                    dtype=self.dtype,
                    uniform=True),
                trainable=self.train)
            tf.get_variable(
                name='alpha',
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=[1] + local_shape + [1],
                    dtype=self.dtype,
                    uniform=True),
                trainable=self.train)
            tf.get_variable(
                name='mu',
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=[1] + local_shape + [self.num_channels if self.soft_coefficients else 1],
                    dtype=self.dtype,
                    uniform=True),
                trainable=self.train)

            if self.bistream_weights is 'independent':
                tf.get_variable(
                    name='h2_w',
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=self.fsiz + [self.num_channels, self.num_channels],
                        dtype=self.dtype,
                        uniform=True),
                    trainable=self.train)
            tf.get_variable(
                name='g2_w',
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=local_shape + [self.num_channels, self.num_channels],
                    dtype=self.dtype,
                    uniform=True),
                trainable=self.train)
            tf.get_variable(
                name='g2_b',
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=[1] + local_shape + [self.num_channels],
                    dtype=self.dtype,
                    uniform=True),
                trainable=self.train)
            tf.get_variable(
                name='kappa',
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=[1] + local_shape + [1],
                    dtype=self.dtype,
                    uniform=True),
                trainable=self.train)
            tf.get_variable(
                name='gamma',
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=[1] + local_shape + [self.num_channels if self.soft_coefficients else 1],
                    dtype=self.dtype,
                    uniform=True),
                trainable=self.train)
            tf.get_variable(
                name='omega',
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=[1] + local_shape + [self.num_channels if self.soft_coefficients else 1],
                    dtype=self.dtype,
                    uniform=True),
                trainable=self.train)

            if self.bn_reuse:
                # Make the batchnorm variables
                scopes = ['c1/bn', 'c2/bn', 'g1/bn', 'g1/bn']
                shapes = [self.num_channels, self.num_channels, self.num_channels, self.num_channels]
                bn_vars = ['moving_mean', 'moving_variance', 'beta', 'gamma']
                for (scp, shp) in zip(scopes, shapes):
                    with tf.variable_scope(scp):
                        for v in bn_vars:
                            tf.get_variable(
                                trainable=self.train,
                                name=v,
                                dtype=self.dtype,
                                shape=[shp],
                                initializer=self.bn_param_initializer)

    def symmetrize_weights(self, w):
        """Apply symmetric weight sharing."""
        if self.use_3d:
            conv_w_flipped = tf.reverse(w, [0, 1, 2])
        else:
            conv_w_flipped = tf.reverse(w, [0, 1])
        conv_w_symm = 0.5 * (conv_w_flipped + w)
        return conv_w_symm

    def conv_op(
            self,
            data,
            weights,
            strides=None,
            deconv_size=None,
            symmetric_weights=False,
            padding=None):
        """3D convolutions for hgru."""
        if padding is None:
            padding='SAME'
        if strides is None:
            if self.use_3d:
                strides = [1, 1, 1, 1, 1]
            else:
                strides = [1, 1, 1, 1]
        if symmetric_weights:
            weights = self.symmetrize_weights(weights)
        if deconv_size is None:
            if self.use_3d:
                activities = tf.nn.conv3d(
                                    data,
                                    weights,
                                    strides=strides,
                                    padding=padding)
            else:
                activities = tf.nn.conv2d(
                                    data,
                                    weights,
                                    strides=strides,
                                    padding=padding)
        else:
            if self.use_3d:
                activities = tf.nn.conv3d_transpose(
                                    data,
                                    weights,
                                    output_shape=deconv_size,
                                    strides=strides,
                                    padding=padding)
            else:
                activities = tf.nn.conv2d_transpose(
                                    data,
                                    weights,
                                    output_shape=deconv_size,
                                    strides=strides,
                                    padding=padding)
        return activities

    def compute_h1(self, h2, x, gate_weights, gate_bias, conv_weights, combine_coeffs):
        # COMPUTE G1
        g1 = self.conv_op(h2,
                          gate_weights,
                           strides=None,
                           symmetric_weights=False,
                           deconv_size=h2.get_shape().as_list(),
                           padding=None)
        g1 += gate_bias
        if self.bn_reuse:
            with tf.variable_scope(
                    '%s_g1/bn' % (self.var_scope),
                    reuse=tf.AUTO_REUSE) as scope:
                g1 = tf.contrib.layers.batch_norm(
                    inputs=g1,
                    scale=True,
                    center=False,
                    fused=True,
                    renorm=False,
                    reuse=tf.AUTO_REUSE,
                    scope=scope,
                    decay=self.bn_decay,
                    updates_collections=None,
                    is_training=self.train_bn)
        else:
            g1 = tf.contrib.layers.batch_norm(
                inputs=g1,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.bn_param_initializer,
                decay=self.bn_decay,
                updates_collections=None,
                is_training=self.train_bn)
        g1 = self.gate_nl(g1)

        # COMPUTE C1
        if self.bistream_weights is 'symmetric':
            c1 = self.conv_op(h2*g1,
                     conv_weights,
                     strides=None,
                     symmetric_weights=self.symmetric_weights,
                     deconv_size=x.get_shape().as_list(),
                     padding=None)
        else:
            c1 = self.conv_op(h2*g1,
                     conv_weights,
                     strides=None,
                     symmetric_weights=self.symmetric_weights,
                     deconv_size=None,
                     padding=None)

        if self.bn_reuse:
            with tf.variable_scope(
                    '%s_c1/bn' % (self.var_scope),
                    reuse=tf.AUTO_REUSE) as scope:
                c1 = tf.contrib.layers.batch_norm(
                    inputs=c1,
                    scale=True,
                    center=False,
                    fused=True,
                    renorm=False,
                    reuse=tf.AUTO_REUSE,
                    scope=scope,
                    decay=self.bn_decay,
                    updates_collections=None,
                    is_training=self.train_bn)
        else:
            c1 = tf.contrib.layers.batch_norm(
                inputs=c1,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.bn_param_initializer,
                decay=self.bn_decay,
                updates_collections=None,
                is_training=self.train_bn)

        # COMBINE
        if self.in_place_integration:
            h1_candidate = x - combine_coeffs[0] * c1 * h2 - combine_coeffs[1] * c1
        else:
            h1_candidate = x - combine_coeffs[0] * c1 * x - combine_coeffs[1] * c1
        return self.h1_nl(h1_candidate)

    def compute_h2(self, h1, h2, gate_weights, gate_bias, conv_weights, combine_coeffs):
        # COMPUTE G2
        g2 = self.conv_op(h1,
                          gate_weights,
                          strides=None,
                          deconv_size=None,
                          symmetric_weights=False,
                          padding=None)
        g2 += gate_bias
        if self.bn_reuse:
            with tf.variable_scope(
                    '%s_g2/bn' % (self.var_scope),
                    reuse=tf.AUTO_REUSE) as scope:
                g2 = tf.contrib.layers.batch_norm(
                    inputs=g2,
                    scale=True,
                    center=False,
                    fused=True,
                    renorm=False,
                    reuse=tf.AUTO_REUSE,
                    scope=scope,
                    decay=self.bn_decay,
                    updates_collections=None,
                    is_training=self.train_bn)
        else:
            g2 = tf.contrib.layers.batch_norm(
                inputs=g2,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.bn_param_initializer,
                decay=self.bn_decay,
                updates_collections=None,
                is_training=self.train_bn)
        g2 = self.gate_nl(g2)

        # COMPUTE C2
        if self.bistream_weights == 'symmetric':
            c2 = self.conv_op(h1,
                     conv_weights,
                     strides=None,
                     deconv_size=h1.get_shape().as_list(),
                     symmetric_weights=self.symmetric_weights,
                     padding=None)
        else:
            c2 = self.conv_op(h1,
                     conv_weights,
                     strides=None,
                     deconv_size=None,
                     symmetric_weights=self.symmetric_weights,
                     padding=None)
        if self.bn_reuse:
            with tf.variable_scope(
                    '%s_c2/bn' % (self.var_scope),
                    reuse=tf.AUTO_REUSE) as scope:
                c2 = tf.contrib.layers.batch_norm(
                    inputs=c2,
                    scale=True,
                    center=False,
                    fused=True,
                    renorm=False,
                    reuse=tf.AUTO_REUSE,
                    scope=scope,
                    decay=self.bn_decay,
                    updates_collections=None,
                    is_training=self.train_bn)
        else:
            c2 = tf.contrib.layers.batch_norm(
                inputs=c2,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.bn_param_initializer,
                decay=self.bn_decay,
                updates_collections=None,
                is_training=self.train_bn)

        if self.in_place_integration:
            h2_candidate = combine_coeffs[0] * c2 + combine_coeffs[1] * c2 * h2 + combine_coeffs[2] * h2
        else:
            h2_candidate = combine_coeffs[0] * c2 + combine_coeffs[1] * c2 * h1 + combine_coeffs[2] * h1
        h2_candidate = self.h2_nl(h2_candidate)

        # COMBINE
        return h2_candidate*g2 + h2*(1-g2)

    def run(self, x, h2):
        """hGRU body."""
        # GET VARIABLES
        with tf.variable_scope(self.var_scope, reuse=True):
            if self.bistream_weights=='independent':
                h1_w = tf.get_variable("h1_w")
                h2_w = tf.get_variable("h2_w")
            else:
                h1_w = tf.get_variable("h1_w")
                h2_w = h1_w
            g1_w = tf.get_variable("g1_w")
            g1_b = tf.get_variable("g1_b")
            mu = tf.get_variable("mu")
            alpha = tf.get_variable("alpha")
            g2_w = tf.get_variable("g2_w")
            g2_b = tf.get_variable("g2_b")
            kappa = tf.get_variable("kappa")
            gamma = tf.get_variable("gamma")
            omega = tf.get_variable("omega")

        # Compute
        h1 = self.compute_h1(h2, x, g1_w, g1_b, h1_w, [mu, alpha])
        h2 = self.compute_h2(h1, h2, g2_w, g2_b, h2_w, [kappa, gamma, omega])
        return h2