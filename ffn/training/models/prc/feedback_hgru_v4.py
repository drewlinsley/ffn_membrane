"""Contextual model with partial filters."""
import warnings
import numpy as np
import tensorflow as tf
import initialization
from pooling import max_pool3d


# Dependency for symmetric weight ops is in models/layers/ff.py
class hGRU(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            layer_name,
            num_in_feats,
            timesteps=3,
            h_repeat=2,
            hgru_dhw=[[3, 7, 7], [3, 5, 5]],
            hgru_k=[24, 32],
            ff_conv_dhw=[[1, 5, 5], [2, 5, 5], [2, 3, 3]],
            ff_conv_k=[32, 48, 64],
            ff_conv_strides=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
            ff_pool_dhw=[[1, 2, 2], [1, 2, 2], [1, 1, 1]],
            ff_pool_strides=[[1, 2, 2], [1, 2, 2], [2, 2, 2]],
            fb_mode='transpose',
            fb_dhw=[[1, 7, 7], [2, 7, 7], [3, 5, 5]],
            fb_k=[8, 16, 32],
            padding='SAME',
            batch_norm=True,
            aux=None,
            train=True):
        """Global initializations and settings."""
        self.in_k = num_in_feats
        self.timesteps = timesteps
        self.padding = padding
        self.train = train
        self.layer_name = layer_name
        self.fb_mode = fb_mode # 'transpose', 'replicate_n_transpose'
        self.h_repeat = h_repeat
        self.batch_norm=batch_norm

        # Sort through and assign the auxilliary variables
        default_vars = self.defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)

        # Kernel shapes
        self.ff_conv_dhw = ff_conv_dhw
        self.ff_conv_k = ff_conv_k
        self.ff_conv_strides = ff_conv_strides
        self.ff_pool_dhw = ff_pool_dhw
        self.ff_pool_strides = ff_pool_strides
        self.hgru_dhw = hgru_dhw
        self.hgru_k = [k for k in hgru_k]
        self.fb_dhw = fb_dhw
        self.fb_k = fb_k

        # Nonlinearities and initializations
        if isinstance(self.recurrent_nl, basestring):
            self.recurrent_nl = self.interpret_nl(self.recurrent_nl)

        self.param_initializer = {
            'moving_mean': tf.constant_initializer(0., dtype=self.dtype),
            'moving_variance': tf.constant_initializer(1., dtype=self.dtype),
            'gamma': tf.constant_initializer(0.1, dtype=self.dtype)
        }
        self.param_trainable = {
            'moving_mean': False,
            'moving_variance': False,
            'gamma': True
        }
        self.param_collections = {
            'moving_mean': None,  # [tf.GraphKeys.UPDATE_OPS],
            'moving_variance': None,  # [tf.GraphKeys.UPDATE_OPS],
            'gamma': None
        }

    def defaults(self):
        """A dictionary containing defaults for auxilliary variables.

        These are adjusted by a passed aux dict variable."""
        return {
            'dtype': tf.float32,
            'gate_bias_init': 'chronos',
            'train': True,
            'recurrent_nl': tf.nn.tanh,
            'gate_nl': tf.nn.sigmoid,
            'ff_nl': tf.nn.elu,
            'normal_initializer': True,
            'symmetric_weights': False,
            'symmetric_gate_weights': False,
            'adapation': True,
            'bn_reuse':None,
            'readout': 'fb',  # l2 or fb
            'include_pooling': True,
            'resize_kernel': tf.image.ResizeMethod.BILINEAR
        }

    def interpret_nl(self, nl_type):
        """Return activation function."""
        if nl_type == 'tanh':
            return tf.nn.tanh
        elif nl_type == 'relu':
            return tf.nn.relu
        elif nl_type == 'elu':
            return tf.nn.elu
        elif nl_type == 'selu':
            return tf.nn.selu
        elif nl_type == 'leaky_relu':
            return tf.nn.leaky_relu
        elif nl_type == 'hard_tanh':
            return lambda z: tf.maximum(tf.minimum(z, 1), 0)
        else:
            raise NotImplementedError(nl_type)

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def symmetric_weights(self, w, name):
        """Apply symmetric weight sharing."""
        conv_w_t = tf.transpose(w, (2, 3, 0, 1))
        conv_w_symm = 0.5 * (conv_w_t + tf.transpose(conv_w_t, (1, 0, 2, 3)))
        conv_w = tf.transpose(conv_w_symm, (2, 3, 0, 1), name=name)
        return conv_w

    def prepare_tensors(self):
        """ Prepare recurrent/forward weight matrices.
        (np.prod([h, w, k]) / 2) - k params in the surround filter
        """
        # FEEDFORWARD KERNELS
        lower_feats = self.in_k
        for idx, (higher_feats, ff_dhw) in enumerate(
                zip(self.ff_conv_k, self.ff_conv_dhw)):
            with tf.variable_scope('ff_%s' % idx):
                if idx < len(self.ff_conv_dhw)-1:
                    # last conv layer doesn't have spot weights
                    setattr(
                        self,
                        'ff_%s_spot_weights' % idx,
                        tf.get_variable(
                            name='spot_weights',
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=[1,1,1] + [lower_feats*3, lower_feats],
                                dtype=self.dtype,
                                uniform=self.normal_initializer),
                            trainable=True))
                setattr(
                    self,
                    'ff_%s_weights' % idx,
                    tf.get_variable(
                        name='weights',
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=ff_dhw + [lower_feats, higher_feats],
                            dtype=self.dtype,
                            uniform=self.normal_initializer),
                        trainable=True))
                setattr(
                    self,
                    'ff_%s_bias' % idx,
                    tf.get_variable(
                        name='bias',
                        dtype=self.dtype,
                        initializer=tf.ones([higher_feats], dtype=self.dtype),
                        trainable=True))
                lower_feats = higher_feats

        # FEEDBACK KERNELS
        lower_feats = self.in_k
        for idx, (higher_feats, fb_dhw) in enumerate(
                zip(self.fb_k, self.fb_dhw)):
            with tf.variable_scope('fb_%s' % idx):
                setattr(
                    self,
                    'fb_%s_weights' % idx,
                    tf.get_variable(
                        name='weights',
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=fb_dhw + [lower_feats, higher_feats],
                            dtype=self.dtype,
                            uniform=self.normal_initializer),
                        trainable=True))
                setattr(
                    self,
                    'fb_%s_bias' % idx,
                    tf.get_variable(
                        name='bias',
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=[lower_feats],
                            dtype=self.dtype,
                            uniform=self.normal_initializer),
                        trainable=True))
            lower_feats = higher_feats

        # HGRU KERNELS
        for idx in range(len(self.hgru_dhw)):
            with tf.variable_scope('hgru_%s' % idx):
                setattr(
                    self,
                    'hgru_%s_W' % idx,
                    tf.get_variable(
                        name='W',
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=self.hgru_dhw[idx] + [self.hgru_k[idx], self.hgru_k[idx]],
                            dtype=self.dtype,
                            uniform=self.normal_initializer),
                        trainable=True))
                g_shape = [1,1,1] + [self.hgru_k[idx]*3, self.hgru_k[idx]]
                setattr(
                    self,
                    'hgru_%s_gain_a_weights' % idx,
                    tf.get_variable(
                        name='gain_a_weights',
                        dtype=self.dtype,
                        trainable=True,
                        initializer=initialization.xavier_initializer(
                            shape=g_shape,
                            dtype=self.dtype,
                            uniform=self.normal_initializer,
                            mask=None)))
                setattr(
                    self,
                    'hgru_%s_gain_b_weights' % idx,
                    tf.get_variable(
                        name='gain_b_weights',
                        dtype=self.dtype,
                        trainable=True,
                        initializer=initialization.xavier_initializer(
                            shape=g_shape,
                            dtype=self.dtype,
                            uniform=self.normal_initializer,
                            mask=None)))
                m_shape = [1,1,1] + [self.hgru_k[idx]*3, self.hgru_k[idx]]
                setattr(
                    self,
                    'hgru_%s_mix_weights' % idx,
                    tf.get_variable(
                        name='mix_weights',
                        dtype=self.dtype,
                        trainable=True,
                        initializer=initialization.xavier_initializer(
                            shape=m_shape,
                            dtype=self.dtype,
                            uniform=self.normal_initializer,
                            mask=None)))

                # Gate bias
                bias_shape = [1, 1, 1, 1, self.hgru_k[idx]]
                if self.gate_bias_init == 'chronos':
                    bias_init = -tf.log(
                        tf.random_uniform(
                            bias_shape,
                            minval=1,
                            maxval=self.timesteps - 1,
                            dtype=self.dtype))
                else:
                    bias_init = tf.ones(bias_shape, dtype=self.dtype)
                setattr(
                    self,
                    'hgru_%s_gain_a_bias' % idx,
                    tf.get_variable(
                        name='gain_a_bias',
                        dtype=self.dtype,
                        trainable=True,
                        initializer=bias_init))
                setattr(
                    self,
                    'hgru_%s_gain_b_bias' % idx,
                    tf.get_variable(
                        name='gain_b_bias',
                        dtype=self.dtype,
                        trainable=True,
                        initializer=bias_init))
                if self.gate_bias_init == 'chronos':
                    bias_init = -bias_init
                else:
                    bias_init = tf.ones(bias_shape, dtype=self.dtype)
                setattr(
                    self,
                    'hgru_%s_mix_bias' % idx,
                    tf.get_variable(
                        name='mix_bias',
                        dtype=self.dtype,
                        trainable=True,
                        initializer=bias_init))

                # combination params
                setattr(
                    self,
                    'hgru_%s_alpha' % idx,
                    tf.get_variable(
                        name='alpha',
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=[1, 1, 1, self.hgru_k[idx]*3, self.hgru_k[idx]],
                            dtype=self.dtype,
                            uniform=self.normal_initializer,
                            mask=None)))
                setattr(
                    self,
                    'hgru_%s_mu' % idx,
                    tf.get_variable(
                        name='mu',
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=[1, 1, 1, 1, self.hgru_k[idx]],
                            dtype=self.dtype,
                            uniform=self.normal_initializer,
                            mask=None)))
                setattr(
                    self,
                    'hgru_%s_kappa' % idx,
                    tf.get_variable(
                        name='kappa',
                        dtype=self.dtype,
                        trainable=False,
                        initializer=initialization.xavier_initializer(
                            shape=[1, 1, 1, self.hgru_k[idx] * 3, self.hgru_k[idx]],
                            dtype=self.dtype,
                            uniform=self.normal_initializer,
                            mask=None)))

                setattr(
                    self,
                    'hgru_%s_omega' % idx,
                    tf.get_variable(
                        name='omega',
                        dtype=self.dtype,
                        trainable=False,
                        initializer=initialization.xavier_initializer(
                            shape=[1, 1, 1, 1, self.hgru_k[idx]],
                            dtype=self.dtype,
                            uniform=self.normal_initializer,
                            mask=None)))

                # adaptation params
                if self.adapation:
                    setattr(
                        self,
                        'eta_%s' % idx,
                        tf.get_variable(
                            name='eta1',
                            dtype=self.dtype,
                            initializer=tf.ones(
                                [self.timesteps+1], dtype=tf.float32)))
                    setattr(
                        self,
                        'eta2_%s' % idx,
                        tf.get_variable(
                            name='eta2',
                            dtype=self.dtype,
                            initializer=tf.ones(
                                [self.timesteps+1], dtype=tf.float32)))

                if self.bn_reuse:
                    # Make the batchnorm variables
                    scopes = ['g1_bn', 'g2_bn', 'c1_bn', 'c2_bn']
                    bn_vars = ['moving_mean', 'moving_variance', 'gamma']
                    for s in scopes:
                        with tf.variable_scope(s):
                            for v in bn_vars:
                                tf.get_variable(
                                    trainable=self.param_trainable[v],
                                    name=v,
                                    dtype=self.dtype,
                                    shape=[self.hgru_k[idx]],
                                    collections=self.param_collections[v],
                                    initializer=self.param_initializer[v])
                    self.param_initializer = None

    def resize_x_to_y(
            self,
            x,
            y,
            kernel,
            strides,
            mode='transpose'):
        """Resize activity x to the size of y using interpolation."""
        y_size = y.get_shape().as_list()
        if mode == 'resize':
            return tf.image.resize_images(
                x,
                y_size[:-1],
                kernel,
                align_corners=True)
        elif mode == 'transpose':
            resized = tf.nn.conv3d_transpose(
                value=x,
                filter=kernel,
                output_shape=y_size,
                strides=[1] + strides + [1],
                padding=self.padding,
                name='resize_x_to_y')
            return resized
        else:
            raise NotImplementedError(mode)

    def generic_combine(self, tensor1, tensor2, kernel, symmetric_weights=False):
        stacked = tf.concat([tensor1, tensor1 * tensor2, tensor2], axis=4)
        stacked = self.conv_3d_op(
            data=stacked,
            weights=kernel,
            strides=[1, 1, 1, 1, 1],
            symmetric_weights=symmetric_weights,
            dilations=[1, 1, 1, 1, 1],
            padding='SAME')
        return stacked

    def conv_3d_op(
            self,
            data,
            weights,
            strides,
            symmetric_weights=False,
            dilations=None,
            padding=None):
        """3D convolutions for hgru."""
        if padding is None:
            padding=self.padding
        if dilations is None:
            dilations = [1, 1, 1, 1, 1]
        w_shape = [int(w) for w in weights.get_shape()]
        if len(w_shape) > 1 and int(w_shape[-2]) > 1:
            # Full convolutions
            if symmetric_weights:
                g = tf.get_default_graph()
                with g.gradient_override_map({'Conv3D': 'SymmetricConv3D'}):
                    activities = tf.nn.conv3d(
                        data,
                        weights,
                        strides,
                        padding=padding)
                        # TODO (jk): removed dilations=dilations to accommodate r1.4
            else:
                activities = tf.nn.conv3d(
                    data,
                    weights,
                    strides,
                    padding=padding)
                    # TODO (jk): removed dilations=dilations to accommodate r1.4
        else:
            raise RuntimeError
        return activities

    def circuit_input(self, x, h2, fb, var_scope):
        """Calculate gain and inh horizontal activities."""
        with tf.variable_scope(var_scope, reuse=True):
            gain_a_kernels = tf.get_variable("gain_a_weights")
            gain_a_bias = tf.get_variable("gain_a_bias")
            gain_b_kernels = tf.get_variable("gain_b_weights")
            gain_b_bias = tf.get_variable("gain_b_bias")
            horizontal_kernels = tf.get_variable("W")
        ## COMPUTE g1a
        g1a_intermediate = self.generic_combine(
            x,
            tf.nn.elu(fb)+1,
            gain_a_kernels,
            symmetric_weights=False) + gain_a_bias
        g1a_intermediate = tf.contrib.layers.batch_norm(
            inputs=g1a_intermediate,
            scale=True,
            center=False,
            fused=True,
            renorm=False,
            param_initializers=self.param_initializer,
            updates_collections=None,
            reuse=self.bn_reuse,
            is_training=self.train)
        g1a = self.gate_nl(g1a_intermediate)
        h2_gated = h2*g1a
        ## COMPUTE g1b
        g1b_intermediate = self.generic_combine(
            h2,
            tf.nn.tanh(fb),
            gain_b_kernels,
            symmetric_weights=False) + gain_b_bias
        g1b_intermediate = tf.contrib.layers.batch_norm(
            inputs=g1b_intermediate,
            scale=True,
            center=False,
            fused=True,
            renorm=False,
            param_initializers=self.param_initializer,
            updates_collections=None,
            reuse=self.bn_reuse,
            is_training=self.train)
        g1b = self.gate_nl(g1b_intermediate)
        x_gated = x*g1b

        # Horizontal activities
        c1 = self.conv_3d_op(
            data=h2_gated,
            weights=horizontal_kernels,
            strides=[1, 1, 1, 1, 1],
            symmetric_weights=self.symmetric_weights,
            dilations=[1, 1, 1, 1, 1])
        return c1, x_gated

    def circuit_output(self, h1, fb, var_scope):
        """Calculate mix and exc horizontal activities."""
        with tf.variable_scope(var_scope, reuse=True):
            mix_kernels = tf.get_variable("mix_weights")
            mix_bias = tf.get_variable("mix_bias")
            horizontal_kernels = tf.get_variable("W")
        g2_intermediate = self.generic_combine(
            h1,
            tf.nn.elu(fb) + 1,
            mix_kernels,
            symmetric_weights=False) + mix_bias
        # with tf.variable_scope(
        #         '%s/g2_bn' % var_scope,
        #         reuse=self.bn_reuse) as scope:
        g2_intermediate = tf.contrib.layers.batch_norm(
            inputs=g2_intermediate,
            scale=True,
            center=False,
            fused=True,
            renorm=False,
            param_initializers=self.param_initializer,
            updates_collections=None,
            reuse=self.bn_reuse,
            is_training=self.train)
        g2 = self.gate_nl(g2_intermediate)

        # Horizontal activities
        c2 = self.conv_3d_op(
            data=h1,
            weights=horizontal_kernels,
            strides=[1, 1, 1, 1, 1],
            symmetric_weights=self.symmetric_weights,
            dilations=[1, 1, 1, 1, 1])
        return c2, g2

    def input_integration(self, x, c1, h2, var_scope):
        """Integration on the input."""
        with tf.variable_scope(var_scope, reuse=True):
            alpha = tf.get_variable("alpha")
            mu = tf.get_variable("mu")
        stacked = self.generic_combine(h2, c1, alpha) + mu
        return x - tf.nn.elu(stacked) + 1 #self.recurrent_nl(stacked)

    def output_integration(self, h1, c2, g2, h2, var_scope):
        """Integration on the output."""
        """Integration on the input."""
        with tf.variable_scope(var_scope, reuse=True):
            kappa = tf.get_variable("kappa")
            omega = tf.get_variable("omega")
        stacked = self.generic_combine(h1, c2, kappa) + omega
        stacked = self.recurrent_nl(stacked)
        return (g2 * h2) + ((1 - g2) * stacked)

    def hgru_ops(self, i0, x, h2, fb, var_scope):
        """hGRU body."""
        c1, x_gated = self.circuit_input(
            x = x,
            h2=h2,
            fb=fb,
            var_scope=var_scope)
        c1 = tf.contrib.layers.batch_norm(
            inputs=c1,
            scale=True,
            center=False,
            fused=True,
            renorm=False,
            param_initializers=self.param_initializer,
            updates_collections=None,
            reuse=self.bn_reuse,
            is_training=self.train)
        h1 = self.input_integration(
            x=x_gated,
            c1=c1,
            h2=h2,
            var_scope=var_scope)


        # Circuit output receives recurrent input h1
        c2, g2 = self.circuit_output(
            h1=h1,
            fb=fb,
            var_scope=var_scope)
        c2 = tf.contrib.layers.batch_norm(
            inputs=c2,
            scale=True,
            center=False,
            fused=True,
            renorm=False,
            param_initializers=self.param_initializer,
            updates_collections=None,
            reuse=self.bn_reuse,
            is_training=self.train)
        h2 = self.output_integration(
            h1=h1,
            c2=c2,
            g2=g2,
            h2=h2,
            var_scope=var_scope)

        return h1, h2


    def full(self, i0, x, l0_h2, l0_fb, l1_h2, l1_fb):
        """hGRU body.
        Take the recurrent h2 from a low level and imbue it with
        information froma high layer. This means to treat the lower
        layer h2 as the X and the higher layer h2 as the recurrent state.
        This will serve as I/E from the high layer along with feedback
        kernels.
        """

        # HGRU 0
        idx = 0
        if self.adapation:
            with tf.variable_scope('hgru_%s' % idx, reuse=True):
                eta1 = tf.get_variable("eta1")
                eta2 = tf.get_variable("eta2")
            e2 = tf.gather(eta2, i0, axis=-1)
            e1 = tf.gather(eta1, i0, axis=-1)
            l0_fb *= e2
        else:
            e1=1
        for i in range(self.h_repeat):
            l0_h2 *= e1
            _, l0_h2 = self.hgru_ops(
                i0=i0,
                x=x,
                h2=l0_h2,
                fb=l0_fb,
                var_scope = 'hgru_%s' % idx)
        if self.batch_norm:
            ff0 = tf.contrib.layers.batch_norm(
                inputs=l0_h2,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        else:
            ff0 = l0_h2


        # FEEDFORWARD 0
        idx = 0
        with tf.variable_scope('ff_%s' % idx, reuse=True):
            spot_weights = tf.get_variable("spot_weights")
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
        ff0 = self.generic_combine(
            x,
            ff0,
            spot_weights)
        if self.batch_norm:
            ff0 = tf.contrib.layers.batch_norm(
                inputs=ff0,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        ff0 = self.ff_nl(ff0) + 1
        ff0 = tf.nn.conv3d(
            input=ff0,
            filter=weights,
            strides=self.ff_conv_strides[idx],
            padding=self.padding)
        if self.batch_norm:
            ff0 = tf.contrib.layers.batch_norm(
                inputs=ff0,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        ff0 = tf.nn.bias_add(
            ff0,
            bias)
        ff0 = self.ff_nl(ff0) + 1
        # POOL
        if self.include_pooling:
            ff0 = max_pool3d(
                bottom=ff0,
                k=self.ff_pool_dhw[idx],
                s=self.ff_pool_strides[idx],
                name='ff_pool_%s' % idx)


        # HGRU 1
        idx = 1
        if self.adapation:
            with tf.variable_scope('hgru_%s' % idx, reuse=True):
                eta1 = tf.get_variable("eta1")
                eta2 = tf.get_variable("eta2")
            e2 = tf.gather(eta2, i0, axis=-1)
            e1 = tf.gather(eta1, i0, axis=-1)
            l1_fb *= e2
        else:
            e1=1
        for i in range(self.h_repeat):
            l1_h2 *= e1
            _, l1_h2 = self.hgru_ops(
                i0=i0,
                x=ff0,
                h2=l1_h2,
                fb=l1_fb,
                var_scope = 'hgru_%s' % idx)
        if self.batch_norm:
            ff1 = tf.contrib.layers.batch_norm(
                inputs=l1_h2,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        else:
            ff1 = l1_h2


        # FEEDFORWARD 1
        idx = 1
        with tf.variable_scope('ff_%s' % idx, reuse=True):
            spot_weights = tf.get_variable("spot_weights")
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
        ff1 = self.generic_combine(
            ff0,
            ff1,
            spot_weights)
        if self.batch_norm:
            ff1 = tf.contrib.layers.batch_norm(
                inputs=ff1,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        ff1 = self.ff_nl(ff1) + 1
        ff1 = tf.nn.conv3d(
            input=ff1,
            filter=weights,
            strides=self.ff_conv_strides[idx],
            padding=self.padding)
        if self.batch_norm:
            ff1 = tf.contrib.layers.batch_norm(
                inputs=ff1,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        ff1 = tf.nn.bias_add(
            ff1,
            bias)
        ff1 = self.ff_nl(ff1) + 1
        # POOL
        if self.include_pooling:
            ff1 = max_pool3d(
                bottom=ff1,
                k=self.ff_pool_dhw[idx],
                s=self.ff_pool_strides[idx],
                name='ff_pool_%s' % idx)


        # FEEDFORWARD 2
        idx = 2
        with tf.variable_scope('ff_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
        ff2 = tf.nn.conv3d(
            input=ff1,
            filter=weights,
            strides=self.ff_conv_strides[idx],
            padding=self.padding)
        if self.batch_norm:
            ff2 = tf.contrib.layers.batch_norm(
                inputs=ff2,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        ff2 = tf.nn.bias_add(
            ff2,
            bias)
        ff2 = self.ff_nl(ff2) + 1
        # POOL
        if self.include_pooling:
            ff2 = max_pool3d(
                bottom=ff2,
                k=self.ff_pool_dhw[idx],
                s=self.ff_pool_strides[idx],
                name='ff_pool_%s' % idx)


        # FEEDBACK 2
        idx=2
        with tf.variable_scope('fb_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
        fb2 = self.resize_x_to_y(x=ff2, y=ff1,
                                  kernel=weights,
                                  mode=self.fb_mode,
                                  strides=self.ff_pool_strides[2])
        if self.batch_norm:
            # with tf.variable_scope('fb_bn' % 2,
            #         reuse=self.bn_reuse) as scope:
            fb2 = tf.contrib.layers.batch_norm(
                inputs=fb2,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        fb2 = tf.nn.bias_add(
            fb2,
            bias)
        fb2 = self.ff_nl(fb2) + 1


        # FEEDBACK 1
        idx=1
        with tf.variable_scope('fb_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
        fb1 = self.resize_x_to_y(x=fb2, y=ff0,
                                  kernel=weights,
                                  mode=self.fb_mode,
                                  strides=self.ff_pool_strides[1])
        if self.batch_norm:
            # with tf.variable_scope('fb_bn' % 2,
            #         reuse=self.bn_reuse) as scope:
            fb1 = tf.contrib.layers.batch_norm(
                inputs=fb1,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        fb1 = tf.nn.bias_add(
            fb1,
            bias)
        l1_fb = fb1
        fb1 = self.ff_nl(fb1) + 1


        # FEEDBACK 0
        idx=0
        with tf.variable_scope('fb_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
        fb0 = self.resize_x_to_y(x=fb1, y=x,
                                  kernel=weights,
                                  mode=self.fb_mode,
                                  strides=self.ff_pool_strides[0])
        if self.batch_norm:
            # with tf.variable_scope('fb_bn' % 2,
            #         reuse=self.bn_reuse) as scope:
            fb0 = tf.contrib.layers.batch_norm(
                inputs=fb0,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        fb0 = tf.nn.bias_add(
            fb0,
            bias)
        l0_fb = fb0

        # Iterate loop
        i0 += 1
        return i0, x, l0_h2, l0_fb, l1_h2, l1_fb


    def condition(self, i0, x, l0_h2, l0_fb, l1_h2, l1_fb):
        """While loop halting condition."""
        return i0 < self.timesteps

    def compute_shape(self, in_length, stride):
        if in_length % stride == 0:
            return in_length/stride
        else:
            return in_length/stride + 1

    def build(self, x, seed):
        """Run the backprop version of the Circuit."""
        self.prepare_tensors()
        i0 = tf.constant(0)

        # Calculate l2 hidden state size
        x_shape = x.get_shape().as_list()
        if self.include_pooling:
            l0_shape = x_shape
            l1_shape = [
                    x_shape[0],
                    self.compute_shape(l0_shape[1], self.ff_pool_strides[0][0]),
                    self.compute_shape(l0_shape[2], self.ff_pool_strides[0][1]),
                    self.compute_shape(l0_shape[3], self.ff_pool_strides[0][2]),
                    self.ff_conv_k[0]]
        else:
            l0_shape = tf.identity(x_shape)
            l1_shape = tf.identity(x_shape)

        # Initialize hidden layer activities
        l0_h2 = tf.ones(l0_shape, dtype=self.dtype)*seed*2 - 1
        l0_fb = tf.zeros(l0_shape, dtype=self.dtype)
        l1_h2 = tf.zeros(l1_shape, dtype=self.dtype)
        l1_fb = tf.zeros(l1_shape, dtype=self.dtype)

        # While loop
        elems = [
            i0,
            x,
            l0_h2,
            l0_fb,
            l1_h2,
            l1_fb]

        returned = tf.while_loop(
            self.condition,
            self.full,
            loop_vars=elems,
            back_prop=True,
            swap_memory=False)

        # Prepare output
        i0, x, l0_h2, l1_h2, l2_h2, fb_h2 = returned

        return l0_h2
