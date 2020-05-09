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
            hgru_dhw=[[3, 7, 7], [3, 5, 5], [1, 3, 3], [1, 1, 1]],
            hgru_k=[8, 16, 32, 8],
            ff_conv_dhw=[[1, 1, 1], [2, 5, 5], [2, 3, 3], [1, 3, 3]],
            ff_conv_k=[8, 16, 32, 48],
            ff_conv_strides=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
            ff_pool_dhw=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
            ff_pool_strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
            fb_mode = 'transpose',
            fb_dhw=[[3, 6, 6], [3, 4, 4], [1, 4, 4]],
            fb_k=[16, 32, 48],
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
                setattr(
                    self,
                    'ff_%s_weights' % idx,
                    tf.get_variable(
                        name='weights',
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=ff_dhw + [lower_feats*3, higher_feats],
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
                g_shape = [1,1,1] + [self.hgru_k[idx], self.hgru_k[idx]]
                setattr(
                    self,
                    'hgru_%s_gain_weights' % idx,
                    tf.get_variable(
                        name='gain_weights',
                        dtype=self.dtype,
                        trainable=True,
                        initializer=initialization.xavier_initializer(
                            shape=g_shape,
                            dtype=self.dtype,
                            uniform=self.normal_initializer,
                            mask=None)))
                m_shape = [1,1,1] + [self.hgru_k[idx], self.hgru_k[idx]]
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
                    'hgru_%s_gain_bias' % idx,
                    tf.get_variable(
                        name='gain_bias',
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
                            name='eta',
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
            bias,
            strides,
            mode='transpose',
            use_bias=True):
        """Resize activity x to the size of y using interpolation."""
        y_size = y.get_shape().as_list()
        if mode == 'resize':
            return tf.image.resize_images(
                x,
                y_size[:-1],
                kernel,
                align_corners=True)
        elif mode == 'transpose':
            # strides = np.asarray(self.pool_strides)
            # strides[1:] *= len(self.ff_conv_k)
            # kernels = np.asarray(self.pooling_kernel)
            # kernels[1:] *= len(self.ff_conv_k)
            # return tf.layers.conv3d_transpose(
            #     inputs=x,
            #     strides=strides,
            #     padding=self.padding,
            #     filters=y_size[-1],
            #     kernel_size=kernels,
            #     trainable=self.train,
            #     use_bias=use_bias,
            #     activation=self.ff_nl)
            resized = tf.nn.conv3d_transpose(
                value=x,
                filter=kernel,
                output_shape=y_size,
                strides=[1] + strides + [1],
                padding=self.padding,
                name='resize_x_to_y')
            resized = tf.nn.bias_add(
                resized,
                bias)
            resized = self.ff_nl(resized)
            return resized
        elif mode == 'replicate_n_transpose':
            resized = tf.image.resize_images(
                x,
                y_size[:-1],
                kernel,
                align_corners=False)
            resized = tf.nn.conv3d_transpose(
                value=resized,
                filter=kernel,
                output_shape=y_size,
                strides=[1, 1, 1, 1, 1],
                padding='SAME',
                name='resize_x_to_y')
            resized = tf.nn.bias_add(
                resized,
                bias)
            resized = self.ff_nl(resized)
            return resized
        else:
            raise NotImplementedError(mode)

    def conv_3d_op(
            self,
            data,
            weights,
            strides,
            symmetric_weights=False,
            dilations=None):
        """3D convolutions for hgru."""
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
                        padding=self.padding)
                        # TODO (jk): removed dilations=dilations to accommodate r1.4
            else:
                activities = tf.nn.conv3d(
                    data,
                    weights,
                    strides,
                    padding=self.padding)
                    # TODO (jk): removed dilations=dilations to accommodate r1.4
        else:
            raise RuntimeError
        return activities

    def circuit_input(self, h2, var_scope):
        """Calculate gain and inh horizontal activities."""
        with tf.variable_scope(var_scope, reuse=True):
            gain_kernels = tf.get_variable("gain_weights")
            gain_bias = tf.get_variable("gain_bias")
            horizontal_kernels = tf.get_variable("W")
        g1_intermediate = self.conv_3d_op(
            data=h2,
            weights=gain_kernels,
            strides=[1, 1, 1, 1, 1],
            symmetric_weights=self.symmetric_gate_weights,
            dilations=[1, 1, 1, 1, 1])
        # with tf.variable_scope(
        #         '%s/g1_bn' % var_scope) as scope:
        g1_intermediate = tf.contrib.layers.batch_norm(
            inputs=g1_intermediate + gain_bias,
            scale=True,
            center=False,
            fused=True,
            renorm=False,
            param_initializers=self.param_initializer,
            updates_collections=None,
            reuse=self.bn_reuse,
            is_training=self.train)
        g1 = self.gate_nl(g1_intermediate)
        h2 *= g1

        # Horizontal activities
        c1 = self.conv_3d_op(
            data=h2,
            weights=horizontal_kernels,
            strides=[1, 1, 1, 1, 1],
            symmetric_weights=self.symmetric_weights,
            dilations=[1, 1, 1, 1, 1])
        return c1, g1

    def circuit_output(self, h1, var_scope):
        """Calculate mix and exc horizontal activities."""
        with tf.variable_scope(var_scope, reuse=True):
            mix_kernels = tf.get_variable("mix_weights")
            mix_bias = tf.get_variable("mix_bias")
            horizontal_kernels = tf.get_variable("W")
        g2_intermediate = self.conv_3d_op(
            data=h1,
            weights=mix_kernels,
            strides=[1, 1, 1, 1, 1],
            symmetric_weights=self.symmetric_gate_weights,
            dilations=[1, 1, 1, 1, 1])

        # with tf.variable_scope(
        #         '%s/g2_bn' % var_scope,
        #         reuse=self.bn_reuse) as scope:
        g2_intermediate = tf.contrib.layers.batch_norm(
            inputs=g2_intermediate + mix_bias,
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
        stacked = tf.concat([x, x*c1, c1], axis=4)
        stacked = tf.nn.conv3d(
            stacked,
            alpha,
            [1,1,1,1,1],
            padding='SAME') + mu
        return self.recurrent_nl(stacked)

    def output_integration(self, h1, c2, g2, h2, var_scope):
        """Integration on the output."""
        """Integration on the input."""
        with tf.variable_scope(var_scope, reuse=True):
            kappa = tf.get_variable("kappa")
            omega = tf.get_variable("omega")
        stacked = tf.concat([h1, h1*c2, c2], axis=4)
        stacked = tf.nn.conv3d(
            stacked,
            kappa,
            [1,1,1,1,1],
            padding='SAME') + omega
        stacked = self.recurrent_nl(stacked)
        return (g2 * h2) + ((1 - g2) * stacked)

    def hgru_ops(self, i0, x, h2, var_scope, layer_idx):
        """hGRU body."""
        c1, g1 = self.circuit_input(
            h2=h2,
            var_scope=var_scope)

        # Circuit input receives recurrent output h2
        # with tf.variable_scope(
        #         '%s/c1_bn' % var_scope,
        #         reuse=self.bn_reuse) as scope:
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

        # Calculate input (-) integration: h1 (4)

        h1 = self.input_integration(
            x=x,
            c1=c1,
            h2=h2,
            var_scope=var_scope)


        # Circuit output receives recurrent input h1
        c2, g2 = self.circuit_output(
            h1=h1,
            var_scope=var_scope)

        # with tf.variable_scope(
        #         '%s/c2_bn' % var_scope,
        #         reuse=self.bn_reuse) as scope:
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

        # Calculate output (+) integration: h2 (8, 9)
        h2 = self.output_integration(
            h1=h1,
            c2=c2,
            g2=g2,
            h2=h2,
            var_scope=var_scope)

        if self.adapation:
            eta = getattr(self, 'eta_%s' % layer_idx)
            e = tf.gather(eta, i0, axis=-1)
            h2 *= e

        return h1, h2


    def cap(self, x, fb_h2, l0_h2):
        # Intermediate FF - h0
        idx = 0
        if self.adapation:
            eta2 = getattr(self, 'eta2_%s' % idx)
            e2 = tf.gather(eta2, self.timesteps, axis=-1)
            fb_h2_processed = fb_h2*e2
        with tf.variable_scope('ff_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
            processed_x = tf.nn.conv3d(
                input=tf.concat([x, fb_h2_processed*x, fb_h2_processed], axis=4),
                filter=weights,
                strides=self.ff_conv_strides[idx],
                padding=self.padding)
            processed_x = tf.nn.bias_add(
                processed_x,
                bias)
            processed_x = self.ff_nl(processed_x)
        # if self.include_pooling:
        #     processed_x = max_pool3d(
        #         bottom=processed_x,
        #         k=self.ff_pool_dhw[idx],
        #         s=self.ff_pool_strides[idx],
        #         name='ff_pool_%s' % 0)
        if self.batch_norm:
            # with tf.variable_scope('ff_bn_%s' % idx,
            #         reuse=self.bn_reuse) as scope:
            processed_x = tf.contrib.layers.batch_norm(
                inputs=processed_x,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        for i in range(self.h_repeat):
            _, l0_h2 = self.hgru_ops(
                i0=self.timesteps,
                x=processed_x,
                h2=l0_h2,
                var_scope = 'hgru_%s' % idx,
                layer_idx=idx)

        return l0_h2


    def full(self, i0, x, l0_h2, l1_h2, l2_h2, fb_h2):
        """hGRU body.
        Take the recurrent h2 from a low level and imbue it with
        information froma high layer. This means to treat the lower
        layer h2 as the X and the higher layer h2 as the recurrent state.
        This will serve as I/E from the high layer along with feedback
        kernels.
        """

        # Intermediate FF - h0
        idx = 0
        if self.adapation:
            eta2 = getattr(self, 'eta2_%s' % idx)
            e2 = tf.gather(eta2, i0, axis=-1)
            fb_h2_processed = fb_h2*e2
        with tf.variable_scope('ff_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
            processed_x = tf.nn.conv3d(
                input=tf.concat([x, fb_h2_processed*x, fb_h2_processed], axis=4),
                filter=weights,
                strides=self.ff_conv_strides[idx],
                padding=self.padding)
            processed_x = tf.nn.bias_add(
                processed_x,
                bias)
            processed_x = self.ff_nl(processed_x)
        # if self.include_pooling:
        #     processed_x = max_pool3d(
        #         bottom=processed_x,
        #         k=self.ff_pool_dhw[idx],
        #         s=self.ff_pool_strides[idx],
        #         name='ff_pool_%s' % 0)
        if self.batch_norm:
            # with tf.variable_scope('ff_bn_%s' % idx,
            #         reuse=self.bn_reuse) as scope:
            processed_x = tf.contrib.layers.batch_norm(
                inputs=processed_x,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        for i in range(self.h_repeat):
            _, l0_h2 = self.hgru_ops(
                i0=i0,
                x=processed_x,
                h2=l0_h2,
                var_scope = 'hgru_%s' % idx,
                layer_idx=idx)


        #Intermediate FF - h1
        idx = 1
        if self.adapation:
            eta2 = getattr(self, 'eta2_%s' % idx)
            e2 = tf.gather(eta2, i0, axis=-1)
            l0_h2_processed = l0_h2*e2
        with tf.variable_scope('ff_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
            processed_l0 = tf.nn.conv3d(
                input=tf.concat([processed_x, l0_h2_processed*processed_x, l0_h2_processed], axis=4),
                filter=weights,
                strides=self.ff_conv_strides[idx],
                padding=self.padding)
            processed_l0 = tf.nn.bias_add(
                processed_l0,
                bias)
            processed_l0 = self.ff_nl(processed_l0)
        # Pool the preceding layer's drive
        if self.include_pooling:
            processed_l0 = max_pool3d(
                bottom=processed_l0,
                k=self.ff_pool_dhw[idx],
                s=self.ff_pool_strides[idx],
                name='ff_pool_%s' % idx)
        if self.batch_norm:
            # with tf.variable_scope('ff_bn_%s' % idx,
            #         reuse=self.bn_reuse) as scope:
            processed_l0 = tf.contrib.layers.batch_norm(
                inputs=processed_l0,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        for i in range(self.h_repeat):
            _, l1_h2 = self.hgru_ops(
                i0=i0,
                x=processed_l0,
                h2=l1_h2,
                var_scope='hgru_%s' % idx,
                layer_idx=idx)



        # Intermediate FF - h2
        idx = 2
        if self.adapation:
            eta2 = getattr(self, 'eta2_%s' % idx)
            e2 = tf.gather(eta2, i0, axis=-1)
            l1_h2_processed = l1_h2*e2
        with tf.variable_scope('ff_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
            processed_l1 = tf.nn.conv3d(
                input=tf.concat([processed_l0, l1_h2_processed*processed_l0, l1_h2_processed], axis=4),
                filter=weights,
                strides=self.ff_conv_strides[idx],
                padding=self.padding)
            processed_l1 = tf.nn.bias_add(
                processed_l1,
                bias)
            processed_l1 = self.ff_nl(processed_l1)
        # Pool the preceding layer's drive
        if self.include_pooling:
            processed_l1 = max_pool3d(
                bottom=processed_l1,
                k=self.ff_pool_dhw[idx],
                s=self.ff_pool_strides[idx],
                name='ff_pool_%s' % idx)
        if self.batch_norm:
            # with tf.variable_scope('ff_bn_%s' % idx,
            #         reuse=self.bn_reuse) as scope:
            processed_l1 = tf.contrib.layers.batch_norm(
                inputs=processed_l1,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        for i in range(self.h_repeat):
            _, l2_h2 = self.hgru_ops(
                i0=i0,
                x=processed_l1,
                h2=l2_h2,
                var_scope='hgru_%s' % idx,
                layer_idx=idx)




        # Intermediate FF
        idx = 3
        if self.adapation:
            eta2 = getattr(self, 'eta2_%s' % idx)
            e2 = tf.gather(eta2, i0, axis=-1)
            l2_h2_processed = l2_h2*e2
        with tf.variable_scope('ff_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
            top = tf.nn.conv3d(
                input=tf.concat([processed_l1, l2_h2_processed*processed_l1, l2_h2_processed], axis=4),
                filter=weights,
                strides=self.ff_conv_strides[idx],
                padding=self.padding)
            top = tf.nn.bias_add(
                top,
                bias)
            top = self.ff_nl(top)
        # Pool the preceding layer's drive
        if self.include_pooling:
            top = max_pool3d(
                bottom=top,
                k=self.ff_pool_dhw[idx],
                s=self.ff_pool_strides[idx],
                name='ff_pool_%s' % idx)
        if self.batch_norm:
            # with tf.variable_scope('ff_bn_%s' % idx,
            #         reuse=self.bn_reuse) as scope:
            top = tf.contrib.layers.batch_norm(
                inputs=top,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)




        # FB
        idx=2
        with tf.variable_scope('fb_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
        top = self.resize_x_to_y(x=top, y=l2_h2,
                                  kernel=weights,
                                  bias=bias,
                                  mode=self.fb_mode,
                                  strides=self.ff_pool_strides[3])
        if self.batch_norm:
            # with tf.variable_scope('fb_bn' % 2,
            #         reuse=self.bn_reuse) as scope:
            top = tf.contrib.layers.batch_norm(
                inputs=top,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        idx=1
        with tf.variable_scope('fb_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
        top = self.resize_x_to_y(x=top, y=l1_h2,
                                  kernel=weights,
                                  bias=bias,
                                  mode=self.fb_mode,
                                  strides=self.ff_pool_strides[2])
        if self.batch_norm:
            # with tf.variable_scope('fb_bn' % 1,
            #         reuse=self.bn_reuse) as scope:
            top = tf.contrib.layers.batch_norm(
                inputs=top,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)
        idx=0
        with tf.variable_scope('fb_%s' % idx, reuse=True):
            weights = tf.get_variable("weights")
            bias = tf.get_variable("bias")
        top = self.resize_x_to_y(x=top, y=fb_h2,
                                  kernel=weights,
                                  bias=bias,
                                  mode=self.fb_mode,
                                  strides=self.ff_pool_strides[1])
        if self.batch_norm:
            # with tf.variable_scope('fb_bn' % 0,
            #         reuse=self.bn_reuse) as scope:
            top = tf.contrib.layers.batch_norm(
                inputs=top,
                scale=True,
                center=True,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                reuse=self.bn_reuse,
                is_training=self.train)

        _, fb_h2 = self.hgru_ops(
            i0=i0,
            x=top,
            h2=fb_h2,
            var_scope='hgru_%s' % 3,
            layer_idx=3)


        # Iterate loop
        i0 += 1
        return i0, x, l0_h2, l1_h2, l2_h2, fb_h2

    def condition(self, i0, x, l0_h2, l1_h2, l2_h2, fb_h2):
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
            l0_shape = [
                    x_shape[0],
                    self.compute_shape(x_shape[1], self.ff_pool_strides[0][0]),
                    self.compute_shape(x_shape[2], self.ff_pool_strides[0][1]),
                    self.compute_shape(x_shape[3], self.ff_pool_strides[0][2]),
                    self.ff_conv_k[0]]
            l1_shape = [
                    x_shape[0],
                    self.compute_shape(l0_shape[1], self.ff_pool_strides[1][0]),
                    self.compute_shape(l0_shape[2], self.ff_pool_strides[1][1]),
                    self.compute_shape(l0_shape[3], self.ff_pool_strides[1][2]),
                    self.ff_conv_k[1]]
            l2_shape = [
                    x_shape[0],
                    self.compute_shape(l1_shape[1], self.ff_pool_strides[2][0]),
                    self.compute_shape(l1_shape[2], self.ff_pool_strides[2][1]),
                    self.compute_shape(l1_shape[3], self.ff_pool_strides[2][2]),
                    self.ff_conv_k[2]]
            fb_shape = x_shape
        else:
            l0_shape = tf.identity(x_shape)
            l1_shape = tf.identity(x_shape)
            l2_shape = tf.identity(x_shape)
            fb_shape = tf.identity(x_shape)

        # Initialize hidden layer activities
        l0_h2 = tf.zeros(l0_shape, dtype=self.dtype)
        l1_h2 = tf.zeros(l1_shape, dtype=self.dtype)
        l2_h2 = tf.zeros(l2_shape, dtype=self.dtype)
        fb_h2 = tf.tile(seed, [1, 1, 1, 1, fb_shape[-1]])

        # While loop
        elems = [
            i0,
            x,
            l0_h2,
            l1_h2,
            l2_h2,
            fb_h2
        ]

        returned = tf.while_loop(
            self.condition,
            self.full,
            loop_vars=elems,
            back_prop=True,
            swap_memory=False)

        # Prepare output
        i0, x, l0_h2, l1_h2, l2_h2, fb_h2 = returned

        # stabilizer
        l0_h2 = self.cap(x, fb_h2, l0_h2)

        return l0_h2
