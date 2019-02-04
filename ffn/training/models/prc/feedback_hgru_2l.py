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
            timesteps,
            hgru_dhw,
            hgru_k,
            ff_conv_dhw,
            ff_conv_k,
            ff_conv_strides=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
            ff_pool_dhw=[[1, 2, 2], [1, 2, 2]],
            ff_pool_strides=[[1, 2, 2], [1, 2, 2]],
            fb_mode = 'transpose',
            fb_dhw=[[1, 2, 2], [1, 2, 2]],
            padding='SAME',
            peephole=False,
            aux=None,
            train=True):
        """Global initializations and settings."""
        self.in_k = num_in_feats
        self.timesteps = timesteps
        self.padding = padding
        self.train = train
        self.layer_name = layer_name
        self.fb_mode = fb_mode # 'transpose', 'replicate_n_transpose'
        self.peephole = peephole

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
        self.hgru_k = hgru_k
        self.fb_dhw = fb_dhw

        # Nonlinearities and initializations
        if isinstance(self.recurrent_nl, basestring):
            self.recurrent_nl = self.interpret_nl(self.recurrent_nl)

        # Handle BN scope reuse
        if self.reuse:
            self.scope_reuse = tf.AUTO_REUSE
        else:
            self.scope_reuse = None
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
            'lesion_alpha': False,
            'lesion_mu': False,
            'lesion_omega': False,
            'lesion_kappa': False,
            'dtype': tf.float32,
            'hidden_init': 'random',
            'gate_bias_init': 'chronos',
            'train': True,
            'recurrent_nl': tf.nn.tanh,
            'gate_nl': tf.nn.sigmoid,
            'ff_nl': tf.nn.elu,
            'normal_initializer': True,
            'symmetric_weights': False,
            'symmetric_gate_weights': False,
            'hgru_gate_dhw': [[1, 1, 1],[1, 1, 1],[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # Gate kernel size
            'hgru_dilations': [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
            'gamma': True,  # Scale P
            'alpha': True,  # divisive eCRF
            'mu': True,  # subtractive eCRF
            'adapation': False,
            'reuse': False,
            'multiplicative_excitation': True,
            'readout': 'fb',  # l2 or fb
            'hgru_ids': ['h1', 'h2', 'fb1'],  # Labels for the hGRUs
            'include_pooling': True,
            'resize_kernel': tf.image.ResizeMethod.BILINEAR,
            'batch_norm': False,  # Not working
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
        # FEEDFORWARD AND FEEDBACK KERNELS
        lower_feats = self.in_k
        for idx, (higher_feats, ff_dhw, fb_dhw) in enumerate(
                zip(self.ff_conv_k,
                    self.ff_conv_dhw,
                    self.fb_dhw)):
            setattr(
                self,
                'fb_kernel_%s' % idx,
                tf.get_variable(
                    name='%s_fb_kernel__%s' % (self.layer_name, idx),
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=fb_dhw + [lower_feats, higher_feats],
                        dtype=self.dtype,
                        uniform=self.normal_initializer),
                    trainable=True))
            setattr(
                self,
                'fb_bias_%s' % idx,
                tf.get_variable(
                    name='%s_fb_bias_%s' % (self.layer_name, idx),
                    dtype=self.dtype,
                    initializer=tf.ones([lower_feats], dtype=self.dtype),
                    trainable=True))
            setattr(
                self,
                'ff_kernel_%s' % idx,
                tf.get_variable(
                    name='%s_ff_kernel_%s' % (self.layer_name, idx),
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=ff_dhw + [lower_feats, higher_feats],
                        dtype=self.dtype,
                        uniform=self.normal_initializer),
                    trainable=True))
            setattr(
                self,
                'ff_bias_%s' % idx,
                tf.get_variable(
                    name='%s_ff_bias_%s' % (self.layer_name, idx),
                    dtype=self.dtype,
                    initializer=tf.ones([higher_feats], dtype=self.dtype),
                    trainable=True))
            lower_feats = higher_feats

        # HGRU KERNELS
        for idx, layer in enumerate(self.hgru_ids):
            with tf.variable_scope(
                    '%s_hgru_weights_%s' % (self.layer_name, layer)):
                setattr(
                    self,
                    'horizontal_kernels_%s' % layer,
                    tf.get_variable(
                        name='%s_horizontal' % self.layer_name,
                        dtype=self.dtype,
                        initializer=initialization.xavier_initializer(
                            shape=self.hgru_dhw[idx] + [self.hgru_k[idx], self.hgru_k[idx]],
                            dtype=self.dtype,
                            uniform=self.normal_initializer),
                        trainable=True))
                g_shape = self.hgru_gate_dhw[idx] + [self.hgru_k[idx], self.hgru_k[idx]]
                setattr(
                    self,
                    'gain_kernels_%s' % layer,
                    tf.get_variable(
                        name='%s_gain' % self.layer_name,
                        dtype=self.dtype,
                        trainable=True,
                        initializer=initialization.xavier_initializer(
                            shape=g_shape,
                            dtype=self.dtype,
                            uniform=self.normal_initializer,
                            mask=None)))
                m_shape = self.hgru_gate_dhw[idx] + [self.hgru_k[idx], self.hgru_k[idx]]
                setattr(
                    self,
                    'mix_kernels_%s' % layer,
                    tf.get_variable(
                        name='%s_mix' % self.layer_name,
                        dtype=self.dtype,
                        trainable=True,
                        initializer=initialization.xavier_initializer(
                            shape=m_shape,
                            dtype=self.dtype,
                            uniform=self.normal_initializer,
                            mask=None)))

                # Gain bias
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
                    'gain_bias_%s' % layer,
                    tf.get_variable(
                        name='%s_gain_bias' % self.layer_name,
                        dtype=self.dtype,
                        trainable=True,
                        initializer=bias_init))
                if self.gate_bias_init == 'chronos':
                    bias_init = -bias_init
                else:
                    bias_init = tf.ones(bias_shape, dtype=self.dtype)
                setattr(
                    self,
                    'mix_bias_%s' % layer,
                    tf.get_variable(
                        name='%s_mix_bias' % self.layer_name,
                        dtype=self.dtype,
                        trainable=True,
                        initializer=bias_init))

                # Divisive params
                if self.alpha and not self.lesion_alpha:
                    setattr(
                        self,
                        'alpha_%s' % layer,
                        tf.get_variable(
                            name='%s_alpha' % self.layer_name,
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=bias_shape,
                                dtype=self.dtype,
                                uniform=self.normal_initializer,
                                mask=None)))
                elif self.lesion_alpha:
                    setattr(
                        self,
                        'alpha_%s' % layer,
                        tf.constant(0.))
                else:
                    setattr(
                        self,
                        'alpha_%s' % layer,
                        tf.constant(1.))

                if self.mu and not self.lesion_mu:
                    setattr(
                        self,
                        'mu_%s' % layer,
                        tf.get_variable(
                            name='%s_mu' % self.layer_name,
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=bias_shape,
                                dtype=self.dtype,
                                uniform=self.normal_initializer,
                                mask=None)))

                elif self.lesion_mu:
                    setattr(
                        self,
                        'mu_%s' % layer,
                        tf.constant(0.))
                else:
                    setattr(
                        self,
                        'mu_%s' % layer,
                        tf.constant(1.))

                if self.gamma:
                    setattr(
                        self,
                        'gamma_%s' % layer,
                        tf.get_variable(
                            name='%s_gamma' % self.layer_name,
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=bias_shape,
                                dtype=self.dtype,
                                uniform=self.normal_initializer,
                                mask=None)))
                else:
                    setattr(
                        self,
                        'gamma_%s' % layer,
                        tf.constant(1.))

                if self.multiplicative_excitation:
                    if self.lesion_kappa:
                        setattr(
                            self,
                            'kappa_%s' % layer,
                            tf.constant(0.))
                    else:
                        setattr(
                            self,
                            'kappa_%s' % layer,
                            tf.get_variable(
                                name='%s_kappa' % self.layer_name,
                                dtype=self.dtype,
                                initializer=initialization.xavier_initializer(
                                    shape=bias_shape,
                                    dtype=self.dtype,
                                    uniform=self.normal_initializer,
                                    mask=None)))
                    if self.lesion_omega:
                        setattr(
                            self,
                            'omega_%s' % layer,
                            tf.constant(0.))
                    else:
                        setattr(
                            self,
                            'omega_%s' % layer,
                            tf.get_variable(
                                name='%s_omega' % self.layer_name,
                                dtype=self.dtype,
                                initializer=initialization.xavier_initializer(
                                    shape=bias_shape,
                                    dtype=self.dtype,
                                    uniform=self.normal_initializer,
                                    mask=None)))
                else:
                    setattr(
                        self,
                        'kappa_%s' % layer,
                        tf.constant(1.))
                    setattr(
                        self,
                        'omega_%s' % layer,
                        tf.constant(1.))
                if self.adapation:
                    setattr(
                        self,
                        'eta_%s' % layer,
                        tf.get_variable(
                            name='%s_eta' % self.layer_name,
                            dtype=self.dtype,
                            initializer=tf.random_uniform(
                                [self.timesteps], dtype=tf.float32)))
                if self.lesion_omega:
                    setattr(
                        self,
                        'omega_%s' % layer,
                        tf.constant(0.))
                if self.lesion_kappa:
                    setattr(
                        self,
                        'kappa_%s' % layer,
                        tf.constant(0.))
                if self.reuse:
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

    def circuit_input(self, h2, layer, var_scope, layer_idx):
        """Calculate gain and inh horizontal activities."""
        gain_kernels = getattr(self, 'gain_kernels_%s' % layer)
        gain_bias = getattr(self, 'gain_bias_%s' % layer)
        horizontal_kernels = getattr(self, 'horizontal_kernels_%s' % layer)
        # h_bias = getattr(self, 'h_bias_%s' % layer)
        g1_intermediate = self.conv_3d_op(
            data=h2,
            weights=gain_kernels,
            strides=[1, 1, 1, 1, 1],
            symmetric_weights=self.symmetric_gate_weights,
            dilations=self.hgru_dilations[layer_idx])
        with tf.variable_scope(
                '%s/g1_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            g1_intermediate = tf.contrib.layers.batch_norm(
                inputs=g1_intermediate + gain_bias,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
        g1 = self.gate_nl(g1_intermediate)
        h2 *= g1

        # Horizontal activities
        c1 = self.conv_3d_op(
            data=h2,
            weights=horizontal_kernels,
            strides=[1, 1, 1, 1, 1],
            symmetric_weights=self.symmetric_weights,
            dilations=self.hgru_dilations[layer_idx])
        return c1, g1

    def circuit_output(self, h1, layer, var_scope, layer_idx):
        """Calculate mix and exc horizontal activities."""
        mix_kernels = getattr(self, 'mix_kernels_%s' % layer)
        mix_bias = getattr(self, 'mix_bias_%s' % layer)
        horizontal_kernels = getattr(self, 'horizontal_kernels_%s' % layer)
        # h_bias = getattr(self, 'h_bias_%s' % layer)
        g2_intermediate = self.conv_3d_op(
            data=h1,
            weights=mix_kernels,
            strides=[1, 1, 1, 1, 1],
            symmetric_weights=self.symmetric_gate_weights,
            dilations=self.hgru_dilations[layer_idx])

        with tf.variable_scope(
                '%s/g2_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            g2_intermediate = tf.contrib.layers.batch_norm(
                inputs=g2_intermediate + mix_bias,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
        g2 = self.gate_nl(g2_intermediate)

        # Horizontal activities
        c2 = self.conv_3d_op(
            data=h1,
            weights=horizontal_kernels,
            strides=[1, 1, 1, 1, 1],
            symmetric_weights=self.symmetric_weights,
            dilations=self.hgru_dilations[layer_idx])
        return c2, g2

    def input_integration(self, x, c1, h2, layer):
        """Integration on the input."""
        alpha = getattr(self, 'alpha_%s' % layer)
        mu = getattr(self, 'mu_%s' % layer)
        return self.recurrent_nl(x - ((alpha * h2 + mu) * c1))

    def output_integration(self, h1, c2, g2, h2, layer):
        """Integration on the output."""
        if self.multiplicative_excitation:
            # Multiplicative gating I * (P + Q)
            gamma = getattr(self, 'gamma_%s' % layer)
            kappa = getattr(self, 'kappa_%s' % layer)
            omega = getattr(self, 'omega_%s' % layer)
            e = gamma * c2
            a = kappa * (h1 + e)
            m = omega * (h1 * e)
            h2_hat = self.recurrent_nl(a + m)
        else:
            # Additive gating I + P + Q
            gamma = getattr(self, 'gamma_%s' % layer)
            h2_hat = self.recurrent_nl(
                h1 + gamma * c2)
        return (g2 * h2) + ((1 - g2) * h2_hat)

    def hgru_ops(self, i0, x, h2, layer, layer_idx):
        """hGRU body."""
        var_scope = '%s_hgru_weights' % layer
        # Circuit input receives recurrent output h2
        c1, g1 = self.circuit_input(
            h2=h2,
            layer=layer,
            var_scope=var_scope,
            layer_idx=layer_idx)
        with tf.variable_scope(
                '%s/c1_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            c1 = tf.contrib.layers.batch_norm(
                inputs=c1,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)

        # Calculate input (-) integration: h1 (4)
        h1 = self.input_integration(
            x=x,
            c1=c1,
            h2=h2,
            layer=layer)

        # Circuit output receives recurrent input h1
        c2, g2 = self.circuit_output(
            h1=h1,
            layer=layer,
            var_scope=var_scope,
            layer_idx=layer_idx)

        with tf.variable_scope(
                '%s/c2_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            c2 = tf.contrib.layers.batch_norm(
                inputs=c2,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)

        # Calculate output (+) integration: h2 (8, 9)
        h2 = self.output_integration(
            h1=h1,
            c2=c2,
            g2=g2,
            h2=h2,
            layer=layer)

        if self.adapation:
            eta = getattr(self, 'eta_%s' % layer)
            e = tf.gather(eta, i0, axis=-1)
            h2 *= e
        return h1, h2

    def full(self, i0, x, l1_h2, l2_h2):
        """hGRU body.
        Take the recurrent h2 from a low level and imbue it with
        information froma high layer. This means to treat the lower
        layer h2 as the X and the higher layer h2 as the recurrent state.
        This will serve as I/E from the high layer along with feedback
        kernels.
        """

        # LAYER 1
        _, l1_h2 = self.hgru_ops(
            i0=i0,
            x=x,
            h2=l1_h2,
            layer='h1',
            layer_idx=0)

        # Intermediate FF
        if self.batch_norm:
            with tf.variable_scope(
                    'l1_h2_bn',
                    reuse=self.scope_reuse) as scope:
                l1_h2 = tf.contrib.layers.batch_norm(
                    inputs=l1_h2,
                    scale=True,
                    center=True,
                    fused=True,
                    renorm=False,
                    param_initializers=self.param_initializer,
                    updates_collections=None,
                    scope=scope,
                    reuse=self.reuse,
                    is_training=self.train)

        # Pool the preceding layer's drive
        if self.include_pooling:
            processed_l1_h2 = max_pool3d(
                bottom=l1_h2,
                k=self.ff_pool_dhw[0],
                s=self.ff_pool_strides[0],
                name='ff_pool_%s' % 0)
        else:
            processed_l1_h2 = l1_h2

        # LAYER 2
        idx = 0
        processed_l1_h2 = tf.nn.conv3d(
            input=processed_l1_h2,
            filter=getattr(self, 'ff_kernel_%s' % idx),
            strides=self.ff_conv_strides[idx],
            padding=self.padding)
        processed_l1_h2 = tf.nn.bias_add(
            processed_l1_h2,
            getattr(self, 'ff_bias_%s' % idx))
        processed_l1_h2 = self.ff_nl(processed_l1_h2)
        if self.batch_norm:
            with tf.variable_scope(
                    'l1_h2_bn_ff_%s' % idx,
                     reuse=self.scope_reuse) as scope:
                processed_l1_h2 = tf.contrib.layers.batch_norm(
                    inputs=processed_l1_h2,
                    scale=True,
                    center=True,
                    fused=True,
                    renorm=False,
                    param_initializers=self.param_initializer,
                    updates_collections=None,
                    scope=scope,
                    reuse=self.reuse,
                    is_training=self.train)
        _, l2_h2 = self.hgru_ops(
            i0=i0,
            x=processed_l1_h2,
            h2=l2_h2,
            layer='h2',
            layer_idx=1)
        if self.batch_norm:
            with tf.variable_scope(
                    'l2_h2_bn',
                    reuse=self.scope_reuse) as scope:
                l2_h2 = tf.contrib.layers.batch_norm(
                    inputs=l2_h2,
                    scale=True,
                    center=True,
                    fused=True,
                    renorm=False,
                    param_initializers=self.param_initializer,
                    updates_collections=None,
                    scope=scope,
                    reuse=self.reuse,
                    is_training=self.train)


        # l2-l1 feedback (FEEDBACK KERNEL is 2x channels)
        _, temp_l1_h2 = self.hgru_ops(
            i0=i0,
            x=l1_h2,
            h2=self.resize_x_to_y(x=l2_h2, y=l1_h2,
                                  kernel=self.fb_kernel_0,
                                  bias=self.fb_bias_0,
                                  mode=self.fb_mode,
                                  strides=self.ff_pool_strides[0]),
            layer='fb1',
            layer_idx=4)

        # Peephole
        if self.peephole:
            l1_h2 = temp_l1_h2 + l1_h2
        else:
            l1_h2 = temp_l1_h2

        # Iterate loop
        i0 += 1
        return i0, x, l1_h2, l2_h2

    def condition(self, i0, x, l1_h2, l2_h2):
        """While loop halting condition."""
        return i0 < self.timesteps

    def compute_shape(self, in_length, stride):
        if in_length % stride == 0:
            return in_length/stride
        else:
            return in_length/stride + 1

    def build(self, x):
        """Run the backprop version of the Circuit."""
        self.prepare_tensors()
        i0 = tf.constant(0)

        # Calculate l2 hidden state size
        x_shape = x.get_shape().as_list()
        if self.include_pooling and len(self.ff_conv_k):
            if len(self.ff_conv_k):
                final_dim = self.ff_conv_k[-1]
            else:
                final_dim = x_shape[-1]
            l2_shape = [
                    x_shape[0],
                    self.compute_shape(x_shape[1], self.ff_pool_strides[0][0]),
                    self.compute_shape(x_shape[2], self.ff_pool_strides[0][1]),
                    self.compute_shape(x_shape[3], self.ff_pool_strides[0][2]),
                    final_dim]
        else:
            l2_shape = tf.identity(x_shape)

        # Initialize hidden layer activities
        if self.hidden_init == 'identity':
            l1_h2 = tf.identity(x)
            l2_h2 = tf.zeros(l2_shape, dtype=self.dtype)
        elif self.hidden_init == 'random':
            l1_h2 = tf.random_normal(x_shape, dtype=self.dtype)
            l2_h2 = tf.random_normal(l2_shape, dtype=self.dtype)
        elif self.hidden_init == 'zeros':
            l1_h2 = tf.zeros(x_shape, dtype=self.dtype)
            l2_h2 = tf.zeros(l2_shape, dtype=self.dtype)
        else:
            raise RuntimeError

        # While loop
        elems = [
            i0,
            x,
            l1_h2,
            l2_h2
        ]
        returned = tf.while_loop(
            self.condition,
            self.full,
            loop_vars=elems,
            back_prop=True,
            swap_memory=False)

        # Prepare output
        i0, x, l1_h2, l2_h2 = returned
        return l1_h2
