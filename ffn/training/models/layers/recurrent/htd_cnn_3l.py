"""Contextual model with partial filters."""
import warnings
import numpy as np
import tensorflow as tf
import initialization
import gradients
from pooling import max_pool

# Dependency for symmetric weight ops is in models/layers/ff.py
class hGRU(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            var_scope,
            timesteps,
            dtype,
            use_3d,
            train,
            train_bn,
            use_in,
            bn_decay,
            in_k,

            hgru1_fsiz,
            hgru2_fsiz,
            hgru3_fsiz,
            hgru_td3_fsiz,
            hgru_td2_fsiz,
            hgru_td1_fsiz,
            hgru_h1_nl,
            hgru_h2_nl,
            hgru_bistream_weights,
            hgru_in_place_integration,
            hgru_symmetric_weights,
            hgru_soft_coefficients,
            belly_up_td,

            ds_fsiz_list,
            ds_conv_repeat,
            ds_k_list,
            ds_pool_list,
            ds_stride_list,

            use_dsus_skip=False,
            use_homunculus=False,
            use_trainable_states=False
            ):

        # global params
        self.var_scope = var_scope
        self.timesteps = timesteps
        self.dtype = dtype
        self.use_3d = use_3d
        self.train = train
        self.train_bn = train_bn
        self.use_in = use_in
        self.bn_decay = bn_decay
        self.in_k = in_k
        self.use_homunculus = use_homunculus
        if use_homunculus:
            raise ValueError('use_homunculus is not enabled yet in this model.')
        self.use_trainable_states = use_trainable_states

        # hgru params
        self.hgru1_fsiz = hgru1_fsiz
        self.hgru2_fsiz = hgru2_fsiz
        self.hgru3_fsiz = hgru3_fsiz
        self.hgru_td3_fsiz = hgru_td3_fsiz
        self.hgru_td2_fsiz = hgru_td2_fsiz
        self.hgru_td1_fsiz = hgru_td1_fsiz
        self.hgru_h1_nl = hgru_h1_nl
        self.hgru_h2_nl = hgru_h2_nl
        self.hgru_bistream_weights = hgru_bistream_weights
        self.hgru_symmetric_weights = hgru_symmetric_weights
        self.hgru_soft_coefficients = hgru_soft_coefficients
        self.hgru_in_place_integration = hgru_in_place_integration
        self.belly_up_td = belly_up_td

        # DS-US params
        self.ds_fsiz_list = ds_fsiz_list
        self.ds_conv_repeat = ds_conv_repeat
        self.ds_k_list = ds_k_list
        self.ds_pool_list = ds_pool_list
        self.ds_stride_list = ds_stride_list
        self.use_dsus_skip = use_dsus_skip
        if use_dsus_skip:
            raise ValueError('use_dsus_skip is not allowed in this model.')

        # bn params
        self.bn_param_initializer = {
                            'moving_mean': tf.constant_initializer(0., dtype=self.dtype),
                            'moving_variance': tf.constant_initializer(1., dtype=self.dtype),
                            'beta': tf.constant_initializer(0., dtype=self.dtype),
                            'gamma': tf.constant_initializer(0.1, dtype=self.dtype)
        }

        from ..feedforward.fgru_original import hGRU
        self.hgru1 = hGRU(var_scope=self.var_scope + '/hgru1',
                          fsiz=self.hgru1_fsiz,
                          num_channels=self.in_k,
                          use_3d=self.use_3d,
                          symmetric_conv_weights=self.hgru_symmetric_weights,
                          bistream_weights=self.hgru_bistream_weights,
                          in_place_integration=self.hgru_in_place_integration, # setting it to False will make +* ops performed on the target activity, not on itself.
                          soft_coefficients=self.hgru_soft_coefficients,
                          h1_nl=self.hgru_h1_nl,
                          h2_nl=self.hgru_h2_nl,
                          gate_nl=tf.nn.sigmoid,
                          bn_reuse=False,
                          train=self.train,
                          train_bn=self.train_bn,
                          bn_decay=self.bn_decay,
                          dtype=self.dtype)
        self.hgru2 = hGRU(var_scope=self.var_scope + '/hgru2',
                          fsiz=self.hgru2_fsiz,
                          num_channels=self.ds_k_list[0],
                          use_3d=self.use_3d,
                          symmetric_conv_weights=self.hgru_symmetric_weights,
                          bistream_weights=self.hgru_bistream_weights,
                          in_place_integration=self.hgru_in_place_integration,
                          soft_coefficients=self.hgru_soft_coefficients,
                          h1_nl=self.hgru_h1_nl,
                          h2_nl=self.hgru_h2_nl,
                          gate_nl=tf.nn.sigmoid,
                          bn_reuse=False,
                          train=self.train,
                          train_bn=self.train_bn,
                          bn_decay=self.bn_decay,
                          dtype=self.dtype)
        self.hgru3 = hGRU(var_scope=self.var_scope + '/hgru3',
                          fsiz=self.hgru3_fsiz,
                          num_channels=self.ds_k_list[1],
                          use_3d=self.use_3d,
                          symmetric_conv_weights=self.hgru_symmetric_weights,
                          bistream_weights=self.hgru_bistream_weights,
                          in_place_integration=self.hgru_in_place_integration,
                          soft_coefficients=self.hgru_soft_coefficients,
                          h1_nl=self.hgru_h1_nl,
                          h2_nl=self.hgru_h2_nl,
                          gate_nl=tf.nn.sigmoid,
                          bn_reuse=False,
                          train=self.train,
                          train_bn=self.train_bn,
                          bn_decay=self.bn_decay,
                          dtype=self.dtype)
        self.hgru_td3 = hGRU(var_scope=self.var_scope + '/hgru_td3',
                             fsiz=self.hgru_td3_fsiz,
                             num_channels=self.ds_k_list[1],
                             use_3d=self.use_3d,
                             symmetric_conv_weights=False,
                             bistream_weights=self.hgru_bistream_weights,
                             in_place_integration=self.hgru_in_place_integration,
                             soft_coefficients=self.hgru_soft_coefficients,
                             h1_nl=self.hgru_h1_nl,
                             h2_nl=self.hgru_h2_nl,
                             gate_nl=tf.nn.sigmoid,
                             bn_reuse=False,
                             train=self.train,
                             train_bn=self.train_bn,
                             bn_decay=self.bn_decay,
                             dtype=self.dtype)
        self.hgru_td2 = hGRU(var_scope=self.var_scope + '/hgru_td2',
                             fsiz=self.hgru_td2_fsiz,
                             num_channels=self.ds_k_list[0],
                             use_3d=self.use_3d,
                             symmetric_conv_weights=False,
                             bistream_weights=self.hgru_bistream_weights,
                             in_place_integration=self.hgru_in_place_integration,
                             soft_coefficients=self.hgru_soft_coefficients,
                             h1_nl=self.hgru_h1_nl,
                             h2_nl=self.hgru_h2_nl,
                             gate_nl=tf.nn.sigmoid,
                             bn_reuse=False,
                             train=self.train,
                             train_bn=self.train_bn,
                             bn_decay=self.bn_decay,
                             dtype=self.dtype)
        self.hgru_td1 = hGRU(var_scope=self.var_scope + '/hgru_td1',
                             fsiz=self.hgru_td1_fsiz,
                             num_channels=self.in_k,
                             use_3d=self.use_3d,
                             symmetric_conv_weights=False,
                             bistream_weights=self.hgru_bistream_weights,
                             in_place_integration=self.hgru_in_place_integration,
                             soft_coefficients=self.hgru_soft_coefficients,
                             h1_nl=self.hgru_h1_nl,
                             h2_nl=self.hgru_h2_nl,
                             gate_nl=tf.nn.sigmoid,
                             bn_reuse=False,
                             train=self.train,
                             train_bn=self.train_bn,
                             bn_decay=self.bn_decay,
                             dtype=self.dtype)

        print('>>>>>>>>>>>>>>>>>>>>>>IS_TRAINING: ' + str(self.train))

    def prepare_tensors(self):

        # HGRU KERNELS
        self.hgru1.prepare_tensors()
        self.hgru2.prepare_tensors()
        self.hgru3.prepare_tensors()
        self.hgru_td3.prepare_tensors()
        self.hgru_td2.prepare_tensors()
        self.hgru_td1.prepare_tensors()

        # HOMU
        # if self.use_homunculus:
        #     self.homunculus = tf.get_variable(
        #                         name='homunculus',
        #                         dtype=self.dtype,
        #                         initializer=tf.zeros(self.timesteps),
        #                         trainable=self.train)

        # DS US KERNELS
        in_k = self.in_k
        for i, (fsiz, psiz, out_k) in enumerate(zip(self.ds_fsiz_list, self.ds_pool_list, self.ds_k_list)):
            for rep in range(self.ds_conv_repeat):
                with tf.variable_scope('ds%s_%s' % (i,rep)):
                    setattr(
                        self,
                        'ds%s_%s_w' % (i,rep),
                        tf.get_variable(
                            name='w',
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=fsiz + [in_k if rep==0 else out_k, out_k],
                                dtype=self.dtype,
                                uniform=True),
                            trainable=self.train))
                with tf.variable_scope('us%s_%s' % (i,rep)):
                    fsiz = [m+n-1 for m,n in zip(fsiz,psiz)] if (rep == self.ds_conv_repeat - 1) else fsiz
                    self.one_by_one = [1 for s in fsiz]
                    setattr(
                        self,
                        'us%s_%s_w' % (i,rep),
                        tf.get_variable(
                            name='w',
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=fsiz + [in_k if rep==0 else out_k, out_k],
                                dtype=self.dtype,
                                uniform=True),
                            trainable=self.train))
            # if (i < len(self.ds_fsiz_list) - 1) and self.use_dsus_skip:
            #     with tf.variable_scope('skip%s' % i):
            #         setattr(
            #             self,
            #             'skip%s_kappa' % i,
            #             tf.get_variable(
            #                 name='kappa',
            #                 dtype=self.dtype,
            #                 initializer=initialization.xavier_initializer(
            #                     shape=self.one_by_one + [out_k],
            #                     dtype=self.dtype,
            #                     uniform=True),
            #                 trainable=self.train))
            #         setattr(
            #             self,
            #             'skip%s_gamma' % i,
            #             tf.get_variable(
            #                 name='gamma',
            #                 dtype=self.dtype,
            #                 initializer=initialization.xavier_initializer(
            #                     shape=self.one_by_one + [out_k],
            #                     dtype=self.dtype,
            #                     uniform=True),
            #                 trainable=self.train))
            #         setattr(
            #             self,
            #             'skip%s_omega' % i,
            #             tf.get_variable(
            #                 name='omega',
            #                 dtype=self.dtype,
            #                 initializer=initialization.xavier_initializer(
            #                     shape=self.one_by_one + [out_k],
            #                     dtype=self.dtype,
            #                     uniform=True),
            #                 trainable=self.train))
            in_k = out_k

    def full(self, x, l1_h2, l2_h2, l3_h2, timestep):
        if self.use_3d:
            max_pool = tf.nn.max_pool3d
            conv = tf.nn.conv3d
            deconv = tf.nn.conv3d_transpose
        else:
            max_pool = tf.nn.max_pool
            conv = tf.nn.conv2d
            deconv = tf.nn.conv2d_transpose

        # HGRU1
        l1_h2 = self.hgru1.run(x, l1_h2)
        if self.use_in:
            l1_h2 = tf.contrib.layers.instance_norm(
                inputs=l1_h2,
                scale=True,
                center=False,
                trainable=self.train_bn)
        else:
            l1_h2 = tf.contrib.layers.batch_norm(
                inputs=l1_h2,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.bn_param_initializer,
                decay=self.bn_decay,
                updates_collections=None,
                reuse=None,
                is_training=self.train_bn)
        ds_in = l1_h2

        # DS
        ds_out_list = []
        ds_in_list = []

        i_ds = 0
        psiz = self.ds_pool_list[i_ds]
        strd = self.ds_stride_list[i_ds]
        ds_in_list.append(ds_in)
        ds_intm = ds_in
        for rep in range(self.ds_conv_repeat):
            with tf.variable_scope('ds%s_%s' % (i_ds, rep), reuse=tf.AUTO_REUSE):
                weights = tf.get_variable(name='w')
            ds_intm = conv(ds_intm, weights, strides=[1,1,1,1,1] if self.use_3d else [1,1,1,1], padding='SAME')
            if self.use_in:
                ds_intm = tf.contrib.layers.instance_norm(
                    inputs=ds_intm,
                    scale=True,
                    center=True,
                    trainable=self.train_bn)
            else:
                ds_intm = tf.contrib.layers.batch_norm(
                    inputs=ds_intm,
                    scale=True,
                    center=True,
                    fused=True,
                    renorm=False,
                    param_initializers=self.bn_param_initializer,
                    decay=self.bn_decay,
                    updates_collections=None,
                    reuse=None,
                    is_training=self.train_bn)
            ds_intm = tf.nn.relu(ds_intm)
        ds_out = max_pool(ds_intm, ksize=[1]+psiz+[1], strides=[1]+strd+[1], padding='SAME')
        ds_out_list.append(ds_out)

        # HGRU2
        l2_h2 = self.hgru2.run(ds_out, l2_h2)
        if self.use_in:
            l1_h2 = tf.contrib.layers.instance_norm(
                inputs=l1_h2,
                scale=True,
                center=False,
                trainable=self.train_bn)
        else:
            l1_h2 = tf.contrib.layers.batch_norm(
                inputs=l1_h2,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.bn_param_initializer,
                decay=self.bn_decay,
                updates_collections=None,
                reuse=None,
                is_training=self.train_bn)
        ds_in = l2_h2

        # DS
        i_ds = 1
        psiz = self.ds_pool_list[i_ds]
        strd = self.ds_stride_list[i_ds]
        ds_in_list.append(ds_in)
        ds_intm = ds_in
        for rep in range(self.ds_conv_repeat):
            with tf.variable_scope('ds%s_%s' % (i_ds, rep), reuse=tf.AUTO_REUSE):
                weights = tf.get_variable(name='w')
            ds_intm = conv(ds_intm, weights, strides=[1,1,1,1,1] if self.use_3d else [1,1,1,1], padding='SAME')
            if self.use_in:
                ds_intm = tf.contrib.layers.instance_norm(
                    inputs=ds_intm,
                    scale=True,
                    center=True,
                    trainable=self.train_bn)
            else:
                ds_intm = tf.contrib.layers.batch_norm(
                    inputs=ds_intm,
                    scale=True,
                    center=True,
                    fused=True,
                    renorm=False,
                    param_initializers=self.bn_param_initializer,
                    decay=self.bn_decay,
                    updates_collections=None,
                    reuse=None,
                    is_training=self.train_bn)
            ds_intm = tf.nn.relu(ds_intm)
        ds_out = max_pool(ds_intm, ksize=[1]+psiz+[1], strides=[1]+strd+[1], padding='SAME')
        ds_out_list.append(ds_out)

        # HGRU3
        l3_h2 = self.hgru3.run(ds_out, l3_h2)
        if self.use_in:
            l3_h2 = tf.contrib.layers.instance_norm(
                inputs=l3_h2,
                scale=True,
                center=False,
                trainable=self.train_bn)
        else:
            l3_h2 = tf.contrib.layers.batch_norm(
                inputs=l3_h2,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.bn_param_initializer,
                decay=self.bn_decay,
                updates_collections=None,
                reuse=None,
                is_training=self.train_bn)
        ds_in = l3_h2

        # TOP DS
        i_ds = 2
        psiz = self.ds_pool_list[i_ds]
        strd = self.ds_stride_list[i_ds]
        ds_in_list.append(ds_in)
        ds_intm = ds_in
        for rep in range(self.ds_conv_repeat):
            with tf.variable_scope('ds%s_%s' % (i_ds, rep), reuse=tf.AUTO_REUSE):
                weights = tf.get_variable(name='w')
            ds_intm = conv(ds_intm, weights, strides=[1,1,1,1,1] if self.use_3d else [1,1,1,1], padding='SAME')
            if self.use_in:
                ds_intm = tf.contrib.layers.instance_norm(
                    inputs=ds_intm,
                    scale=True,
                    center=False,
                    trainable=self.train_bn)
            else:
                ds_intm = tf.contrib.layers.batch_norm(
                    inputs=ds_intm,
                    scale=True,
                    center=False,
                    fused=True,
                    renorm=False,
                    param_initializers=self.bn_param_initializer,
                    decay=self.bn_decay,
                    updates_collections=None,
                    reuse=None,
                    is_training=self.train_bn)
            ds_intm = tf.nn.relu(ds_intm)
        ds_out = max_pool(ds_intm, ksize=[1]+psiz+[1], strides=[1]+strd+[1], padding='SAME')
        ds_out_list.append(ds_out)
        us_in = ds_out

        # TOP US
        i_ds = 2
        strd = self.ds_stride_list[i_ds]
        us_intm = us_in
        for rep in reversed(range(self.ds_conv_repeat)):
            with tf.variable_scope('us%s_%s' % (i_ds, rep), reuse=tf.AUTO_REUSE):
                weights = tf.get_variable(name='w')
            low_shape = ds_in_list[i_ds].get_shape().as_list()[:-1] + [us_intm.get_shape().as_list()[-1]] if (rep > 0) else ds_in_list[i_ds].get_shape().as_list()
            strides = [1]+strd+[1] if (rep == self.ds_conv_repeat - 1) else [1]+self.one_by_one+[1]
            us_intm = deconv(us_intm, weights,
                             output_shape=low_shape,
                             strides=strides, padding='SAME')
            if self.use_in:
                us_intm = tf.contrib.layers.instance_norm(
                    inputs=us_intm,
                    scale=True,
                    center=False,
                    trainable=self.train_bn)
            else:
                us_intm = tf.contrib.layers.batch_norm(
                                inputs=us_intm,
                                scale=True,
                                center=False,
                                fused=True,
                                renorm=False,
                                param_initializers=self.bn_param_initializer,
                                decay=self.bn_decay,
                                updates_collections=None,
                                reuse=None,
                                is_training=self.train_bn)
            us_out = tf.nn.relu(us_intm)

        # HGRU_TD3
        if self.belly_up_td:
            fb_act3 = self.hgru_td3.run(us_out, l3_h2)
        else:
            fb_act3 = self.hgru_td3.run(l3_h2, us_out)
        if self.use_in:
            fb_act3 = tf.contrib.layers.instance_norm(
                inputs=fb_act3,
                scale=True,
                center=False,
                trainable=self.train_bn)
        else:
            fb_act3 = tf.contrib.layers.batch_norm(
                inputs=fb_act3,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.bn_param_initializer,
                decay=self.bn_decay,
                updates_collections=None,
                reuse=None,
                is_training=self.train_bn)
        l3_h2 = fb_act3
        us_in = l3_h2

        # US
        i_ds = 1
        strd = self.ds_stride_list[i_ds]
        us_intm = us_in
        for rep in reversed(range(self.ds_conv_repeat)):
            with tf.variable_scope('us%s_%s' % (i_ds, rep), reuse=tf.AUTO_REUSE):
                weights = tf.get_variable(name='w')
            low_shape = ds_in_list[i_ds].get_shape().as_list()[:-1] + [us_intm.get_shape().as_list()[-1]] if (rep > 0) else ds_in_list[i_ds].get_shape().as_list()
            strides = [1]+strd+[1] if (rep == self.ds_conv_repeat - 1) else [1]+self.one_by_one+[1]
            us_intm = deconv(us_intm, weights,
                             output_shape=low_shape,
                             strides=strides, padding='SAME')
            if self.use_in:
                us_intm = tf.contrib.layers.instance_norm(
                    inputs=us_intm,
                    scale=True,
                    center=False,
                    trainable=self.train_bn)
            else:
                us_intm = tf.contrib.layers.batch_norm(
                                inputs=us_intm,
                                scale=True,
                                center=False,
                                fused=True,
                                renorm=False,
                                param_initializers=self.bn_param_initializer,
                                decay=self.bn_decay,
                                updates_collections=None,
                                reuse=None,
                                is_training=self.train_bn)
            us_out = tf.nn.relu(us_intm)

        # HGRU_TD2
        if self.belly_up_td:
            fb_act2 = self.hgru_td2.run(us_out, l2_h2)
        else:
            fb_act2 = self.hgru_td2.run(l2_h2, us_out)
        if self.use_in:
            fb_act2 = tf.contrib.layers.instance_norm(
                inputs=fb_act2,
                scale=True,
                center=False,
                trainable=self.train_bn)
        else:
            fb_act2 = tf.contrib.layers.batch_norm(
                inputs=fb_act2,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.bn_param_initializer,
                decay=self.bn_decay,
                updates_collections=None,
                reuse=None,
                is_training=self.train_bn)
        l2_h2 = fb_act2
        us_in = l2_h2

        # US
        i_ds = 0
        strd = self.ds_stride_list[i_ds]
        us_intm = us_in
        for rep in reversed(range(self.ds_conv_repeat)):
            with tf.variable_scope('us%s_%s' % (i_ds, rep), reuse=tf.AUTO_REUSE):
                weights = tf.get_variable(name='w')
            low_shape = ds_in_list[i_ds].get_shape().as_list()[:-1] + [us_intm.get_shape().as_list()[-1]] if (rep > 0) else ds_in_list[i_ds].get_shape().as_list()
            strides = [1]+strd+[1] if (rep == self.ds_conv_repeat - 1) else [1]+self.one_by_one+[1]
            us_intm = deconv(us_intm, weights,
                             output_shape=low_shape,
                             strides=strides, padding='SAME')
            if self.use_in:
                us_intm = tf.contrib.layers.instance_norm(
                    inputs=us_intm,
                    scale=True,
                    center=False,
                    trainable=self.train_bn)
            else:
                us_intm = tf.contrib.layers.batch_norm(
                                inputs=us_intm,
                                scale=True,
                                center=False,
                                fused=True,
                                renorm=False,
                                param_initializers=self.bn_param_initializer,
                                decay=self.bn_decay,
                                updates_collections=None,
                                reuse=None,
                                is_training=self.train_bn)
            us_out = tf.nn.relu(us_intm)

        # HGRU_TD1
        if self.belly_up_td:
            fb_act1 = self.hgru_td1.run(us_out, l1_h2)
        else:
            fb_act1 = self.hgru_td1.run(l1_h2, us_out)
        if self.use_in:
            fb_act1 = tf.contrib.layers.instance_norm(
                inputs=fb_act1,
                scale=True,
                center=False,
                trainable=self.train_bn)
        else:
            fb_act1 = tf.contrib.layers.batch_norm(
                inputs=fb_act1,
                scale=True,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.bn_param_initializer,
                decay=self.bn_decay,
                updates_collections=None,
                reuse=None,
                is_training=self.train_bn)
        l1_h2 = fb_act1

        # # HOMUNCULUS (DISABLED IN THIS OBJECT)
        # if self.use_homunculus:
        #     homu = tf.nn.sigmoid(tf.gather(self.homunculus, timestep))
        #     l1_h2 = fb_act1*homu + l1_h2*(1-homu)
        # else:
        #     l1_h2 = fb_act1

        return l1_h2, l2_h2, l3_h2

    def compute_shape(self, in_length, stride):
        if in_length % stride == 0:
            return in_length/stride
        else:
            return in_length/stride + 1

    def build(self, x, ffn_seed=None):
        """Run the backprop version of the Circuit."""
        self.prepare_tensors()

        # Calculate hidden state size
        x_shape = x.get_shape().as_list()
        l1_h2_shape = x_shape
        # l2
        int_shape = list(x_shape[1:-1])
        str = self.ds_stride_list[0]
        for dim in range(len(int_shape)):
            int_shape[dim] = self.compute_shape(int_shape[dim], str[dim])
        l2_h2_shape = [x_shape[0]] + int_shape + [self.ds_k_list[0]]
        # l3
        int_shape = list(l2_h2_shape[1:-1])
        str = self.ds_stride_list[1]
        for dim in range(len(int_shape)):
            int_shape[dim] = self.compute_shape(int_shape[dim], str[dim])
        l3_h2_shape = [x_shape[0]] + int_shape + [self.ds_k_list[1]]

        # Initialize hidden layer activities
        if ffn_seed is not None:
            l1_h2 = tf.ones(l1_h2_shape, dtype=self.dtype) * ffn_seed * 2 - 1
        if self.use_trainable_states:
            if ffn_seed is None:
                l1_h2 = tf.get_variable(
                            name='l1_h2',
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=[1] + l1_h2_shape[1:],
                                dtype=self.dtype,
                                uniform=True),
                            trainable=self.train)
                l1_h2 = tf.tile(l1_h2, [x_shape[0]] + [1]*(4 if self.use_3d else 3))
            l2_h2 = tf.get_variable(
                            name='l2_h2',
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=[1] + l2_h2_shape[1:],
                                dtype=self.dtype,
                                uniform=True),
                            trainable=self.train)
            l2_h2 = tf.tile(l2_h2, [x_shape[0]] + [1]*(4 if self.use_3d else 3))
            l3_h2 = tf.get_variable(
                            name='l3_h2',
                            dtype=self.dtype,
                            initializer=initialization.xavier_initializer(
                                shape=[1] + l3_h2_shape[1:],
                                dtype=self.dtype,
                                uniform=True),
                            trainable=self.train)
            l3_h2 = tf.tile(l3_h2, [x_shape[0]] + [1]*(4 if self.use_3d else 3))
        else:
            if ffn_seed is None:
                l1_h2 = tf.zeros(l1_h2_shape, dtype=self.dtype)
            l2_h2 = tf.zeros(l2_h2_shape, dtype=self.dtype)
            l3_h2 = tf.zeros(l3_h2_shape, dtype=self.dtype)

        for i in range(self.timesteps):
            l1_h2_out, l2_h2_out, l3_h2_out = self.full(x, l1_h2, l2_h2, l3_h2, i)
            l1_h2 = l1_h2_out
            l2_h2 = l2_h2_out
            l3_h2 = l3_h2_out

        return l1_h2
