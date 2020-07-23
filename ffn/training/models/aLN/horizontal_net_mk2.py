"""Contextual model with partial filters."""
import operator
from functools import reduce

import tensorflow as tf
import normalization
import helpers_mk2

from pooling import max_pool
import initialization

# Dependency for symmetric weight ops is in models/layers/ff.py
class horizontal_net(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)



    def __init__(
            self,
            name,
            x_shape,
            b,              # number of bundles
            c,              # capsule dims
            timesteps=1,
            c_h_ext=15,
            l_h_ext=15,
            aux=None,
            train=True):
        """Global initializations and settings."""
        if x_shape[-1] % c > 0:
            raise ValueError('init: input shape is not divisible by c')
        self.name = name
        self.x_shape = x_shape
        self.b = b # number of bundles
        self.c = c # caps dims
        self.f = self.x_shape[-1]/c # number of feats
        self.timesteps = timesteps
        self.c_h_ext = c_h_ext
        self.l_h_ext = l_h_ext
        self.train = train

        # Sort through and assign the auxilliary variables
        default_vars = self.defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)

        # Handle BN scope reuse
        if self.use_bn_per_ts:
            self.batchnorm_var_scope = self.name
        else:
            self.batchnorm_var_scope = None
        if self.bn_reuse:
            self.scope_reuse = tf.AUTO_REUSE
        else:
            self.scope_reuse = None
        self.param_initializer = {
            'moving_mean': tf.constant_initializer(0.),
            'moving_variance': tf.constant_initializer(1.),
            'gamma': tf.constant_initializer(0.1)
        }
        self.param_trainable = {
            'moving_mean': True,
            'moving_variance': True,
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
            'train': True,
            'normal_initializer': True,
            'eps': 0.000001,
            'use_independent_labels_filter':True,
            'use_while_loop':False,
            'fixed_label_ind': None,
            'use_bn_per_ts':True,
            'bn_reuse':False
            }



    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)



    def prepare_tensors(self):
        """
        """
        # HORIZONTAL FILTERS
        self.caps_filter = \
            tf.get_variable(
                name='%s_caps_filter' % (self.name),
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=[self.c_h_ext, self.c_h_ext] + [self.f * self.c, self.b * self.f * self.c],
                    uniform=self.normal_initializer),
                trainable=True)

        if self.use_independent_labels_filter:
            self.labels_filter = \
                tf.get_variable(
                    name='%s_labels_filter' % (self.name),
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=[self.l_h_ext, self.l_h_ext] + [self.f, self.b * self.f],
                        uniform=self.normal_initializer),
                    trainable=True)
        else:
            self.labels_filter = \
                helpers_mk2.caps2scalar_filter(
                    self.caps_filter,
                    self.c,
                    self.eps)

        # CAPS COMPARE FILTER
        self.compare_filter = \
            tf.get_variable(
                name='%s_compare_filter' % (self.name),
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=[1, 1] + [self.f * 3, self.f],
                    uniform=self.normal_initializer),
                trainable=True)
        # self.compare_filter = None

        # LABEL GATES
        self.labels_deletegate_filter = \
            tf.get_variable(
                name='%s_labels_deletegate_filter' % (self.name),
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=[self.c_h_ext, self.c_h_ext] + [self.f*2, self.f],
                    uniform=self.normal_initializer),
                trainable=True)
        # self.labels_writegate_filter = \
        #     tf.get_variable(
        #         name='%s_labels_writegate_filter' % (self.name),
        #         dtype=self.dtype,
        #         initializer=initialization.xavier_initializer(
        #             shape=[self.c_h_ext, self.c_h_ext] + [self.f, self.f],
        #             uniform=self.normal_initializer),
        #         trainable=True)
        self.labels_writegate_filter = None

        # STATIC BIASES
        self.bundle_competition_bias = \
            tf.get_variable(
                name='%s_bundle_competition_bias' % (self.name),
                dtype=self.dtype,
                shape=[1, 1, 1, self.b],
                initializer=tf.constant_initializer([1.0], dtype=self.dtype),
                trainable=True)
        # self.bundle_competition_bias = \
        #     tf.constant(
        #         name='%s_bundle_competition_bias' % (self.name),
        #         dtype=self.dtype,
        #         shape=[1, 1, 1, self.b],
        #         value=1.0)
        self.fig_ground_competition_bias = \
            tf.get_variable(
                name='%s_fig_ground_competition_bias' % (self.name),
                dtype=self.dtype,
                shape=[1, 1, 1, 2],
                initializer=tf.constant_initializer([1.0], dtype=self.dtype),
                trainable=True)
        # self.fig_ground_competition_bias = \
        #     tf.constant(
        #         name='%s_fig_ground_competition_bias' % (self.name),
        #         dtype=self.dtype,
        #         shape=[1, 1, 1, 2],
        #         value=1.0)
        self.update_gain = \
            tf.get_variable(
                name='%s_update_gain' % (self.name),
                dtype=self.dtype,
                shape=[1, 1, 1, self.f],
                initializer=tf.constant_initializer([0.0], dtype=self.dtype),
                trainable=True)

        import numpy as np
        fixed_labels_mask = np.zeros([1, 1, 1, self.f])
        if self.fixed_label_ind is not None:
            for idx in self.fixed_label_ind:
                fixed_labels_mask[0, 0, 0, idx] = 1.
            self.fixed_labels_mask = tf.constant(fixed_labels_mask, dtype=self.dtype)
        else:
            self.fixed_labels_mask = None



    def full(self, i0, caps, labels, labels_spt_delta, labels_tmp_delta):
        """ DEPRECIATED
        """
        labels_new = helpers_mk2.label_backward(
                                caps, labels, caps, labels,
                                self.caps_filter, self.compare_filter, self.labels_filter,
                                self.labels_deletegate_filter, self.labels_writegate_filter,
                                self.bundle_competition_bias, self.fig_ground_competition_bias, self.update_gain,
                                self.b, self.c, self.c,
                                batchnorm_var_scope=self.batchnorm_var_scope)
        # Iterate loop
        if self.fixed_labels_mask is not None:
            labels_new = tf.multiply(self.fixed_labels_mask, labels) + \
                         tf.multiply(1 - self.fixed_labels_mask, labels_new)

        # get label spatial delta
        delta = tf.sqrt(tf.square(labels_new - labels))
        labels_spt_delta += delta

        # get label temporal delta
        write_head = tf.one_hot(indices = i0, depth=self.timesteps, dtype=self.dtype)
        write_head = tf.expand_dims(tf.expand_dims(write_head, axis=[0]), axis=[0])
        write_head = tf.tile(write_head, [self.x_shape[0], 1, 1])
        write_head = tf.expand_dims(write_head, axis=-1)
        labels_tmp_delta += write_head*tf.reduce_sum(delta, axis=[1, 2, 3], keep_dims=True)
        i0 += 1
        return i0, caps, labels_new, labels_spt_delta, labels_tmp_delta



    def condition(self, i0, caps, labels, labels_change, labels_tmp_delta):
        """While loop halting condition."""
        return i0 < self.timesteps



    def build(self, caps, labels):
        self.prepare_tensors()
        i0 = tf.constant(0)
        labels_spt_delta = tf.zeros_like(labels)
        labels_tmp_delta = tf.zeros([self.x_shape[0],1, self.timesteps,1])
        if self.use_while_loop:
            elms = [i0, caps, labels, labels_spt_delta, labels_tmp_delta]
            returned = tf.while_loop(
                self.condition,
                self.full,
                loop_vars = elms,
                back_prop=True,
                swap_memory=True)
            i0, caps, labels_new, labels_spt_delta, labels_tmp_delta = returned
        else:
            labels_old = labels
            for t in range(self.timesteps):
                labels_new = helpers_mk2.label_backward(
                                caps, labels_old, caps, labels_old,
                                self.caps_filter, self.compare_filter, self.labels_filter,
                                self.labels_deletegate_filter, self.labels_writegate_filter,
                                self.bundle_competition_bias, self.fig_ground_competition_bias, self.update_gain,
                                self.b, self.c, self.c,
                                batchnorm_var_scope=self.batchnorm_var_scope)

                # Iterate loop
                if self.fixed_labels_mask is not None:
                    labels_new = tf.multiply(self.fixed_labels_mask, labels_old) + \
                                 tf.multiply(1 - self.fixed_labels_mask, labels_new)

                # get label spatial delta
                delta = tf.sqrt(tf.square(labels_new - labels))
                labels_spt_delta += delta

                # get label temporal delta
                write_head = tf.one_hot(indices=i0, depth=self.timesteps, dtype=self.dtype)
                write_head = tf.expand_dims(tf.expand_dims(write_head, axis=[0]), axis=[0])
                write_head = tf.tile(write_head, [self.x_shape[0], 1, 1])
                write_head = tf.expand_dims(write_head, axis=-1)
                labels_tmp_delta += write_head * tf.reduce_sum(delta, axis=[1, 2, 3], keep_dims=True)

                labels_old = labels_new
        return caps, labels_new, labels_spt_delta, labels_tmp_delta

