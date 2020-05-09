"""Contextual model with partial filters."""
import operator
from functools import reduce

import tensorflow as tf
from layers.feedforward import normalization
from layers.feedforward.aLN import helpers

from layers.feedforward.pooling import max_pool
from ops import initialization

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
            h_ext=15,
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
        self.h_ext = h_ext
        self.train = train

        # Sort through and assign the auxilliary variables
        default_vars = self.defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)

        # Handle batchnorm scope reuse
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
            'dtype': tf.float32,
            'train': True,
            'normal_initializer': True,
            'reuse': False,
            'batch_norm': False,  # Not working
            'bn_per_time':False,
            'eps': 0.000001,
            'use_independent_labels_filter':True,
            'labels_update_mode': 'pushpull',
            'use_labels_gate':False,
            'cap_labels':False,
            'fixed_label_ind': None,
            }



    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)



    def prepare_tensors(self):
        """
        """

        # recurrent kernels
        setattr(
            self,
            'caps_filter',
            tf.get_variable(
                name='%s_caps_filter' % (self.name),
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=[self.h_ext, self.h_ext] + [self.f*self.c, self.b*self.f*self.c],
                    uniform=self.normal_initializer),
                trainable=True))
        if self.use_independent_labels_filter:
            setattr(
                self,
                'labels_filter',
                tf.get_variable(
                    name='%s_labels_filter' % (self.name),
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=[self.h_ext, self.h_ext] + [self.f, self.b * self.f],
                        uniform=self.normal_initializer),
                    trainable=True))
        else:
            setattr(
                self,
                'labels_filter',
                helpers.caps2scalar_filter(self.caps_filter,self.c))

        # label update variables
        if self.labels_update_mode == 'pushpull':
            setattr(
                self,
                'pushpull_scale',
                tf.get_variable(
                    name='%s_pushpull_scale' % (self.name),
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=[1, 1, 1, self.f],
                        uniform=self.normal_initializer),
                    trainable=True))
            setattr(
                self,
                'pushpull_bias',
                tf.get_variable(
                    name='%s_pushpull_bias' % (self.name),
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=[1, 1, 1, self.f],
                        uniform=self.normal_initializer),
                    trainable=True))
            self.sensitivity_const=None
        elif self.labels_update_mode == 'average':
            setattr(
                self,
                'sensitivity_const',
                tf.get_variable(
                    name='%s_sensitivity_const' % (self.name),
                    dtype=self.dtype,
                    shape=[1, 1, 1, self.f],
                    initializer=tf.constant_initializer([0.5], dtype=self.dtype),
                    trainable=True))
            self.pushpull_scale=None
            self.pushpull_bias=None

        # label gate variables
        if self.use_labels_gate:
            setattr(
                self,
                'labels_gate_scale',
                tf.get_variable(
                    name='%s_labels_gate_scale' % (self.name),
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=[1, 1, 1, self.f],
                        uniform=self.normal_initializer),
                    trainable=True))
            setattr(
                self,
                'labels_gate_bias',
                tf.get_variable(
                    name='%s_labels_gate_bias' % (self.name),
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=[1, 1, 1, self.f],
                        uniform=self.normal_initializer),
                    trainable=True))
        else:
            self.labels_gate_scale = None
            self.labels_gate_bias = None

        # Auxilliary variables
        setattr(
            self,
            'tolerance_const',
            tf.get_variable(
                name='%s_tolerance_const' % (self.name),
                dtype=self.dtype,
                shape=[1, 1, 1, self.f],
                initializer=tf.constant_initializer([0.], dtype=self.dtype),
                trainable=True))
        setattr(
            self,
            'decay_const',
            tf.get_variable(
                name='%s_decay_const' % (self.name),
                dtype=self.dtype,
                shape=[1, 1, 1, self.f],
                initializer=tf.constant_initializer([0.0], dtype=self.dtype),
                trainable=True))
        import numpy as np
        fixed_labels_mask = np.zeros([1, 1, 1, self.f])
        if self.fixed_label_ind is not None:
            for idx in self.fixed_label_ind:
                fixed_labels_mask[0, 0, 0, idx] = 1.
        self.fixed_labels_mask = tf.constant(fixed_labels_mask, dtype=self.dtype)



    def full(self, i0, caps_match, caps2_norm, labels, labels_net_change):
        """ DEPRECIATED
        """
        labels_new = helpers.label_backward(
            labels,
            labels,
            caps_match,
            self.labels_filter,
            self.decay_const,
            self.b,
            labels_gate_scale=self.labels_gate_scale,
            labels_gate_bias=self.labels_gate_bias,
            caps1_norm=caps2_norm,
            labels_update_mode=self.labels_update_mode,
            sensitivity_const=self.sensitivity_const,
            pushpull_scale=self.pushpull_scale,
            pushpull_bias=self.pushpull_bias,
            cap_labels=self.cap_labels,
            fixed_labels_mask=self.fixed_labels_mask)
        if self.bn_per_time:
            print('horizontal: using batchnorm per ts')
            labels_new = normalization.batch(
                bottom=labels_new,
                renorm=True,
                name='in_model_bn_' + str(i0),
                training=True)
        # Iterate loop
        labels_net_change += tf.sqrt(tf.square(labels_new - labels))
        i0 += 1
        return i0, caps_match, caps2_norm, labels_new, labels_net_change



    def condition(self, i0, caps_match, caps2_norm, labels, labels_accumulator):
        """While loop halting condition."""
        return i0 < self.timesteps



    def build(self, caps, labels):
        self.prepare_tensors()
        i0 = tf.constant(0)
        labels_change = tf.zeros_like(labels)
        # Custom iterations
        caps_intermediate1, caps_match, caps1_norm, caps2_norm = \
            helpers.get_caps_match(
                caps,
                caps,
                self.caps_filter,
                self.tolerance_const,
                self.c,
                self.b,
                self.eps)

        if self.use_while_loop:
            elms = [i0, caps_match, caps2_norm, labels, labels_change]
            returned = tf.while_loop(
                self.condition,
                self.full,
                loop_vars = elms,
                back_prop=True,
                swap_memory=False)
            i0, caps_match, caps2_norm, labels, labels_change = returned
        else:
            labels_old = labels
            for t in range(self.timesteps):
                labels = helpers.label_backward(
                    labels_old,
                    labels_old,
                    caps_match,
                    self.labels_filter,
                    self.decay_const,
                    self.b,
                    labels_gate_scale=self.labels_gate_scale,
                    labels_gate_bias=self.labels_gate_bias,
                    caps1_norm=caps2_norm,
                    labels_update_mode=self.labels_update_mode,
                    sensitivity_const=self.sensitivity_const,
                    pushpull_scale=self.pushpull_scale,
                    pushpull_bias=self.pushpull_bias,
                    cap_labels=self.cap_labels,
                    fixed_labels_mask=self.fixed_labels_mask)
                if self.bn_per_time:
                    print('horizontal: using batchnorm per ts')
                    labels = normalization.batch(
                        bottom=labels,
                        renorm=True,
                        name='in_model_bn_' + str(i0),
                        training=True)
                # Iterate loop
                labels_change += tf.sqrt(tf.square(labels - labels_old))
                labels_old=labels
        return caps, labels, labels_change

