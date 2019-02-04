# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simplest FFN model, as described in https://arxiv.org/abs/1611.00421."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .. import model


# Note: this model was originally trained with conv3d layers initialized with
# TruncatedNormalInitializedVariable with stddev = 0.01.
def _predict_object_mask(net, depth=9, is_training=True, adabn=False):
  """Computes single-object mask prediction."""

  conv = tf.contrib.layers.conv3d

  if not is_training:
    if not adabn:
        bn_decay = 1.0
  else:
    bn_decay = 0.99
  bn_decay = 1.0
  train_bn = False

  print(bn_decay)
  print(train_bn)
  with tf.contrib.framework.arg_scope([conv], num_outputs=32,
                                      kernel_size=(3, 3, 3),
                                      padding='SAME'):
    net = tf.contrib.layers.batch_norm(
                        inputs=net,
                        scale=True,
                        center=True,
                        fused=True,
                        renorm=False,
                        decay=bn_decay,
                        updates_collections=None,
                        param_initializers={'moving_mean': tf.constant_initializer(0.),
                                            'moving_variance': tf.constant_initializer(1.),
                                            'gamma': tf.constant_initializer(0.1)
                                            },
                        scope='in',
                        is_training=train_bn)
    net = conv(net, scope='conv0_a')
    net = conv(net, scope='conv0_b', activation_fn=None)

    for i in range(1, depth):
      with tf.name_scope('residual%d' % i):
        net = tf.contrib.layers.batch_norm(
                    inputs=net,
                    scale=True,
                    center=True,
                    fused=True,
                    renorm=False,
                    decay=bn_decay,
                    updates_collections=None,
                    param_initializers={'moving_mean': tf.constant_initializer(0.),
                                        'moving_variance': tf.constant_initializer(1.),
                                        'gamma': tf.constant_initializer(0.1)
                                        },
                    scope='res_%da'% i,
                    is_training=train_bn)
        in_net = net
        net = tf.nn.relu(net)
        net = conv(net, scope='conv%d_a' % i)
        net = conv(net, scope='conv%d_b' % i, activation_fn=None)
        net += in_net

  net = tf.contrib.layers.batch_norm(
                    inputs=net,
                    scale=True,
                    center=True,
                    fused=True,
                    renorm=False,
                    decay=bn_decay,
                    updates_collections=None,
                    param_initializers={'moving_mean': tf.constant_initializer(0.),
                                        'moving_variance': tf.constant_initializer(1.),
                                        'gamma': tf.constant_initializer(0.1)
                                        },
                    scope='out',
                    is_training=train_bn)
  net = tf.nn.relu(net)
  logits = conv(net, 1, (1, 1, 1), activation_fn=None, scope='conv_lom')

  import numpy as np
  acc = 0
  for x in tf.trainable_variables():
      prod = np.prod(x.get_shape().as_list())
      acc += prod
  print('>>>>>>>>>>>>>>>>>>>>>>TRAINABLE VARS: '+str(acc))
  print('>>>>>>>>>>>>>>>>>>>>>>BN-TRAIN: ' + str(train_bn))

  return logits


class ConvStack3DFFNModel(model.FFNModel):
  dim = 3

  def __init__(self, with_membrane=False, fov_size=None, deltas=None, batch_size=None, depth=9, is_training=True, adabn=False, reuse=False, tag='', TA=None):
    super(ConvStack3DFFNModel, self).__init__(deltas, batch_size, with_membrane, validation_mode=not(is_training), tag=tag)
    self.set_uniform_io_size(fov_size)
    self.depth = depth
    self.reuse = reuse
    self.TA = TA
    self.is_training=is_training
    self.adabn=adabn

  def define_tf_graph(self):
    self.show_center_slice(self.input_seed)

    if self.input_patches is None:
      self.input_patches = tf.placeholder(
          tf.float32, [1] + list(self.input_image_size[::-1]) +[1],
          name='patches')

    net = tf.concat([self.input_patches, self.input_seed], 4)

    with tf.variable_scope('seed_update', reuse=self.reuse):
      logit_update = _predict_object_mask(net, self.depth, is_training=self.is_training, adabn=self.adabn)

    logit_seed = self.update_seed(self.input_seed, logit_update)

    # Make predictions available, both as probabilities and logits.
    self.logits = logit_seed
    self.logistic = tf.sigmoid(logit_seed)

    if self.labels is not None:
      self.set_up_sigmoid_pixelwise_loss(logit_seed)
      if self.TA is None:
        self.set_up_optimizer()
      else:
        self.set_up_optimizer(TA=self.TA)
      self.show_center_slice(logit_seed)
      self.show_center_slice(self.labels, sigmoid=False)
      self.add_summaries()

    # ADABN: Add only non-bn vars to saver
    var_list = tf.global_variables()
    moving_ops_names = ['moving_mean:', 'moving_variance:']
    # var_list = [
    #       x for x in var_list
    #       if x.name.split('/')[-1].split(':')[0] + ':'
    #       not in moving_ops_names]
    # self.saver = tf.train.Saver(
    #       var_list=var_list,
    #       keep_checkpoint_every_n_hours=100)
    # ADABN: Define bn-var initializer to reset moments every iteration
    moment_list = [
      x for x in tf.global_variables()
      if x.name.split('/')[-1].split(':')[0] + ':'
      in moving_ops_names]
    self.moment_list = moment_list
    self.ada_initializer = tf.variables_initializer(
      var_list=moment_list)

    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
