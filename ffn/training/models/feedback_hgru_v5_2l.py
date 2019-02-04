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
def _predict_object_mask(input_patches, input_seed, depth=9, is_training=True):
  """Computes single-object mask prediction."""

  in_k = 14
  ff_k = [18, 18]
  x = tf.contrib.layers.conv3d(tf.concat([input_patches], axis=4),
                                 scope='conv0_a',
                                 num_outputs=in_k,
                                 kernel_size=(1, 7, 7),
                                 padding='SAME')

  from .prc import feedback_hgru_v5_2l
  with tf.variable_scope('recurrent'):
      hgru_net = feedback_hgru_v5_2l.hGRU(layer_name='hgru_net',
                                        num_in_feats=in_k,
                                        timesteps=8, #3,
                                        h_repeat=1, #2,
                                        hgru_dhw=[[1, 7, 7], [3, 5, 5]],
                                        hgru_k=[in_k, ff_k[0]],
                                        hgru_symmetric_weights=True,
                                        ff_conv_dhw=[[1, 7, 7], [1, 5, 5]],
                                        ff_conv_k=ff_k,
                                        ff_kpool_multiplier=2,
                                        ff_pool_dhw=[[1, 2, 2], [2, 2, 2]],
                                        ff_pool_strides=[[1, 2, 2], [2, 2, 2]],
                                        fb_mode='transpose',
                                        fb_dhw=[[1, 8, 8], [2, 6, 6]],
                                        fb_k=ff_k,
                                        padding='SAME',
                                        batch_norm=True,
                                        bn_reuse=False, ## TRUE NOT COMPLETELY IMPLEMENTED
                                        gate_bn=True,
                                        aux=None,
                                        train=is_training)

      net = hgru_net.build(x, input_seed)
  finalbn_param_initializer = {
      'moving_mean': tf.constant_initializer(0., dtype=tf.float32),
      'moving_variance': tf.constant_initializer(1., dtype=tf.float32),
      'gamma': tf.constant_initializer(0.1, dtype=tf.float32)
  }
  finalbn_param_trainable = {
      'moving_mean': False,
      'moving_variance': False,
      'gamma': is_training
  }
  net = tf.contrib.layers.batch_norm(
      inputs=net,
      scale=True,
      center=False,
      fused=True,
      renorm=False,
      param_initializers=finalbn_param_initializer,
      updates_collections=None,
      is_training=finalbn_param_trainable)
  logits = tf.contrib.layers.conv3d(net,
                                    scope='conv_lom1',
                                    num_outputs=in_k,
                                    kernel_size=(1, 1, 1),
                                    activation_fn=None)
  logits = tf.contrib.layers.batch_norm(
      inputs=logits,
      scale=True,
      center=False,
      fused=True,
      renorm=False,
      param_initializers=finalbn_param_initializer,
      updates_collections=None,
      is_training=finalbn_param_trainable)
  logits = tf.contrib.layers.conv3d(logits,
                                    scope='conv_lom2',
                                    num_outputs=1,
                                    kernel_size=(1, 1, 1),
                                    activation_fn=None)
  import numpy as np
  extras = 0
  hgru_w = 0
  ff_fb = 0
  for x in tf.trainable_variables():
      prod = np.prod(x.get_shape().as_list())
      if ('hgru' in x.name):
          if ('W' in x.name):
              hgru_w += prod/4
          elif ('mlp' in x.name):
              hgru_w += prod
          else:
              print(x.name + ' '+ str(prod))
              extras += prod
      elif ('ff' in x.name) | ('fb' in x.name) | ('conv0' in x.name) | ('conv_lom' in x.name):
          if ('weight' in x.name):
              ff_fb += prod
          else:
              print(x.name + ' ' + str(prod))
              extras += prod
      else:
          print(x.name + ' ' + str(prod))
          extras += prod
  hgru_w = int(hgru_w)
  print('>>>>>>>>>>>>>>>>>>>>>>TRAINABLE VARS: ' + 'horizontal('+str(hgru_w)+') vertical('+str(ff_fb)+') extras('+str(extras)+')')
  print('>>>>>>>>>>>>>>>>>>>>>>TRAINABLE VARS: ' + 'total(' + str(hgru_w+ff_fb+extras) + ')')
  print('>>>>>>>>>>>>>>>>>>>>>>IS_TRAINING: ' + str(is_training))
  return logits


class ConvStack3DFFNModel(model.FFNModel):
  dim = 3

  def __init__(self, fov_size=None, deltas=None, batch_size=None, depth=9, is_training=True, reuse=False, tag=''):
    super(ConvStack3DFFNModel, self).__init__(deltas, batch_size, tag=tag)
    self.set_uniform_io_size(fov_size)
    self.depth = depth
    self.is_training = is_training
    self.reuse=reuse

  def define_tf_graph(self):
    self.show_center_slice(self.input_seed)

    if self.input_patches is None:
      self.input_patches = tf.placeholder(
          tf.float32, [1] + list(self.input_image_size[::-1]) +[1],
          name='patches')

    with tf.variable_scope('seed_update', reuse=self.reuse):
      logit_update = _predict_object_mask(self.input_patches, self.input_seed, self.depth, is_training=self.is_training)

    logit_seed = self.update_seed(self.input_seed, logit_update)

    # Make predictions available, both as probabilities and logits.
    self.logits = logit_seed
    self.logistic = tf.sigmoid(logit_seed)

    if self.labels is not None:
      self.set_up_sigmoid_pixelwise_loss(logit_seed)
      self.set_up_optimizer()
      self.show_center_slice(logit_seed)
      self.show_center_slice(self.labels, sigmoid=False)
      self.add_summaries()

    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
