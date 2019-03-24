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
def _predict_object_mask(input_patches, input_seed, depth=9, is_training=True, adabn=False):
  """Computes single-object mask prediction."""

  train_bn = True
  bn_decay = 0.95
  if not is_training:
    if not adabn:
        bn_decay = 1.0
        train_bn = False

  in_k = 14

  if input_patches.get_shape().as_list()[-1] == 2:
      image = tf.expand_dims(input_patches[:,:,:,:,0], axis=4)
      membrane = tf.expand_dims(input_patches[:,:,:,:,1], axis=4)
      image_k = in_k-1
  else:
      image = input_patches
      image_k = in_k

  x = tf.contrib.layers.conv3d(image,
                                 scope='conv0_a',
                                 num_outputs=image_k,
                                 kernel_size=(1, 12, 12),
                                 padding='SAME')
  if input_patches.get_shape().as_list()[-1] == 2:
      print('*' * 60)
      print('FFN-hgru-v5: using membrane as input')
      membrane = membrane*33 + 128.
      x = tf.concat([x, membrane], axis=4)

  from .layers.recurrent import htd_cnn_3l
  with tf.variable_scope('htd_net'):
      hgru_net = htd_cnn_3l.hGRU(var_scope='htd_net',
                              timesteps=8,
                              dtype=tf.float32,
                              use_3d=True,
                              train=is_training,
                              train_bn=train_bn,
                              use_in=False,
                              bn_decay=bn_decay,
                              in_k=in_k,

                              hgru1_fsiz=[1, 7, 7],
                              hgru2_fsiz=[3, 5, 5],
                              hgru3_fsiz=[3, 3, 3],
                              hgru_td3_fsiz=[1, 1, 1],
                              hgru_td2_fsiz=[1, 1, 1],
                              hgru_td1_fsiz=[1, 1, 1],
                              hgru_h1_nl=tf.nn.relu,
                              hgru_h2_nl=tf.nn.relu,
                              hgru_bistream_weights='independent',
                              hgru_in_place_integration=False, #########
                              hgru_symmetric_weights=True,
                              hgru_soft_coefficients=True,
                              belly_up_td=False,

                              ds_fsiz_list=[[1, 7, 7], [1, 5, 5], [1, 3, 3]],
                              ds_conv_repeat=1,
                              ds_k_list=[18, 18, 18],
                              ds_pool_list=[[1, 2, 2], [2, 2, 2], [1, 2, 2]],
                              ds_stride_list=[[1, 2, 2], [2, 2, 2], [1, 2, 2]],
                              use_trainable_states=True)

      net = hgru_net.build(x, ffn_seed=input_seed)

  finalbn_param_initializer = {
      'moving_mean': tf.constant_initializer(0., dtype=tf.float32),
      'moving_variance': tf.constant_initializer(1., dtype=tf.float32),
      'gamma': tf.constant_initializer(0.1, dtype=tf.float32)
  }
  net = tf.nn.relu(net)
  net = tf.contrib.layers.batch_norm(
      inputs=net,
      scale=True,
      center=True,
      fused=True,
      renorm=False,
      decay=bn_decay,
      param_initializers=finalbn_param_initializer,
      is_training=train_bn)
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
      decay=bn_decay,
      param_initializers=finalbn_param_initializer,
      is_training=train_bn)
  logits = tf.nn.relu(logits)
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
          if ('w' in x.name):
              hgru_w += prod/2
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
  print('>>>>>>>>>>>>>>>>>>>>>>BN-TRAIN: ' + str(train_bn))
  return logits


class ConvStack3DFFNModel(model.FFNModel):
  dim = 3

  def __init__(self, with_membrane=False, fov_size=None, optional_output_size=None, deltas=None, batch_size=None, depth=9, is_training=True, adabn=False, reuse=False, tag='', TA=None):
    super(ConvStack3DFFNModel, self).__init__(deltas, batch_size, with_membrane, validation_mode=not(is_training), tag=tag)

    self.optional_output_size = optional_output_size
    self.set_uniform_io_size(fov_size)
    self.depth = depth
    self.reuse=reuse
    self.TA=TA
    self.is_training=is_training
    self.adabn=adabn

  def define_tf_graph(self):
    self.show_center_slice(self.input_seed)

    if self.input_patches is None:
      self.input_patches = tf.placeholder(
          tf.float32, [1] + list(self.input_image_size[::-1]) +[1],
          name='patches')

    with tf.variable_scope('seed_update', reuse=self.reuse):
      logit_update = _predict_object_mask(self.input_patches, self.input_seed,
                                          depth=self.depth, is_training=self.is_training, adabn=self.adabn)
    if self.optional_output_size is not None:
      dx = self.input_seed_size[0] - self.optional_output_size[0]
      dy = self.input_seed_size[1] - self.optional_output_size[1]
      dz = self.input_seed_size[2] - self.optional_output_size[2]
      logit_update_cropped = logit_update[:,
                                          dz // 2: -(dz - dz // 2),
                                          dy // 2: -(dy - dy // 2),
                                          dx // 2: -(dx - dx // 2),
                                          :]
      logit_update_padded = tf.pad(logit_update_cropped, [[0, 0],
                                           [dz // 2, dz - dz // 2],
                                           [dy // 2, dy - dy // 2],
                                           [dx // 2, dx - dx // 2],
                                           [0, 0]])
      mask = tf.pad(tf.ones_like(logit_update_cropped), [[0, 0],
                                           [dz // 2, dz - dz // 2],
                                           [dy // 2, dy - dy // 2],
                                           [dx // 2, dx - dx // 2],
                                           [0, 0]])
      self.loss_weights *= mask
      logit_seed = self.update_seed(self.input_seed, logit_update_padded)
    else:
      logit_seed = self.update_seed(self.input_seed, logit_update)

    # Make predictions available, both as probabilities and logits.
    self.logits = logit_seed

    if self.labels is not None:
      self.logistic = tf.sigmoid(logit_seed)
      self.set_up_sigmoid_pixelwise_loss(logit_seed)
      self.show_center_slice(logit_seed)
      self.show_center_slice(self.labels, sigmoid=False)
      if self.TA is None:
        self.set_up_optimizer(max_gradient_entry_mag=0.0)
      else:
        self.set_up_optimizer(max_gradient_entry_mag=0.0, TA=self.TA)

      self.add_summaries()

    # ADABN: Add only non-bn vars to saver
    var_list = tf.global_variables()
    moving_ops_names = ['moving_mean', 'moving_variance']
    # var_list = [
    #       x for x in var_list
    #       if x.name.split('/')[-1].split(':')[0] + ':'
    #       not in moving_ops_names]
    # self.saver = tf.train.Saver(
    #       var_list=var_list,
    #       keep_checkpoint_every_n_hours=100)
    # ADABN: Define bn-var initializer to reset moments every iteration
    moment_list = [x for x in tf.global_variables() if (moving_ops_names[0] in x.name) | (moving_ops_names[1] in x.name)]
    self.moment_list = None
    # self.moment_list = moment_list
    self.ada_initializer = tf.variables_initializer( var_list=moment_list)

    self.fgru_moment_list = [x for x in moment_list if 'recurrent' in x.name]
    self.fgru_ada_initializer = tf.variables_initializer(var_list=self.fgru_moment_list)
    self.ext_moment_list =  [x for x in moment_list if 'recurrent' not in x.name]
    self.ext_ada_initializer = tf.variables_initializer(var_list=self.ext_moment_list)

    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
