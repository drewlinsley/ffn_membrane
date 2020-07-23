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

  in_k = 24
  ff_k = [24, 28, 32]
  ff_kpool_multiplier = 2

  train_bn = True
  bn_decay = 0.95
  # if not is_training:
  #   if not adabn:
  #       bn_decay = 1.0
  #       train_bn = False

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
                                 kernel_size=(1, 5, 5),
                                 padding='SAME')
  if input_patches.get_shape().as_list()[-1] == 2:
      print('FFN-hgru-v5: using membrane as input')
      x = tf.concat([x, membrane], axis=4)
  x = tf.contrib.layers.conv3d(x,
                                 scope='conv0_b',
                                 num_outputs=x.get_shape().as_list()[-1],
                                 kernel_size=(1, 5, 5),
                                 padding='SAME')
  from .prc import feedback_hgru_v5_3l_nu_f_v2_in as feedback_hgru_v5_3l_nu_f
  with tf.variable_scope('recurrent'):
      hgru_net = feedback_hgru_v5_3l_nu_f.hGRU(layer_name='hgru_net',
                                        num_in_feats=in_k,
                                        timesteps=8, #6, #8,
                                        h_repeat=1,
                                        hgru_dhw=[[1, 11, 11], [3, 5, 5], [5, 5, 5]],
                                        hgru_k=[in_k, ff_k[0], ff_k[1]],
                                        hgru_symmetric_weights=False,
                                        ff_conv_dhw=[[1, 7, 7], [1, 5, 5], [1, 5, 5]],
                                        ff_conv_k=ff_k,
                                        ff_kpool_multiplier=ff_kpool_multiplier,
                                        ff_pool_dhw=[[1, 2, 2], [2, 2, 2], [1, 2, 2]],
                                        ff_pool_strides=[[1, 2, 2], [2, 2, 2], [1, 2, 2]],
                                        fb_mode='transpose',
                                        fb_dhw=[[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                                        fb_k=ff_k,
                                        padding='SAME',
                                        batch_norm=True,
                                        bn_reuse=False,
                                        gate_bn=True,
                                        aux=None,
                                        train=train_bn,
                                        bn_decay=bn_decay)

      net = hgru_net.build(x, input_seed)
  finalbn_param_initializer = {
      'moving_mean': tf.constant_initializer(0., dtype=tf.float32),
      'moving_variance': tf.constant_initializer(1., dtype=tf.float32),
      'gamma': tf.constant_initializer(0.1, dtype=tf.float32)
  }
  net = tf.contrib.layers.instance_norm(
      inputs=net,
      scale=True,
      center=True,
      param_initializers=finalbn_param_initializer,
      trainable=train_bn)
  logits = tf.contrib.layers.conv3d(net,
                                    scope='conv_lom1',
                                    num_outputs=in_k,
                                    kernel_size=(1, 5, 5),
                                    activation_fn=tf.nn.relu)
  # COMMENTED (326)
  # logits = tf.contrib.layers.batch_norm(
  #     inputs=logits,
  #     scale=False,
  #     center=False,
  #     fused=True,
  #     renorm=False,
  #     decay=bn_decay,
  #     param_initializers=finalbn_param_initializer,
  #     is_training=train_bn)
  logits = tf.contrib.layers.conv3d(logits,
                                    scope='conv_lom2',
                                    num_outputs=1,
                                    kernel_size=(1, 1, 1),
                                    activation_fn=None)
  # logits = tf.clip_by_value(logits, -4.5, 4.5) # (326)
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
  print('>>>>>>>>>>>>>>>>>>>>>>BN-TRAIN: ' + str(train_bn))
  return logits


class ConvStack3DFFNModel(model.FFNModel):
  dim = 3

  def __init__(self, with_membrane=False, fov_size=None, optional_output_size=None, deltas=None, batch_size=None, depth=9,
               is_training=True, adabn=False, reuse=False, tag='', TA=None, grad_clip_val=0.0):
    super(ConvStack3DFFNModel, self).__init__(deltas, batch_size, with_membrane, validation_mode=not(is_training), tag=tag)

    self.optional_output_size = optional_output_size
    self.set_uniform_io_size(fov_size)
    self.depth = depth
    self.reuse=reuse
    self.TA=TA
    self.is_training=is_training
    self.adabn=adabn
    if grad_clip_val is None:
        self.grad_clip_val = 0.0
    else:
        self.grad_clip_val = grad_clip_val

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
        self.set_up_optimizer(max_gradient_entry_mag=self.grad_clip_val)
      else:
        self.set_up_optimizer(max_gradient_entry_mag=self.grad_clip_val, TA=self.TA)

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
