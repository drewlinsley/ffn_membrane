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
"""Utilities to configure TF optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from absl import flags


def optimizer_from_flags(TA):
  lr = TA.learning_rate
  if TA.optimizer == 'momentum':
    return tf.train.MomentumOptimizer(lr, TA.momentum)
  elif TA.optimizer == 'sgd':
    return tf.train.GradientDescentOptimizer(lr)
  elif TA.optimizer == 'adagrad':
    return tf.train.AdagradOptimizer(lr)
  elif TA.optimizer == 'adam':
    return tf.train.AdamOptimizer(learning_rate=lr,
                                  beta1=TA.adam_beta1,
                                  beta2=TA.adam_beta2,
                                  epsilon=TA.epsilon)
  elif TA.optimizer == 'rmsprop':
    return tf.train.RMSPropOptimizer(lr, TA.rmsprop_decay,
                                     momentum=TA.momentum,
                                     epsilon=TA.epsilon)
  else:
    raise ValueError('Unknown optimizer: %s' % TA.optimizer)
