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
"""Runs FFN inference within a dense bounding box.

Inference is performed within a single process.
"""

import os
import time

from google.protobuf import text_format
from absl import app
from absl import flags
from tensorflow import gfile
import tensorflow as tf
from ffn.utils import bounding_box_pb2
from ffn.inference import inference
from ffn.inference import inference_flags
import train_functional


FLAGS = flags.FLAGS

flags.DEFINE_string('bounding_box', None,
                    'BoundingBox proto in text format defining the area '
                    'to segmented.')


def main(unused_argv):
  request = inference_flags.request_from_flags()
  if not gfile.Exists(request.segmentation_output_dir):
    gfile.MakeDirs(request.segmentation_output_dir)

  bbox = bounding_box_pb2.BoundingBox()
  text_format.Parse(FLAGS.bounding_box, bbox)

  # Training
  import os
  batch_size = 16
  max_steps = 3000#10*250/batch_size #250
  hdf_dir = os.path.split(request.image.hdf5)[0]
  load_ckpt_path = request.model_checkpoint_path
  save_ckpt_path = os.path.split(load_ckpt_path)[0]+'_topup_'+ os.path.split(os.path.split(hdf_dir)[0])[1]
  # import ipdb;ipdb.set_trace()
  with tf.Graph().as_default():
      with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks, merge_devices=True)):
          # SET UP TRAIN MODEL
          print('>>>>>>>>>>>>>>>>>>>>>>SET UP TRAIN MODEL')

          TA = train_functional.TrainArgs(train_coords= os.path.join(hdf_dir, 'tf_record_file'),
                                          data_volumes='jk:' + os.path.join(hdf_dir, 'grayscale_maps.h5') + ':raw',
                                          label_volumes='jk:' + os.path.join(hdf_dir, 'groundtruth.h5') + ':stack',
                                          train_dir=save_ckpt_path,
                                          model_name=request.model_name,
                                          model_args=request.model_args,
                                          image_mean=request.image_mean,
                                          image_stddev=request.image_stddev,
                                          max_steps=max_steps,
                                          optimizer='adam',
                                          load_from_ckpt=load_ckpt_path,
                                          batch_size=batch_size)
          global TA
          model_class = import_symbol(TA.model_name)
          seed = int(time.time() + TA.task * 3600 * 24)
          logging.info('Random seed: %r', seed)
          random.seed(seed)
          eval_tracker, model, secs, load_data_ops, summary_writer, merge_summaries_op = \
                      build_train_graph(model_class, TA,
                                        save_ckpt=False, with_membrane=TA.with_membrane, **json.loads(TA.model_args))


          # SET UP INFERENCE MODEL
          print('>>>>>>>>>>>>>>>>>>>>>>SET UP INFERENCE MODEL')
          print('>>>>>>>>>>>>>>>>>>>>>>COUNTED %s VARIABLES PRE-INFERENCE' % len(tf.trainable_variables()))
          runner = inference.Runner()
          runner.start(
              request,
              batch_size=1,
              topup={'train_dir': FLAGS.train_dir},
              reuse=tf.AUTO_REUSE,
              tag='_inference') #TAKES SESSION
          print('>>>>>>>>>>>>>>>>>>>>>>COUNTED %s VARIABLES POST-INFERENCE' % len(tf.trainable_variables()))

          # START TRAINING
          print('>>>>>>>>>>>>>>>>>>>>>>START TOPUP TRAINING')
          sess = train_functional.train_ffn(
              TA, eval_tracker, model, runner.session, load_data_ops, summary_writer, merge_summaries_op)

          # saver.save(sess, "/tmp/model.ckpt")

          # START INFERENCE
          print('>>>>>>>>>>>>>>>>>>>>>>START INFERENCE')
          # saver.restore(sess, "/tmp/model.ckpt")
          runner.run((bbox.start.z, bbox.start.y, bbox.start.x),
                     (bbox.size.z, bbox.size.y, bbox.size.x))

          counter_path = os.path.join(request.segmentation_output_dir, 'counters.txt')
          if not gfile.Exists(counter_path):
            runner.counters.dump(counter_path)

          sess.close()

if __name__ == '__main__':
  app.run(main)
