import sys
import os
import subprocess
import sys
import numpy as np
import train_functional

if __name__ == '__main__':

    batch_size = int(sys.argv[1])

    script_root = '/home/drew/ffn'
    net_name_obj = 'convstack_3d_in' #'convstack_3d_bn' #'feedback_hgru_v5_3l_notemp' #'feedback_hgru_generic_longfb_3l_long'#'feedback_hgru_generic_longfb_3l' #'feedback_hgru_3l_dualch' #'feedback_hgru_2l'  # 'convstack_3d'
    net_name = net_name_obj
    # volumes_name_list = ['isbi2013']
    # volumes_name_list = ['neuroproof']
    # volumes_name_list = ['isbi2013',
    #                      'cremi_a',
    #                      'cremi_b',
    #                      'cremi_c',
    #                      'berson']
    # volumes_name_list = ['neuroproof',
    #                       'isbi2013',
    #                      'cremi_a',
    #                      'cremi_b',
    #                      'cremi_c']
    # volumes_name_list = ['neuroproof',
    #		         'cremi_a',
    #			 'cremi_b',
    # 			 'cremi_c',
    #			 'berson']
    volumes_name_list = ['neuroproof',
			 'isbi2013',
			 'berson']
    # volumes_name_list = ['cremi_a',
    #                      'cremi_b',
    #                      'cremi_c']
    # volumes_name_list = ['berson_w_memb']
    tfrecords_name = 'allbutcremi'
    dataset_type = 'train' #'val' #'train'
    with_membrane = False

    # fov_type = 'traditional_fov'
    # fov_size = [33, 33, 33]
    # deltas = [8, 8, 8]
    # fov_type = 'flat_fov'
    # fov_size = [41, 41, 21]
    # deltas = [10, 10, 5]
    fov_type = 'wide_fov'
    fov_size = [57, 57, 13]
    deltas = [8, 8, 3]

    hdf_root = os.path.join('/media/data_cifs/connectomics/datasets/third_party/', fov_type)
    ckpt_root = os.path.join('/media/data_cifs/connectomics/ffn_ckpts', fov_type)

    validation_mode = False
    adabn = True
    load_from_ckpt = 'None'
    # ISBI2013
    # load_from_ckpt = '/media/data_cifs/connectomics/ffn_ckpts/wide_fov/convstack_3d_bn_isbi2013_r0/model.ckpt-615071'
    # load_from_ckpt = '/media/data_cifs/connectomics/ffn_ckpts/wide_fov/feedback_hgru_v5_3l_notemp_isbi2013_r0/model.ckpt-427216'
    # allbutfib
    # load_from_ckpt = '/media/data_cifs/connectomics/ffn_ckpts/wide_fov/convstack_3d_bn_allbutfib_r0/model.ckpt-596887'
    # load_from_ckpt = '/media/data_cifs/connectomics/ffn_ckpts/wide_fov/feedback_hgru_v5_3l_notemp_allbutfib_r0/model.ckpt-557947'
    # allburberson
    # load_from_ckpt = '/media/data_cifs/connectomics/ffn_ckpts/wide_fov/convstack_3d_bn_allbutberson_r0/model.ckpt-598310'
    # load_from_ckpt = '/media/data_cifs/connectomics/ffn_ckpts/wide_fov/feedback_hgru_v5_3l_notemp_allbutberson_r0/model.ckpt-488533'
    # cremi_abc
    # load_from_ckpt = '/media/data_cifs/connectomics/ffn_ckpts/wide_fov/convstack_3d_bn_cremi_abc_r0/model.ckpt-583777'
    # load_from_ckpt = '/media/data_cifs/connectomics/ffn_ckpts/wide_fov/feedback_hgru_v5_3l_notemp_cremi_abc_r0/model.ckpt-270552'

    max_steps = 16*1000000/batch_size
    optimizer = 'adam' #'adam' #'sgd'
    image_mean = 128
    image_stddev = 33

    print('>>>>>>>>>>>>>>>>>>>>> Dataset = ' + tfrecords_name)
    cond_name = net_name + '_' + tfrecords_name + '_r0' #+ str(irep)
    coords_fullpath = os.path.join(hdf_root, tfrecords_name, dataset_type, 'tf_record_file')

    data_string = ''
    label_string = ''
    for i, vol in enumerate(volumes_name_list):
        volume_fullpath = os.path.join(hdf_root, vol, dataset_type, 'grayscale_maps.h5')
        groundtruth_fullpath = os.path.join(hdf_root, vol, dataset_type, 'groundtruth.h5')
        if len(volumes_name_list)==1:
            partition_prefix='jk'
        else:
            partition_prefix=str(i)
        data_string += partition_prefix + ':' + volume_fullpath + ':raw'
        label_string += partition_prefix + ':' + groundtruth_fullpath + ':stack'
        if i < len(volumes_name_list)-1:
            data_string += ','
            label_string += ','


    command = 'python ' + os.path.join(script_root, 'train_old.py') + \
              ' --train_coords ' + coords_fullpath + \
              ' --data_volumes ' + data_string + \
              ' --label_volumes ' + label_string + \
              ' --train_dir ' + os.path.join(ckpt_root, cond_name) + \
              ' --model_name '+net_name_obj+'.ConvStack3DFFNModel' + \
              ' --model_args "{\\"depth\\": 12, \\"fov_size\\": ' + str(fov_size) + ', \\"deltas\\": ' + str(deltas) + '}"' + \
              ' --image_mean ' + str(image_mean) + \
              ' --image_stddev ' + str(image_stddev) + \
              ' --max_steps=' + str(max_steps) + \
              ' --optimizer ' + optimizer + \
              ' --load_from_ckpt ' + load_from_ckpt + \
              ' --batch_size=' + str(batch_size) + \
              ' --with_membrane=' + str(with_membrane) + \
              ' --validation_mode=' + str(validation_mode) + \
              ' --adabn=' + str(adabn)

    ############# TODO(jk): USE DATA VOLUMES FOR MULTI VOLUME TRAINING????
    subprocess.call(command, shell=True)

    # import tensorflow as tf
    # with tf.Graph().as_default():
    #     with tf.device(tf.train.replica_device_setter(0, merge_devices=True)):
    #     # with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks, merge_devices=True)):
    #         # SET UP TRAIN MODEL
    #         print('>>>>>>>>>>>>>>>>>>>>>>SET UP TRAIN MODEL')
    #         import logging
    #         from ffn.training.import_util import import_symbol
    #         import time
    #         import random
    #         import json
    #         TA = train_functional.TrainArgs(train_coords=coords_fullpath,
    #                                         data_volumes=data_string,
    #                                         label_volumes=label_string,
    #                                         train_dir=os.path.join(ckpt_root, cond_name),
    #                                         model_name=net_name_obj+'.ConvStack3DFFNModel',
    #                                         model_args='{"depth": 12, "fov_size": ' + str(fov_size) + ', "deltas": ' + str(deltas) + '}',
    #                                         image_mean=image_mean,
    #                                         image_stddev=image_stddev,
    #                                         max_steps=max_steps,
    #                                         optimizer=optimizer,
    #                                         load_from_ckpt=load_from_ckpt,
    #                                         batch_size=batch_size,
    #                                         with_membrane=with_membrane)
    #         model_class = import_symbol(TA.model_name)
    #         seed = int(time.time() + TA.task * 3600 * 24)
    #         logging.info('Random seed: %r', seed)
    #         random.seed(seed)
    #         eval_tracker, model, secs, load_data_ops, summary_writer, merge_summaries_op = \
    #             train_functional.build_train_graph(model_class, TA, save_ckpt=False,
    #                                                **json.loads(TA.model_args))
    #
    #         # # START SESSION
    #         saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.25)
    #         scaffold = tf.train.Scaffold(saver=saver)
    #         sess = tf.train.MonitoredTrainingSession(
    #             master=TA.master,
    #             is_chief=(TA.task == 0),
    #             save_summaries_steps=30,
    #             save_checkpoint_secs=secs,  # 10000/FLAGS.batch_size, #save_checkpoint_steps=10000/FLAGS.batch_size,
    #             config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True),
    #             checkpoint_dir=TA.train_dir,
    #             scaffold=scaffold)
    #         # Start supervisor.
    #         # sv = tf.train.Supervisor(
    #         #     logdir=TA.train_dir,
    #         #     is_chief=(TA.task == 0),
    #         #     saver=model.saver,
    #         #     summary_op=None,
    #         #     save_summaries_secs=60,
    #         #     save_model_secs=secs,
    #         #     recovery_wait_secs=5)
    #         # sess = sv.prepare_or_wait_for_session(
    #         #     TA.master,
    #         #     config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
    #         # START TRAINING
    #         print('>>>>>>>>>>>>>>>>>>>>>>START TRAINING')
    #         sess = train_functional.train_ffn(
    #             TA, eval_tracker, model, sess, load_data_ops, summary_writer, merge_summaries_op)
    #
