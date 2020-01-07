import os
import time
import logging
import argparse
import numpy as np
from config import Config
from membrane.models import l3_fgru_constr as fgru
from membrane.models import seung_unet3d_adabn as unet
from utils.hybrid_utils import pad_zeros
from tqdm import tqdm
import pandas as pd
from lxml import etree
from scipy.spatial import distance
from glob import glob
from ops import data_loader
from ops import data_to_tfrecords
import tensorflow as tf


logger = logging.getLogger()
logger.setLevel(logging.INFO)
AUGMENTATIONS = [{'lr_flip': []}, {'ud_flip': []}]


def train(
        ffn_transpose=(0, 1, 2),
        path_extent=None,  # [1, 1, 1],
        cube_size=128,
        epochs=100,
        lr=1e-3,
        train_dataset='/media/data_cifs/connectomics/tf_records/synapses_v0_train.tfrecords',
        test_dataset='/media/data_cifs/connectomics/tf_records/synapses_v0_train.tfrecords',  # Needs to be val
        batch_size=1,
        image_size=(1, 128, 128, 128, 2),
        label_size=(1, 128, 128, 128, 2),
        summary_dir='tf_summaries/',
        steps_to_save=1000,
        ckpt_path='synapse_checkpoints/',
        rotate=False):
    """Apply the FFN routines using fGRUs."""
    config = Config()
    if isinstance(path_extent, str):
        path_extent = np.array([int(s) for s in path_extent.split(',')])
    synapse_files = np.array(glob(os.path.join(config.synapse_vols, '*.npz')))
    model_shape = list(np.load(synapse_files[0])['vol'].shape)[1:]
    label_shape = list(np.load(synapse_files[0])['label'].shape)[1:]

    """
    # Create tfrecord loaders
    tf_reader = {
        'volume': {
            'dtype': tf.float32,
            'reshape': image_size
        },
        'label': {
            'dtype': tf.float32,
            'reshape': label_size
        }
    }
    tf_dict = {
        'volume': fixed_len_feature(dtype='string'),
        'label': fixed_len_feature(dtype='string')
    }
    train_images, train_labels = data_loader.inputs(
        dataset=train_dataset,
        batch_size=batch_size,
        input_shape=image_size,
        label_shape=label_size,
        tf_dict=tf_dict,
        data_augmentations=AUGMENTATIONS,
        num_epochs=epochs,
        tf_reader_settings=tf_reader,
        shuffle=True)
    test_images, test_labels = data_loader.inputs(
        dataset=test_dataset,
        batch_size=batch_size,
        input_shape=image_size,
        label_shape=label_size,
        tf_dict=tf_dict,
        data_augmentations=[],
        num_epochs=epochs,
        tf_reader_settings=tf_reader,
        shuffle=False)
    """

    # Get new model training variables
    sess, saver, train_dict, test_dict = unet.main(
        tf_records={'train_dataset': train_dataset, 'test_dataset': test_dataset},
        train_input_shape=model_shape,
        train_label_shape=label_shape,
        test_input_shape=model_shape,
        test_label_shape=label_shape,
        gpu_device='/gpu:0')

    # Start up tfrecord coordinators and threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Training loop
    try:
        start = time.time()
        while not coord.should_stop():
            # TODO: Change to CCE
            feed_dict = {train_dict['lr']: lr}
            ret = sess.run(train_dict, feed_dict=feed_dict)
            train_loss = ret['train_loss']
            train_f1 = ret['train_f1']
            train_precision = ret['train_precision']
            train_recall = ret['train_recall']
            end = time.time()
            print(
                'Time elapsed: {} |'
                'Synapse prediction loss: {} | '
                'F1: {} | '
                'Recall: {} |'
                'Precision: {}'.format(
                    end - start,
                    train_loss,
                    train_f1,
                    train_recall,
                    train_precision))
            if idx % steps_to_save == 0:
                print('Saving checkpoint to: {}'.format())
                saver.save(
                    sess,
                    ckpt_path,
                    global_step=epoch)
    except tf.errors.OutOfRangeError:
        print(
            'Done training for {} epochs, {} steps.'.format(
                epochs, step))
        print('Saved to: {}'.format(ckpt_path))
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_extent',
        dest='path_extent',
        type=str,
        default='3,3,3',
        help='Provide extent of segmentation in 128^3 volumes.')
    args = parser.parse_args()
    start = time.time()
    train(**vars(args))
    end = time.time()
    print('Training took {}'.format(end - start))
