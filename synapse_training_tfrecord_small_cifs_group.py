import os
import time
import logging
import argparse
import numpy as np
from config import Config
from membrane.models import seung_unet3d_adabn_small_group as unet
from utils.hybrid_utils import pad_zeros
from tqdm import tqdm
import pandas as pd
from lxml import etree
from scipy.spatial import distance
from glob import glob
from ops import data_loader
from ops import data_to_tfrecords
import tensorflow as tf
from matplotlib import pyplot as plt


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train(
        ffn_transpose=(0, 1, 2),
        path_extent=None,  # [1, 1, 1],
        cube_size=128,
        epochs=100,
        lr=1e-2,
        train_dataset='/media/data_cifs/connectomics/tf_records/synapses_v6_train.tfrecords',
        test_dataset='/media/data_cifs/connectomics/tf_records/synapses_v6_val.tfrecords',  # Needs to be val
        batch_size=32,
        #  image_size=(1, 160, 160, 160, 2),
        #  label_size=(1, 160, 160, 160, 2),
        summary_dir='tf_summaries/',
        steps_to_save=5000,
        ckpt_path='new_synapse_checkpoints_new_dataloader_bigger_weight/',
        rotate=False):
    """Apply the FFN routines using fGRUs."""
    config = Config()
    if isinstance(path_extent, str):
        path_extent = np.array([int(s) for s in path_extent.split(',')])
    synapse_files = np.array(glob(os.path.join(config.synapse_vols, '*.npz')))
    model_shape = list(np.load(synapse_files[0])['vol'].shape)[1:]
    label_shape = list(np.load(synapse_files[0])['label'].shape)[1:]

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
        idx = 0
        while not coord.should_stop():
            # TODO: Change to CCE
            feed_dict = {train_dict['lr']: lr}
            ret = sess.run(train_dict, feed_dict=feed_dict)
            train_loss = ret['train_loss']
            train_acc = ret['train_accuracy']
            train_f1 = ret['train_f1']
            train_precision = ret['train_precision']
            train_recall = ret['train_recall']
            pos_weight = ret['pos_weight']
            end = time.time()
            print(
                'Time elapsed: {:.2f} |'
                'Step: {} |'
                'Synapse prediction loss: {:.4f} | '
                'Pos weight 0, 1: {:.1f} {:.1f} | '
                'Accuracy: {:.2f} | '
                'F1: {:.2f} | '
                'Recall: {:.2f} |'
                'Precision: {:.2f}'.format(
                    end - start,
                    idx,
                    train_loss,
                    pos_weight[0], pos_weight[1],
                    train_acc,
                    train_f1,
                    train_recall,
                    train_precision))
            if idx % steps_to_save == 0:
                print('Saving checkpoint to: {}'.format(ckpt_path))
                saver.save(
                    sess,
                    ckpt_path,
                    global_step=idx)
                # f = plt.figure()
                # plt.subplot(141)
                # plt.imshow(ret['train_labels'][0, 64, ..., 0])
                # plt.subplot(142)
                # plt.imshow(ret['train_images'][0, 64, ..., 1])
                # plt.subplot(143)
                # plt.imshow(ret['train_images'][0, 64, ..., 0])
                # plt.subplot(144)
                # plt.imshow(ret['train_logits'][0, 64, ..., 0])
                # plt.savefig('{}.png'.format(idx))
                # plt.close(f)
                np.savez(
                    os.path.join(ckpt_path, '{}'.format(idx)),
                    labels=ret['train_labels'][0, 64, ...],
                    images=ret['train_images'][0, 64, ...],
                    preds=ret['train_logits'][0, 64, ...])
            idx += 1
    except tf.errors.OutOfRangeError:
        print(
            'Done training for {} epochs, {} steps.'.format(
                epochs, idx))
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

