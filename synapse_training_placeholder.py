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


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def augment(vo, augs):
    """Augment volume with augmentation au."""
    for au in augs:
        if au is 'rot90':
            for z in range(vo.shape[0]):
                vo[z] = np.rot90(vo[z], 1, (1, 2))
        elif au is 'rot180':
            for z in range(vo.shape[0]):
                vo[z] = np.rot90(vo[z], 2, (1, 2))
        elif au is 'rot270':
            for z in range(vo.shape[0]):
                vo[z] = np.rot90(vo[z], 3, (1, 2))
        elif au is 'lr_flip':
            vo = vo[..., ::-1]
        elif au is 'ud_flip':
            vo = vo[..., ::-1, :]
        elif au is 'depth_flip':
            vo = vo[..., ::-1, :, :]
    return vo


def undo_augment(vo, augs, debug_mem=None):
    """Augment volume with augmentation au."""
    for au in augs:
        if au is 'rot90':
            for z in range(vo.shape[1]):
                vo[0, z] = np.rot90(vo[0, z], -1, (1, 2))  # -90
        elif au is 'rot180':
            for z in range(vo.shape[1]):
                vo[0, z] = np.rot90(vo[0, z], -2, (1, 2))  # -180
        elif au is 'rot270':
            for z in range(vo.shape[1]):
                vo[0, z] = np.rot90(vo[0, z], -3, (1, 2))  # -270
        elif au is 'lr_flip':
            vo = vo[..., ::-1, :]  # Note: 3-channel volumes
        elif au is 'ud_flip':
            vo = vo[..., ::-1, :, :]
        elif au is 'depth_flip':
            vo = vo[..., ::-1, :, :, :]
    return vo


def train(
        ffn_transpose=(0, 1, 2),
        path_extent=None,  # [1, 1, 1],
        cube_size=128,
        epochs=100,
        lr=1e-3,
        seed_file='synapses/annotation.xml',
        summary_dir='tf_summaries/',
        ckpt_path='synapse_checkpoints/',
        rotate=False):
    """Apply the FFN routines using fGRUs."""
    config = Config()
    if isinstance(path_extent, str):
        path_extent = np.array([int(s) for s in path_extent.split(',')])
    synapse_files = np.array(glob(os.path.join(config.synapse_vols, '*.npz')))
    model_shape = list(np.load(synapse_files[0])['vol'].shape)[1:]
    label_shape = list(np.load(synapse_files[0])['label'].shape)[1:]

    # Pull out DB row to split CV folds
    labels = []
    for sy in synapse_files:
        labels += [np.load(sy)['row'][-1]]
    labels = np.array(labels)
    synapse_files = synapse_files[np.random.permutation(len(synapse_files))]
    train_idx = np.ones(len(synapse_files))
    train_idx[-100:] = 0  # TODO: Explicitly balance this
    val_idx = train_idx == 0
    train_idx = train_idx == 1
    train_files = synapse_files[train_idx]
    val_files = synapse_files[val_idx]

    # Get new model training variables
    sess, saver, train_dict, test_dict = unet.main(
        train_input_shape=model_shape,
        train_label_shape=label_shape,
        test_input_shape=model_shape,
        test_label_shape=label_shape,
        gpu_device='/gpu:0')

    # Training loop
    epochs = np.arange(epochs)
    for epoch in tqdm(epochs, desc='Epoch'):
        shuffle_idx = np.random.permutation(len(train_files))
        shuffle_files = train_files[shuffle_idx]
        start = time.time()
        for idx, f in enumerate(shuffle_files):
            # Grab seed from berson list
            lf = np.load(f)
            # row = lf['row']
            membranes = lf['vol']  # [None]
            label = lf['label'][None]

            # Train on synapse detection (0, 1, 2)
            feed_dict = {
                train_dict['train_images']: membranes,
                train_dict['train_labels']: label,
                train_dict['lr']: lr,
            }
            # TODO: Change to CCE
            ret = sess.run(train_dict, feed_dict=feed_dict)
            train_loss = ret['train_loss']
            train_f1 = ret['train_f1']
            train_precision = ret['train_precision']
            train_recall = ret['train_recall']
            if idx % 100 == 0:
                end = time.time()
                try:
                    print(
                        'Time elapsed: {} |'
                        'Synapse prediction loss: {} | '
                        'F1: {} | '
                        'Recall: {} |'
                        'Precision: {}'.format(
                            end,
                            train_loss,
                            train_f1,
                            train_recall,
                            train_precision))
                except:
                    print('Failed on step {}'.format(idx))
        print('Saving checkpoint to: {}'.format())
        saver.save(
            sess,
            ckpt_path,
            global_step=epoch)


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

