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


logger = logging.getLogger()
logger.setLevel(logging.INFO)
SEL_THINGS = {
    'ribbons': 1,
    # 'extra_ribbons': 4,
    'amacrines': 4,
    # 'extra_amacrines': 6
}
FIX_RADIUS = [64, 64, 16]  # X/Y/Z indicator


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


def pull_volume(seed, path_extent, config, center=False):
    """Grab a volume at seed location."""
    if np.all(path_extent == 1):
        raise RuntimeError('Set path_extent to 2,2,2')
    else:
        vol = np.zeros((np.array(config.shape) * path_extent))
        for z in range(path_extent[0]):
            for y in range(path_extent[1]):
                for x in range(path_extent[2]):
                    path = config.path_str % (
                        pad_zeros(seed[0] + x, 4),
                        pad_zeros(seed[1] + y, 4),
                        pad_zeros(seed[2] + z, 4),
                        pad_zeros(seed[0] + x, 4),
                        pad_zeros(seed[1] + y, 4),
                        pad_zeros(seed[2] + z, 4))
                    v = np.fromfile(
                        path, dtype='uint8').reshape(config.shape)
                    vol[
                        z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                        y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                        x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v  # nopep8
    vol = vol.astype(np.float32) / 255.
    _vol = vol.shape
    return vol, _vol


def create_indicator(size, synapse_info, batch_size=1, debug=True, dtype=np.float32):
    """Create label volume with 0/1(ribbon) and 0/1(amacrine) channels."""
    label = np.zeros((np.concatenate(([batch_size], size, [2]))), dtype=dtype)
    if debug:
        label[:, size[0] // 2, size[1] // 2, size[2] // 2] = 1
    else:
        for _, row in synapse_info.iterrows():
            label[sd] = st
    return label


def get_seeds(seed_file, size, fix_radius=FIX_RADIUS, sel_things=SEL_THINGS, center=True):
    """Extract seed coordinates from an xml file."""
    with open(seed_file, 'r') as f:
        seed_file = f.read()
    root = etree.fromstring(seed_file)
    df = []
    for k, v in sel_things.iteritems():
        nodes = root.getchildren()[v].getchildren()[0]
        for node in nodes:
            radius = fix_radius
            x = int(node.get('x'))
            y = int(node.get('y'))
            z = int(node.get('z'))
            if center:
                # Center the seed using size
                import ipdb;ipdb
                x, y, z = x - size[0], y - size[1], z - size[2]

            fx = x // 128
            fy = y // 128
            fz = z // 128
            df += [[x, y, z, fx, fy, fz, radius[0], radius[1], radius[2], v]]
    df = pd.DataFrame(
        np.asarray(df).astype(float).round().astype(int),
        columns=['x', 'y', 'z', 'fx', 'fy', 'fz', 'rx', 'ry', 'rz', 'thing'])

    # Greedily merge within an extent-sized cube
    row_coords = []
    for row_id, row in df.iterrows():
        row_coords += [[row['x'], row['y'], row['z']]]
    dm = distance.squareform(distance.pdist(row_coords, metric='cityblock'))
    exclude = []
    merge_mask = np.logical_and(dm < 384, dm > 0)
    merge_rows = np.zeros_like(dm)
    for row_id, row in enumerate(merge_mask):
        if row_id not in exclude:
            add_ids = np.where(row)[0]
            new_zeros = np.zeros_like(row)
            new_zeros[add_ids] = True
            exclude += add_ids.tolist()
        merge_rows[row_id] = new_zeros
    return df, merge_rows


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
    model_shape = path_extent * cube_size
    seeds, merges = get_seeds(
        seed_file,
        size=model_shape)  # Import seeds from berson's list
    seeds = seeds[seeds['x'] == 1304]
    epochs = np.arange(epochs)

    # Get new model training variables
    sess, saver, train_dict, test_dict = unet.main(
        train_input_shape=np.concatenate((
            model_shape, [2])).tolist(),
        train_label_shape=np.concatenate((
            model_shape, [2])).tolist(),
        test_input_shape=np.concatenate((
            model_shape, [2])).tolist(),
        test_label_shape=np.concatenate((
            model_shape, [2])).tolist(),
        gpu_device='/gpu:0')

    # Training loop
    for epoch in tqdm(epochs, desc='Epoch'):
        shuffle_idx = np.arange(len(seeds))  # np.random.permutation(len(seeds))
        shuffle_seeds = seeds.iloc[shuffle_idx]
        shuffle_merges = merges[shuffle_idx]
        for (idx, row), merge in zip(shuffle_seeds.iterrows(), shuffle_merges):
            # Grab seed from berson list
            seed = np.array([row['fx'], row['fy'], row['fz']])
            synapse_type = row['thing']
            vol, _vol = pull_volume(
                seed=seed,
                path_extent=path_extent,
                config=config)

            # Create indicator volume
            # TODO: cityblock not working and verify w/ berson synapses are correct
            synapse_info = shuffle_seeds.iloc[np.where(merge)[0]]
            label = create_indicator(
                size=_vol,
                synapse_info=synapse_info)

            # Train on synapse detection (0, 1, 2)
            feed_dict = {
                train_dict['train_images']: membranes,
                train_dict['train_labels']: label,
                train_dict['lr']: lr,
            }
            ret = sess.run(train_dict['train_op'], feed_dict=feed_dict)
            train_loss = ret['train_loss']
            train_f1 = ret['train_f1']
            train_precision = ret['train_precision']
            train_recall = ret['train_recall']
            # train_logits = ret['train_logits']
            # probs = ret['logits']
            if idx % 100 == 0:
                print(
                    'Synapse prediction loss: {} | '
                    'F1: {} | '
                    'Recall: {} |'
                    'Precision: {}'.format(
                        train_loss,
                        train_f1,
                        train_recall,
                        train_precision))
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

