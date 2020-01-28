import os
import time
import logging
import argparse
import nibabel as nib
import numpy as np
from config import Config
from membrane.models import l3_fgru_constr as fgru
from utils.hybrid_utils import pad_zeros
from tqdm import tqdm
import pandas as pd
from lxml import etree
# from scipy.spatial import distance
from skimage.filters import gaussian


logger = logging.getLogger()
logger.setLevel(logging.INFO)
SEL_THINGS = {
    'ribbon': 1,
    # 'extra_ribbons': 4,
    'amacrine': 4,
    'extra out of volume ac synapses': 4,
    'ectopic synapses onto m1 7056': 4,
    'ectopics onto m1 x1': 4,
    # 'extra_amacrines': 6
}
FIX_RADIUS = [24, 24, 24]  # X/Y/Z indicator


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


def pull_volume(seed, path_extent, config, membrane=False):
    """Grab a volume at seed location."""
    if membrane:
        path_str = config.nii_mem_str
    else:
        path_str = config.path_str
    vol = np.zeros((np.array(config.shape) * path_extent))
    for z in range(path_extent[0]):
        for y in range(path_extent[1]):
            for x in range(path_extent[2]):
                path = path_str % (
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
                del v
    vol = vol.astype(np.float32) / 255.
    _vol = vol.shape
    # Note that vol is transposed to zyx
    return vol, _vol


def pull_nii_volume(seed, path_extent, config, membrane=False):
    """Grab a volume at seed location."""
    if membrane:
        path_str = config.nii_mem_str
    else:
        path_str = config.path_str
    vol = np.zeros((np.array(config.shape) * path_extent))
    for z in range(path_extent[0]):
        for y in range(path_extent[1]):
            for x in range(path_extent[2]):
                path = path_str % (
                    pad_zeros(seed[0] + x, 4),
                    pad_zeros(seed[1] + y, 4),
                    pad_zeros(seed[2] + z, 4),
                    pad_zeros(seed[0] + x, 4),
                    pad_zeros(seed[1] + y, 4),
                    pad_zeros(seed[2] + z, 4))
                fl = nib.load(path)
                v = fl.get_fdata()
                vol[
                    z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                    y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                    x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v  # nopep8
                del v
    vol = vol.astype(np.float32) / 255.
    _vol = vol.shape
    # Note that vol is transposed to zyx
    return vol, _vol


def create_indicator(
        size,
        anchor_row,
        synapse_info,
        pc_label=None,
        debug=False,
        smooth=False,
        precompute=False,
        dtype=np.float32):
    """Create label volume with 0/1(ribbon) and 0/1(amacrine) channels.
    Will be either top-left or center-based

    anchor_row is the anchor synapse
    synapse_info is extra synapses"""
    if debug:
        label = np.zeros((np.concatenate((size, [2]))), dtype=dtype)
        label[:, size[0] // 2, size[1] // 2, size[2] // 2] = 1
        return label

    # Create anchor synapse
    st = np.minimum(anchor_row['thing'], 2) - 1
    center = np.array(
        [anchor_row['offx'], anchor_row['offy'], anchor_row['offz']])
    if pc_label is None:
        inds = np.indices(size).T
        # d = np.sum(np.abs(inds - center), axis=len(center)).transpose((2, 1, 0))
        d = np.linalg.norm(inds - center, axis=len(center))  # .transpose((2, 1, 0))
        bump = d <= anchor_row['rx']  # Only using one rad
        if smooth:
            bump = gaussian(
                bump.transpose(1, 2, 0),
                sigma=5.,
                multichannel=False,
                preserve_range=True,
                truncate=100).transpose(2, 0, 1)
    else:
        bump = np.copy(pc_label[..., st])
    mask = np.zeros((np.concatenate((size, [2]))), dtype=dtype)
    mask[..., st] += bump.astype(dtype)
    if precompute:
        return mask
    if len(synapse_info):
        anchor_fx = anchor_row['fx']
        anchor_fy = anchor_row['fy']
        anchor_fz = anchor_row['fz']
        for _, it_row in synapse_info.iterrows():
            st = np.minimum(it_row['thing'], 1) - 1
            # Compute offx,offy,offz relative to anchor
            delta_x = it_row['fx'] - anchor_fx
            delta_y = it_row['fy'] - anchor_fy
            delta_z = it_row['fz'] - anchor_fz
            offx = it_row['offx'] + delta_x * anchor_fx
            offy = it_row['offy'] + delta_y * anchor_fy
            offz = it_row['offz'] + delta_z * anchor_fz
            center = np.array([offx, offy, offz])
            # d = np.sum((inds - center) ** 2, axis=len(center)).transpose((2, 1, 0))
            d = np.linalg.norm(inds - center, axis=len(center)).transpose((2, 1, 0))
            bump = d < anchor_row['rx']  # Only using one rad
            if smooth:
                bump = gaussian(
                    bump.transpose(1, 2, 0),
                    sigma=5.,
                    multichannel=False,
                    preserve_range=True,
                    truncate=100).transpose(2, 0, 1)
            mask[..., st] += bump.astype(dtype)
    label = np.minimum(1, mask)  # clamp to 0, 1
    return label


def get_label(comment):
    """Get label from xml."""
    for k, v in SEL_THINGS.iteritems():
        if k in comment:
            return v
    raise NotImplementedError(comment)


def get_seeds(
        seed_file,
        size,
        path_extent,
        seed_path_extent,
        cube_size,
        fix_radius=FIX_RADIUS,
        sel_things=SEL_THINGS,
        center=True):
    """Extract seed coordinates from an xml file."""
    with open(seed_file, 'r') as f:
        seed_file = f.read()
    root = etree.fromstring(seed_file)
    df = []
    # center_offset = (path_extent * cube_size) // 2
    center_offset = (path_extent // 2) * cube_size  # quantized
    things = root.getchildren()
    for thing in things:
        nodes = thing.getchildren()[0]
        comment = thing.get('comment')
        if comment is None:
            continue
        v = get_label(comment.lower())
        for node in nodes:
            radius = fix_radius
            _id = int(node.get('id'))
            x = int(node.get('x'))
            y = int(node.get('y'))
            z = int(node.get('z'))
            # adjx = np.round(x / float(cube_size)).astype(int) * cube_size
            # adjy = np.round(y / float(cube_size)).astype(int) * cube_size
            # adjz = np.round(z / float(cube_size)).astype(int) * cube_size
            adjx = (x // cube_size) * cube_size
            adjy = (y // cube_size) * cube_size
            adjz = (z // cube_size) * cube_size
            if center:
                # Center the seed using size
                adjx = x - center_offset[0]
                adjy = y - center_offset[1]
                adjz = z - center_offset[2]
            offx = x - cube_size * (adjx / cube_size)
            offy = y - cube_size * (adjy / cube_size)
            offz = z - cube_size * (adjz / cube_size)
            # fx = np.round(adjx / float(cube_size)).astype(int)
            # fy = np.round(adjy / float(cube_size)).astype(int)
            # fz = np.round(adjz / float(cube_size)).astype(int)
            correctx = x - adjx
            correcty = y - adjy
            correctz = z - adjz
            fx = adjx // cube_size
            fy = adjy // cube_size
            fz = adjz // cube_size
            df += [[_id, x, y, z, adjx, adjy, adjz, fx, fy, fz, offx, offy, offz, correctx, correcty, correctz, radius[0], radius[1], radius[2], v]]  # noqa
    df = pd.DataFrame(
        np.asarray(df).astype(float).round().astype(int),
        columns=['_id', 'x', 'y', 'z', 'adjx', 'adjy', 'adjz', 'fx', 'fy', 'fz', 'offx', 'offy', 'offz', 'correctx', 'correcty', 'correctz', 'rx', 'ry', 'rz', 'thing'])  # noqa

    # Greedily merge within an extent-sized cube
    row_coords = []
    for row_id, row in df.iterrows():
        row_coords += [[row['fx'], row['fy'], row['fz']]]
    # dm = distance.squareform(distance.pdist(row_coords, metric='cityblock'))
    row_coords = np.array(row_coords)
    seed_path_extent = seed_path_extent // 2  # radius
    dm = np.zeros((len(row_coords), len(row_coords)), dtype=np.float32)
    for ri, r in enumerate(row_coords):
        for ci, c in enumerate(row_coords):
            d = c - r  # Signed diff  -- r < c
            test = np.logical_and(d > 0, d < seed_path_extent[0])
            dm[ri, ci] = np.all(test)
    exclude = []
    merge_rows = np.zeros_like(dm)
    for row_id, row in enumerate(dm):
        if row_id not in exclude:
            add_ids = np.where(row)[0]
            new_zeros = np.zeros_like(row)
            new_zeros[add_ids] = True
            exclude += add_ids.tolist()
        merge_rows[row_id] = new_zeros
    return df, merge_rows


def crop_vol(vol, crop_shape, correctx, correcty, correctz, row):
    """Crop volume to specified size, centering the synapse."""
    half_shape = np.array(crop_shape) // 2
    correctzyx = np.array([correctz, correcty, correctx])
    upper_corner = correctzyx - half_shape
    lower_corner = correctzyx + half_shape
    assert np.all(upper_corner > 0), 'Messed up crop_vol.'
    zslice = slice(upper_corner[0], lower_corner[0])
    yslice = slice(upper_corner[1], lower_corner[1])
    xslice = slice(upper_corner[2], lower_corner[2])
    row['offz'] -= upper_corner[0]
    row['offy'] -= upper_corner[1]
    row['offx'] -= upper_corner[2]
    return vol[zslice, yslice, xslice], tuple(crop_shape), row


def train(
        ffn_transpose=(0, 1, 2),
        path_extent=[3, 3, 3],
        seed_path_extent=[1, 1, 1],  # For merging labels
        cube_size=128,
        epochs=100,
        lr=1e-3,
        crop_shape=[80, 160, 160],
        seed_file='synapses/annotation.xml',
        seed_npz='synapses/annotation.xml.npz',
        rotate=False):
    """Apply the FFN routines using fGRUs."""
    config = Config()
    if isinstance(path_extent, str):
        path_extent = np.array([int(s) for s in path_extent.split(',')])
    if isinstance(crop_shape, str):
        crop_shape = np.array([int(s) for s in crop_shape.split(',')])
    seed_path_extent = np.array(seed_path_extent)
    if crop_shape is None:
        model_shape = path_extent * cube_size
    else:
        model_shape = crop_shape
    if not os.path.exists(seed_npz):
        print('Computing seeds')
        seeds, merges = get_seeds(
            seed_file,
            path_extent=path_extent,
            seed_path_extent=seed_path_extent,
            cube_size=cube_size,
            size=model_shape)  # Import seeds from berson's list
        np.savez(seed_file, seeds=seeds, merges=merges)
    else:
        print('Loading seeds')
        sf = np.load(seed_npz)
        seeds, merges = sf['seeds'], sf['merges']
        seeds = pd.DataFrame(
            seeds,
            columns=['_id', 'x', 'y', 'z', 'adjx', 'adjy', 'adjz', 'fx', 'fy', 'fz', 'offx', 'offy', 'offz', 'correctx', 'correcty', 'correctz', 'rx', 'ry', 'rz', 'thing'])  # noq

    # Get membrane stuff
    membrane_test_dict, membrane_sess = fgru.main(
        test=None,
        evaluate=True,
        adabn=True,
        gpu_device='/gpu:0',
        return_sess=False,
        force_return_model=True,
        test_input_shape=np.concatenate((
            model_shape, [1])).tolist(),
        test_label_shape=np.concatenate((
            model_shape, [3])).tolist(),
        checkpoint=config.membrane_ckpt)

    # Loop over all seeds
    fails = []
    # seeds = seeds[seeds.thing == 4]
    # idx = np.where(seeds._id == 27948)[0]
    # seeds = seeds.iloc[idx]
    # merges = merges[idx]
    for (idx, row), merge in tqdm(
            zip(seeds.iterrows(), merges),
            desc='Saving seeds'):
        # Grab seed from berson list
        seed = np.array([row['fx'], row['fy'], row['fz']])
        vol, _vol = pull_volume(
            seed=seed,
            path_extent=path_extent,
            config=config)

        # Crop if requested
        if crop_shape is not None:
            og_row = np.copy(row)
            vol, _vol, row = crop_vol(
                vol,
                row=row,
                crop_shape=crop_shape,
                correctx=row['offx'],
                correcty=row['offy'],
                correctz=row['offz'])

        synapse_info = seeds.iloc[np.where(merge)[0]]
        label = create_indicator(
            # pc_label=it_pc_label,
            size=_vol,
            anchor_row=row,
            synapse_info=synapse_info)
        try:
            membranes, _ = pull_nii_volume(
                seed=seed,
                path_extent=path_extent,
                config=config,
                membrane=True)
            membranes, _, _ = crop_vol(
                membranes,
                row=og_row,
                crop_shape=crop_shape,
                correctx=og_row['offx'],
                correcty=og_row['offy'],
                correctz=og_row['offz'])
        except Exception:
            feed_dict = {
                membrane_test_dict['test_images']: vol[None, ..., None]
            }
            it_test_dict = membrane_sess.run(
                membrane_test_dict,
                feed_dict=feed_dict)
            membranes = it_test_dict['test_logits']
            membranes = membranes.max(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_extent',
        dest='path_extent',
        type=str,
        default='3,3,3',
        help='Provide extent of segmentation in 128^3 volumes.')
    parser.add_argument(
        '--crop_shape',
        dest='crop_shape',
        type=str,
        default='160,160,160',
        help='Provide extent of segmentation in 128^3 volumes.')
    args = parser.parse_args()
    start = time.time()
    train(**vars(args))
    end = time.time()
    print('Training took {}'.format(end - start))
