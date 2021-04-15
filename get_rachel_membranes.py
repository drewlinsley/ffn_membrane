import os
import time
import logging
import argparse
import itertools
import nibabel as nib
import numpy as np
from config import Config
from google.protobuf import text_format
from ffn.inference import inference
from ffn.inference import inference_pb2
from membrane.models import l3_fgru_constr as fgru
from utils.hybrid_utils import recursive_make_dir
from utils.hybrid_utils import pad_zeros
from utils.hybrid_utils import _bump_logit_map
from utils.hybrid_utils import rdirs
from copy import deepcopy
from tqdm import tqdm
import functools
import staintools


logger = logging.getLogger()
logger.setLevel(logging.INFO)
AUGS = ['uniform', 'pixel', 'rot270', 'rot90', 'rot180', 'flip_left', 'flip_right', 'depth_flip']  # ['pixel']  # 'depth_flip']  # , 'rot90', 'rot180', 'rot270']
AUGS = ['flip_left', 'flip_right', 'depth_flip']
AUGS = []
ROTS = []  # ['rot90', 'rot180']  # 'rot90', 'rot180', 'rot270']
TEST_TIME_AUGS = functools.reduce(
    lambda x, y: list(
        itertools.combinations(AUGS, y)) + x,
    list(range(len(AUGS) + 1)), [])[:-1]
# PAUGS = []
# for aug in TEST_TIME_AUGS:
#     t = np.array([1 if 'rot' in x else 0 for x in TEST_TIME_AUGS]).sum()
#     if t <= 1:
#         PAUGS += [aug]
# TEST_TIME_AUGS = PAUGS
PAUGS = deepcopy(TEST_TIME_AUGS)
for rot in ROTS:
    it_augs = []
    for idx in range(len(TEST_TIME_AUGS)):
        ita = list(TEST_TIME_AUGS[idx])
        if 'depth_flip' not in ita:
            it_augs += [[rot] + ita]
    PAUGS += it_augs
TEST_TIME_AUGS = [list(p) for p in PAUGS]


def get_membranes_nii(config, seed, path_extent, pull_from_db, return_membrane=False):
    if not pull_from_db:
        seed = seed
    else:
        seed = db.get_next_synapse_coordinate()
        if seed is None:
            raise RuntimeError('No more coordinantes to process!')

    coords, idxs = [], []
    empty = False
    vol = np.zeros((np.array(config.shape) * np.array(path_extent)), dtype=np.float32)  # shape * path_extent
    test_vol = np.zeros((np.array(config.shape) * np.array(path_extent)), dtype=np.float32)
    shape = config.shape
    membrane = True
    for z in range(path_extent[0]):
        for y in range(path_extent[1]):
            for x in range(path_extent[2]):
                for dr in config.mem_dirs:
                    # coord = [seed['x'] + x, seed['y'] + y, seed['z'] + z]
                    coord = [seed['x'] + x, seed['y'] + y, seed['z'] + z]
                    vp = config.path_str.replace("%s", "{}")
                    vp = vp.format(pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4), pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4))
                    ims = np.fromfile(vp, dtype=np.uint8).reshape(128, 128, 128)
                    test_vol[
                        z * shape[0]: z * shape[0] + shape[0],  # nopep8
                        y * shape[1]: y * shape[1] + shape[1],  # nopep8
                        x * shape[2]: x * shape[2] + shape[2]] = ims
                    if "%s" in dr:
                        dr = dr.replace("%s", "{}")
                        tp = dr.format(pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4), pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4))
                    else:
                        tp = os.path.join(
                            dr,
                            "x{}".format(pad_zeros(coord[0], 4)),
                            "y{}".format(pad_zeros(coord[1], 4)),
                            "z{}".format(pad_zeros(coord[2], 4)),
                            "110629_k0725_mag1_x{}_y{}_z{}.nii".format(
                                pad_zeros(coord[0], 4),
                                pad_zeros(coord[1], 4),
                                pad_zeros(coord[2], 4)))
                    if os.path.exists(tp):
                        zp = nib.load(tp)
                        h = zp.dataobj
                        v = h.get_unscaled()
                        zp.uncache()
                        del zp, h
                        vol[
                            z * shape[0]: z * shape[0] + shape[0],  # nopep8
                            y * shape[1]: y * shape[1] + shape[1],  # nopep8
                            x * shape[2]: x * shape[2] + shape[2]] = v
                        break
                    else:
                        membrane = None
    if membrane is not None:
        # vol = np.zeros((np.array(config.shape) * np.array(path_extent)), dtype=np.float32)  # shape * path_extent
        # membrane = build_vol(vol=vol, vols=vols, coords=idxs, shape=config.shape)
        membrane = vol
        membrane[np.isnan(membrane)] = 0
        assert membrane.max() > 1, 'Membrane is scaled to [0, 1]. Fix this!'
        if return_membrane:
            return membrane
    else:
        return False


def get_membranes(config, seed, pull_from_db, return_membrane=False):
    if not pull_from_db:
        seed = seed
    else:
        seed = db.get_next_synapse_coordinate()
        if seed is None:
            raise RuntimeError('No more coordinantes to process!')
    try:
        path = config.read_mem_str % (
            pad_zeros(seed['x'], 4),
            pad_zeros(seed['y'], 4),
            pad_zeros(seed['z'], 4),
            pad_zeros(seed['x'], 4),
            pad_zeros(seed['y'], 4),
            pad_zeros(seed['z'], 4))
        membrane = np.load('{}.npy'.format(path))
    except:
        path = config.write_mem_str % (
            pad_zeros(seed['x'], 4),
            pad_zeros(seed['y'], 4),
            pad_zeros(seed['z'], 4),
            pad_zeros(seed['x'], 4),
            pad_zeros(seed['y'], 4),
            pad_zeros(seed['z'], 4))
        membrane = np.load('{}.npy'.format(path))
    assert membrane.max() > 1  # , 'Membrane is scaled to [0, 1]. Fix this!'
    if return_membrane:
        return membrane
    # Check vol/membrane scale
    # vol = (vol / 255.).astype(np.float32)
    membrane[np.isnan(membrane)] = 0.
    vol = np.stack((vol, membrane), -1)[None] / 255.
    return vol, None


def augment(vo, augs):
    """Augment volume with augmentation au."""
    for au in augs:
        if au is 'rot90':
            vo = np.rot90(vo, 1, (2, 3))
        elif au is 'rot180':
            vo = np.rot90(vo, 2, (2, 3))
        elif au is 'rot270':
            vo = np.rot90(vo, 3, (2, 3))
        elif au is 'lr_flip':
            vo = vo[..., ::-1]
        elif au is 'ud_flip':
            vo = vo[..., ::-1, :]
        elif au is 'depth_flip':
            vo = vo[:, ::-1]
        elif au is 'noise':
            vo += np.random.rand(*vo.shape) * 1e-1
            vo = np.clip(vo, 0, 1)
    return vo


def undo_augment(vo, augs, debug_mem=None):
    """Augment volume with augmentation au."""
    for au in augs:
        if au is 'rot90':
            vo = np.rot90(vo, -1, (2, 3))
        elif au is 'rot180':
            vo = np.rot90(vo, -2, (2, 3))
        elif au is 'rot270':
            vo = np.rot90(vo, -3, (2, 3))
        elif au is 'lr_flip':
            vo = vo[..., ::-1, :]  # Note: 3-channel volumes
        elif au is 'ud_flip':
            vo = vo[..., ::-1, :, :]
        elif au is 'depth_flip':
            vo = vo[:, ::-1]
        elif au is 'noise':
            pass
    return vo


def get_segmentation(
        idx,
        data_path=None,
        move_threshold=None,  # 0.7,
        segment_threshold=None,  # 0.5,
        validate=False,
        seed=None,
        savetype='.nii',
        shift_z=None,
        shift_y=None,
        shift_x=None,
        x=None,
        y=None,
        z=None,
        membrane_type='probability',
        ffn_transpose=(0, 1, 2),
        prev_coordinate=None,
        membrane_only=False,
        segment_only=False,
        merge_segment_only=False,
        seg_vol=None,
        deltas='[15, 15, 3]',  # '[27, 27, 6]'
        seed_policy='PolicyMembrane',  # 'PolicyPeaks'
        membrane_slice=[64, 384, 384],  # 576
        membrane_overlap_factor=[0.5, 0.5, 0.5],  # [0.875, 2./3., 2./3.],
        debug_resize=False,
        debug_nii=False,
        path_extent=None,  # [1, 1, 1],
        rotate=False):
    """Apply the FFN routines using fGRUs."""
    # TEST_TIME_AUGS = None
    config = Config()
    assert move_threshold is not None
    assert segment_threshold is not None
    # config.membrane_ckpt = "/media/data_cifs/connectomics/checkpoints/ffn_mem_model_wong3d_0_wong3d_0_2021_03_10_16_00_23_126603/model_47000.ckpt-47000"
    # config.membrane_ckpt = "/media/data_cifs/connectomics/checkpoints/ffn_mem_model_wong3d_0_wong3d_0_2021_03_15_22_03_00_916465/model_28000.ckpt-28000"
    path_extent = np.asarray([int(x) for x in path_extent.split(",")])
    model_shape = (config.shape * path_extent)
    mpath = '/localscratch/middle_cube_membranes_for_ffn_training'
    membrane_slice = [60, 120*2, 120*2]
    config.shape = np.asarray([60, 120, 120])
    membrane_slice = None  # [60, 120*2, 120*2]
    config.shape = np.asarray([60, 280, 880])
    model_shape = config.shape
    if idx == 0:
        # 1. select a volume
        vol = np.load("/media/data_cifs/connectomics/datasets/filtered_wong_berson.npz")
        vol = vol["volume"]
        vol = vol.astype(np.float32) / 255.
        vol = vol[..., :280, :880]
        # vol = vol[..., :120*2, :120 * 7]
        _vol = vol.shape
        print(('seed: %s' % seed))
        print(('mpath: %s' % mpath))
        print(('volume size: (%s, %s, %s)' % (
            _vol[0],
            _vol[1],
            _vol[2])))

        # 2. Predict its membranes
        # volume size: (60, 290, 900)
        predict_membranes = True
        if os.path.exists("{}.npy".format(mpath)):
            predict_membranes = False
        if predict_membranes:
            membrane_model_shape = model_shape
            # model_shape = (config.shape * path_extent)
            if 1:
                membranes = fgru.main(
                    test=vol,
                    evaluate=True,
                    adabn=True,
                    gpu_device='/gpu:0',
                    test_input_shape=np.concatenate((
                        membrane_model_shape, [1])).tolist(),
                    test_label_shape=np.concatenate((
                        membrane_model_shape, [3])).tolist(),
                    checkpoint=config.membrane_ckpt)
                membranes = np.concatenate(membranes, 0).max(-1)  # mean

            vol = vol.transpose(ffn_transpose)  # ).astype(np.uint8)
            membranes = np.stack(
                (vol, membranes), axis=-1).astype(np.float32) * 255.
            np.save(mpath, membranes)
            print('Saved membrane volume to %s' % mpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--idx',
        dest='idx',
        type=int,
        default=0,
        help='Segmentation version.')
    parser.add_argument(
        '--seed',
        dest='seed',
        type=str,
        default='14,15,18',
        help='Center volume for segmentation.')
    parser.add_argument(
        '--seed_policy',
        dest='seed_policy',
        type=str,
        default='PolicyMembrane',
        help='Policy for finding FFN seeds.')
    parser.add_argument(
        '--path_extent',
        dest='path_extent',
        type=str,
        default='1,2,6',
        help='Provide extent of segmentation in 128^3 volumes.')
    parser.add_argument(
        '--move_threshold',
        dest='move_threshold',
        type=float,
        default=0.8,  # 0.8
        help='Movement threshold. Higher is more likely to move.')
    parser.add_argument(
        '--segment_threshold',
        dest='segment_threshold',
        type=float,
        default=0.6,  # 0.6
        help='Segment threshold..')
    parser.add_argument(
        '--membrane_slice',
        dest='membrane_slice',
        type=str,
        default=None,
        help='Membrane chunking along z axis.')
    parser.add_argument(
        '--validate',
        dest='validate',
        action='store_true',
        help='Force berson validation dataset.')
    parser.add_argument(
        '--rotate',
        dest='rotate',
        action='store_true',
        help='Rotate the input data.')
    parser.add_argument(
        '--membrane_only',
        dest='membrane_only',
        action='store_true',
        help='Only process membranes.')
    parser.add_argument(
        '--segment_only',
        dest='segment_only',
        action='store_true',
        help='Only process segments.')
    parser.add_argument(
        '--merge_segment_only',
        dest='merge_segment_only',
        action='store_true',
        help='Only process merge segments.')
    args = parser.parse_args()
    start = time.time()
    get_segmentation(**vars(args))
    end = time.time()
    print(('Segmentation took {}'.format(end - start)))
