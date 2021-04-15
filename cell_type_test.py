import os
import time
import logging
import argparse
import itertools
import nibabel as nib
import numpy as np
import tensorflow as tf
from config import Config
# from google.protobuf import text_format
# from ffn.inference import inference
# from ffn.inference import inference_pb2
from membrane.models import l3_fgru_constr as fgru
from utils.hybrid_utils import recursive_make_dir
from utils.hybrid_utils import pad_zeros
from utils.hybrid_utils import _bump_logit_map
from utils.hybrid_utils import rdirs
from copy import deepcopy
from tqdm import tqdm
from skimage import morphology
from glob2 import glob
import functools


BU_DIRS = [
    "/cifs/data/tserre/CLPS_Serre_Lab/connectomics/mag1_membranes_npy",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics/mag1_membranes",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics_data_v0/mag1_membranes",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data/mag1_membranes",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch/mag1_membranes",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_v0/mag1_membranes",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v1/mag1_membranes",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v2/connectomics_data/mag1_membranes",
]

logger = logging.getLogger()
logger.setLevel(logging.INFO)
AUGS = ['lr_flip', 'ud_flip', 'depth_flip']  # , 'rot90', 'rot180', 'rot270']
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


def reconstruct(vol, shape, z_splits, y_splits, x_splits, membrane_slice, adj_membrane_slice, normalize=False):
    rmembranes = np.zeros(shape, dtype=np.float32)
    print("Recon shape: {}".format(rmembranes.shape))
    count = 0
    vols = []
    if normalize:
        normalization = np.zeros(shape, dtype=vol.dtype)
    for z_idx in z_splits:
        for y_idx in y_splits:
            for x_idx in x_splits:
                if z_idx == 0:
                    zu = z_idx
                    zo = z_idx + membrane_slice[0]
                else:
                    zu = z_idx - adj_membrane_slice[0]
                    zo = zu + membrane_slice[0]
                if y_idx == 0:
                    yu = y_idx
                    yo = y_idx + membrane_slice[1]
                else:
                    yu = y_idx - adj_membrane_slice[1]
                    yo = yu + membrane_slice[1]
                if x_idx == 0:
                    xu = x_idx
                    xo = x_idx + membrane_slice[2]
                else:
                    xu = x_idx - adj_membrane_slice[2]
                    xo = xu + membrane_slice[2]
                rmembranes[
                    zu: zo,
                    yu: yo,
                    xu: xo] += vol[count].squeeze()  #  * bump_map
                count += 1
    return rmembranes


def load_data(seed, config, path_extent):
    """Cycle through the npz paths and fall back on niis."""
    for d in BU_DIRS:
        mpath = os.path.join(
            d, "x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.raw.npy".format(
                pad_zeros(seed[0], 4),
                pad_zeros(seed[1], 4),
                pad_zeros(seed[2], 4),
                pad_zeros(seed[0], 4),
                pad_zeros(seed[1], 4),
                pad_zeros(seed[2], 4)))
        if os.path.exists(mpath):
            return np.load(mpath), False
    return load_niis(seed, config, path_extent), True


def load_niis(seed, config, path_extent):
    """Load nii files."""
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
                vol[                    z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                    y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                    x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v  # nopep8
    return vol


def sigmoid(x):
    return 1./(1. + np.exp(-x))

# @njit(parallel=True, fastmath=True, nogil=True)
def build_vol(vol, vols, coords, shape):
    """Insert d into vol."""
    for idx in range(len(vols)):
        x, y, z = coords[idx]
        vol[
            x * shape[0]: x * shape[0] + shape[0],  # nopep8
            y * shape[1]: y * shape[1] + shape[1],  # nopep8
            z * shape[2]: z * shape[2] + shape[2]] = vols[idx]
    return vol


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
    assert membrane.max() > 1, 'Membrane is scaled to [0, 1]. Fix this!'
    if return_membrane:
        return membrane
    # Check vol/membrane scale
    # vol = (vol / 255.).astype(np.float32)
    membrane[np.isnan(membrane)] = 0.
    vol = np.stack((vol, membrane), -1)[None] / 255.
    return vol, None


def get_membranes_nii(config, seed, path_extent, pull_from_db, return_membrane=False, force_fail=False):
    if force_fail:
        return False, False
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
    membrane_check = []
    try:
        for z in range(path_extent[0]):
            for y in range(path_extent[1]):
                for x in range(path_extent[2]):
                    coord = [seed['x'] + x, seed['y'] + y, seed['z'] + z]
                    vp = config.path_str.replace("%s", "{}")
                    vp = vp.format(pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4), pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4))
                    ims = np.fromfile(vp, dtype=np.uint8).reshape(128, 128, 128)
                    test_vol[
                        z * shape[0]: z * shape[0] + shape[0],  # nopep8
                        y * shape[1]: y * shape[1] + shape[1],  # nopep8
                        x * shape[2]: x * shape[2] + shape[2]] = ims
                    dr = config.write_mem_per_vol_str.replace("%s", "{}")
                    tp = dr.format(pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4), pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4))
                    tp = "{}.npy".format(tp)
                    if os.path.exists(tp):
                        v = np.load(tp)
                        print(v.shape)
                        vol[
                            z * shape[0]: z * shape[0] + shape[0],  # nopep8
                            y * shape[1]: y * shape[1] + shape[1],  # nopep8
                            x * shape[2]: x * shape[2] + shape[2]] = v
                        membrane_check.append(1)
                    else:
                        membrane_check.append(0)
            if np.all(membrane_check):
                membrane = vol
                membrane[np.isnan(membrane)] = 0
                assert membrane.max() > 1, 'Membrane is scaled to [0, 1]. Fix this!'
                if return_membrane:
                    return membrane, True
        else:
            return False, False
    except:
            return False, False


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
        segment_threshold=0.6,
        validate=False,
        seed=None,
        savetype='.nii',
        shift_z=None,
        shift_y=None,
        shift_x=None,
        x=None,
        y=None,
        z=None,
        # cell_ckpt_path='/cifs/data/tserre_lrs/projects/prj_connectomics/ffn_membrane_v2/celltype_ckpts',
        # cell_ckpt_path='/cifs/data/tserre_lrs/projects/prj_connectomics/ffn_membrane_v2/celltype_ckpts/good.ckpt',
        cell_ckpt_path='celltype_models/',
        membrane_type='probability',
        ffn_transpose=(0, 1, 2),
        prev_coordinate=None,
        membrane_only=False,
        segment_only=False,
        merge_segment_only=False,
        seg_vol=None,
        deltas='[15, 15, 3]',  # '[27, 27, 6]'
        membrane_slice=[128, 384, 384],  # 576
        # membrane_overlap_factor=[0.5, 0.5, 0.5],  # [0.875, 2./3., 2./3.],
        membrane_overlap_factor=[1., 1., 1.],  # [0.875, 2./3., 2./3.],
        debug_resize=False,
        debug_nii=False,
        path_extent=None):
    """Apply the FFN routines using fGRUs."""
    config = Config()
    if x is None and y is None and z is None:
        seed = np.array([int(s) for s in seed.split(',')])
    else:
        seed = np.array([x, y, z])
    if isinstance(path_extent, str):
        path_extent = np.array([int(s) for s in path_extent.split(',')])
    if isinstance(membrane_slice, str):
        membrane_slice = [int(s) for s in membrane_slice.split(',')]
    if len(membrane_slice) != 3:
        raise RuntimeError('membrane_slice needs to be a len(3) list.')
    assert segment_threshold is not None
    rdirs(seed, config.write_mem_str)
    model_shape = (config.shape * path_extent)
    mpath = config.write_mem_str % (
        pad_zeros(seed[0], 4),
        pad_zeros(seed[1], 4),
        pad_zeros(seed[2], 4),
        pad_zeros(seed[0], 4),
        pad_zeros(seed[1], 4),
        pad_zeros(seed[2], 4))
    rmpath = config.read_mem_str % (
        pad_zeros(seed[0], 4),
        pad_zeros(seed[1], 4),
        pad_zeros(seed[2], 4),
        pad_zeros(seed[0], 4),
        pad_zeros(seed[1], 4),
        pad_zeros(seed[2], 4))

    if idx == 0:
        # 1. select a volume
        vol, predict_membranes = load_data(seed, config, path_extent)
        vol = vol.astype(np.float32) / 255.
        _vol = vol.shape
        print('seed: %s' % seed)
        print('mpath: %s' % mpath)
        print('volume size: (%s, %s, %s)' % (
            _vol[0],
            _vol[1],
            _vol[2]))

        # 2. Predict its membranes
        TEST_TIME_AUGS = None
        if predict_membranes:
            membrane_model_shape = model_shape
            model_shape = (config.shape * path_extent)
            if membrane_slice is not None:
                assert isinstance(
                    membrane_slice, list), 'Make membrane_slice a list.'
                # Split up membrane along z-axis into k-voxel chunks
                # Include an overlap so that you have 1 extra slice per dim
                adj_membrane_slice = (np.array(
                    membrane_slice) * membrane_overlap_factor).astype(int)
                z_splits = np.arange(
                    adj_membrane_slice[0],
                    model_shape[0],
                    adj_membrane_slice[0])
                y_splits = np.arange(
                    adj_membrane_slice[1],
                    model_shape[1],
                    adj_membrane_slice[1])
                x_splits = np.arange(
                    adj_membrane_slice[2],
                    model_shape[2],
                    adj_membrane_slice[2])
                z_splits = np.concatenate([[0], z_splits])
                y_splits = np.concatenate([[0], y_splits])
                x_splits = np.concatenate([[0], x_splits])
                adj_membrane_slice = membrane_slice - adj_membrane_slice
                vols = []
                for z_idx in z_splits:
                    for y_idx in y_splits:
                        for x_idx in x_splits:
                            if z_idx == 0:
                                zu = z_idx
                                zo = z_idx + membrane_slice[0]
                            else:
                                zu = z_idx - adj_membrane_slice[0]
                                zo = zu + membrane_slice[0]
                            if y_idx == 0:
                                yu = y_idx
                                yo = y_idx + membrane_slice[1]
                            else:
                                yu = y_idx - adj_membrane_slice[1]
                                yo = yu + membrane_slice[1]
                            if x_idx == 0:
                                xu = x_idx
                                xo = x_idx + membrane_slice[2]
                            else:
                                xu = x_idx - adj_membrane_slice[2]
                                xo = xu + membrane_slice[2]
                            vols += [vol[
                                zu.astype(int): zo.astype(int),
                                yu.astype(int): yo.astype(int),
                                xu.astype(int): xo.astype(int)]]
                try:
                    vol = np.stack(vols)
                except Exception:
                    import pdb;pdb.set_trace()
                    print(
                        'Mismatch in volume_size/membrane slicing {}'.format(
                            [vs.shape for vs in vols]))
                    os._exit(1)
                rec_vol = reconstruct(vols, model_shape, z_splits, y_splits, x_splits, membrane_slice, adj_membrane_slice, normalize=False)
                del vols  # Garbage collect
                membrane_model_shape = membrane_slice
            print(membrane_model_shape)
            print(vol.shape)
            seed_dict = {
                xc: se for se, xc in zip(seed, ['x', 'y', 'z'])}
            if TEST_TIME_AUGS is not None:
                membranes, sess, test_dict = fgru.main(
                    test=vol,
                    evaluate=True,
                    adabn=True,
                    gpu_device='/gpu:0',
                    return_sess=True,
                    test_input_shape=np.concatenate((
                        membrane_model_shape, [1])).tolist(),
                    test_label_shape=np.concatenate((
                        membrane_model_shape, [3])).tolist(),
                    checkpoint=config.membrane_ckpt)
                for it_aug in TEST_TIME_AUGS:
                    aug_vol = augment(vo=vols, augs=it_aug)
                    for mi, td in tqdm(
                            enumerate(aug_vol),
                            total=len(aug_vol),
                            desc='Processing membranes {}'.format(it_aug)):
                        td = td[None]
                        feed_dict = {
                            test_dict['test_images']: td[..., None],
                        }
                        it_test_dict = sess.run(
                            test_dict,
                            feed_dict=feed_dict)
                        it_membranes = it_test_dict['test_logits']
                        membranes[mi] += undo_augment(
                            it_membranes, it_aug[::-1], membranes[mi])
                denom = np.array(len(TEST_TIME_AUGS) + 1.).astype(vol.dtype)
                membranes = ((np.stack(membranes).mean(1) + 1e-8) / denom).max(-1)  # noqa
                del aug_vol
            else:
                membrane_model_shape = membrane_slice
                membranes, test_dict, sess = fgru.main(
                    test=vol,
                    evaluate=True,
                    adabn=True,
                    return_sess=True,
                    gpu_device='/gpu:0',
                    test_input_shape=np.concatenate((
                        membrane_model_shape, [1])).tolist(),
                    test_label_shape=np.concatenate((
                        membrane_model_shape, [3])).tolist(),
                    checkpoint=config.membrane_ckpt)
                membranes = np.concatenate(membranes, 0)  # mean
        else:
            membrane_model_shape = membrane_slice
            test_dict, sess = fgru.main(
                test=vol[..., 0][..., None],
                evaluate=True,
                adabn=True,
                return_sess=True,
                force_return_model=True,
                gpu_device='/gpu:0',
                test_input_shape=np.concatenate((
                    membrane_model_shape, [1])).tolist(),
                test_label_shape=np.concatenate((
                    membrane_model_shape, [3])).tolist(),
                checkpoint=config.membrane_ckpt)
            membranes = vol[..., 1][..., None]
            vol = vol[..., 0][..., None]

        if len(membranes.shape) == 5:
            membranes = membranes.max(-1)
        model_shape = [128, vol.shape[2], vol.shape[3], 2]  # 1
        label_shape = [128, vol.shape[2], vol.shape[3], 1]  # 1
        from membrane.models import l3_fgru_constr_adabn_synapse as unet
        test_dict, sess = unet.main_cell_type(
            sess=sess,
            train_input_shape=[z for z in model_shape],
            train_label_shape=[z for z in label_shape],
            test_input_shape=[z for z in model_shape],
            test_label_shape=[z for z in label_shape],
            checkpoint=tf.train.latest_checkpoint(cell_ckpt_path),  # tf.train.latest_checkpoint(os.path.sep.join(ckpt_path.split(os.path.sep)[:-1])),
            return_restore_saver=False,
            force_return_model=True,
            evaluate_cell_type=True,
            gpu_device='/gpu:0')
        cell_preds = []
        # vol and membrane should be [0, 255]
        for i in range(len(vol)): 
            feed_dict = {
               test_dict['test_images']: np.stack((vol[i][None], membranes[i][None]), -1) * 255.,
            }
            out_dict = sess.run(test_dict, feed_dict=feed_dict)
            cell_preds.append(out_dict['test_logits'])

        if 1:  # membrane_slice is not None and not loaded_membranes:
            rmembranes = np.zeros(_vol, dtype=np.float32)
            count = 0
            vols = []
            # from matplotlib import pyplot as plt
            # rec_vol = reconstruct(vols, model_shape, z_splits, y_splits, x_splits, membrane_slice, adj_membrane_slice, normalize=False)
            rmembranes = reconstruct(cell_preds, rec_vol.shape, z_splits, y_splits, x_splits, membrane_slice, adj_membrane_slice, normalize=False)
            # rec_mems = reconstruct(membranes, rec_vol.shape, z_splits, y_splits, x_splits, membrane_slice, adj_membrane_slice, normalize=False)
            # plt.subplot(131);plt.imshow(rec_vol[256], cmap="Greys_r");plt.subplot(132);plt.imshow(rec_mems[256]);plt.subplot(133);plt.imshow(rmembranes[256]);plt.show()
            rmembranes = morphology.remove_small_objects(rmembranes > segment_threshold, min_size=4096).astype(np.float32)
            # ims=120;plt.imshow(np.stack([original_vol[ims], original_vol[ims], original_vol[ims], (rmembranes[ims] / 2.) + 0.5], -1));plt.show()
            del normalization, bump_map
            for z in range(path_extent[0]):
                for y in range(path_extent[1]):
                    for x in range(path_extent[2]):
                        path = config.write_muller_per_vol_str % (
                            pad_zeros(seed[0] + x, 4),
                            pad_zeros(seed[1] + y, 4),
                            pad_zeros(seed[2] + z, 4),
                            pad_zeros(seed[0] + x, 4),
                            pad_zeros(seed[1] + y, 4),
                            pad_zeros(seed[2] + z, 4))
                        seg = rmembranes[
                            z * config.shape[0]: z * config.shape[0] + config.shape[0],
                            y * config.shape[1]: y * config.shape[1] + config.shape[1],
                            x * config.shape[2]: x * config.shape[2] + config.shape[2]]
                        recursive_make_dir(path)
                        np.save(path, seg)
                        print("SAVED TO {}".format(path))
    return True  # Success!


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
        '--path_extent',
        dest='path_extent',
        type=str,
        default='9,9,3',  # '3,3,3',
        help='Provide extent of segmentation in 128^3 volumes.')
    parser.add_argument(
        '--segment_threshold',
        dest='segment_threshold',
        type=float,
        default=0.8,
        help='Segment threshold..')
    # parser.add_argument(
    #     '--membrane_slice',
    #     dest='membrane_slice',
    #     type=str,
    #     default=None,
    #     help='Membrane chunking along z axis.')
    parser.add_argument(
        '--validate',
        dest='validate',
        action='store_true',
        help='Force berson validation dataset.')
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
    print('Segmentation took {}'.format(end - start))
