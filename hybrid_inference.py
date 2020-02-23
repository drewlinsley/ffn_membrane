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
from synapse_test import get_data as get_membranes


logger = logging.getLogger()
logger.setLevel(logging.INFO)
AUGS = ['lr_flip', 'ud_flip', 'depth_flip']  # , 'rot90', 'rot180', 'rot270']
ROTS = []  # ['rot90', 'rot180']  # 'rot90', 'rot180', 'rot270']
TEST_TIME_AUGS = reduce(
    lambda x, y: list(
        itertools.combinations(AUGS, y)) + x,
    range(len(AUGS) + 1), [])[:-1]
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
    assert move_threshold is not None
    assert segment_threshold is not None
    rdirs(seed, config.mem_str)
    model_shape = (config.shape * path_extent)
    mpath = config.mem_str % (
        pad_zeros(seed[0], 4),
        pad_zeros(seed[1], 4),
        pad_zeros(seed[2], 4),
        pad_zeros(seed[0], 4),
        pad_zeros(seed[1], 4),
        pad_zeros(seed[2], 4))
    if idx == 0:
        # 1. select a volume
        if not validate:
            if np.all(path_extent == 1):
                path = config.path_str % (
                    pad_zeros(seed[0], 4),
                    pad_zeros(seed[1], 4),
                    pad_zeros(seed[2], 4),
                    pad_zeros(seed[0], 4),
                    pad_zeros(seed[1], 4),
                    pad_zeros(seed[2], 4))
                vol = np.fromfile(path, dtype='uint8').reshape(config.shape)
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
        else:
            data = np.load(config.test_segmentation_path)
            vol = data['volume'][:model_shape[0]]
            seed = [99, 99, 99]
            mpath = config.mem_str % (
                pad_zeros(seed[0], 4),
                pad_zeros(seed[1], 4),
                pad_zeros(seed[2], 4),
                pad_zeros(seed[0], 4),
                pad_zeros(seed[1], 4),
                pad_zeros(seed[2], 4))
            rdirs(seed, config.mem_str)
        vol = vol.astype(np.float32) / 255.
        _vol = vol.shape
        print('seed: %s' % seed)
        print('mpath: %s' % mpath)
        print('volume size: (%s, %s, %s)' % (
            _vol[0],
            _vol[1],
            _vol[2]))

        # 2. Predict its membranes
        predict_membranes = True
        if segment_only:
            try:
                seed_dict = {
                    xc: se for se, xc in zip(seed, ['x', 'y', 'z'])}
                membranes = np.zeros((np.array(config.shape) * path_extent))
                for z in range(path_extent[0]):
                    for y in range(path_extent[1]):
                        for x in range(path_extent[2]):
                            seed_dict = {xc: se for se, xc in zip(
                                [seed[0] + x, seed[1] + y, seed[2] + z],
                                ['x', 'y', 'z'])}
                            m = get_membranes(
                                seed=seed_dict,
                                pull_from_db=False,
                                config=config,
                                return_membrane=True)
                            membranes[
                                z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                                y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                                x * config.shape[2]: x * config.shape[2] + config.shape[2]] = m  # nopep8
                membranes[np.isnan(membranes)] = 0.
                membranes /= 255.
                predict_membranes = False
                print('Restored membranes from previous run.')
            except Exception as e:
                print('Error: {}'.format(e))
                print(
                    'Failed to load membranes for this location.'
                    'Rerunning them (Slow!).')
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
                original_vol = np.copy(vol)
                try:
                    vol = np.stack(vols)
                except Exception:
                    print(
                        'Mismatch in volume_size/membrane slicing {}'.format(
                            [vs.shape for vs in vols]))
                    os._exit(1)
                del vols  # Garbage collect
                membrane_model_shape = membrane_slice
            print membrane_model_shape
            print vol.shape
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
                    aug_vol = augment(vo=vol, augs=it_aug)
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
            if membrane_slice is not None:
                # Reconstruct, accounting for overlap
                # membrane_model_shape = tuple(list(_vol) + [3])
                rmembranes = np.zeros(_vol, dtype=np.float32)
                count = 0
                vols = []
                normalization = np.zeros_like(rmembranes)
                if TEST_TIME_AUGS is not None:
                    bump_map = _bump_logit_map(membranes[count].shape)
                    bump_map = 1 - bump_map / bump_map.min()
                else:
                    bump_map = 1.
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
                                xu: xo] += membranes[count] * bump_map
                            if normalization is not None:
                                normalization[
                                    zu: zo,
                                    yu: yo,
                                    xu: xo] += bump_map  # 1.
                            count += 1
                if normalization is not None:
                    rmembranes /= normalization
                membranes = rmembranes  # [None]
                vol = original_vol
                del rmembranes, normalization

            # 3. Concat the volume w/ membranes and pass to FFN
            # if membrane_type == 'probability':
            #     print 'Membrane: %s' % membrane_type
            #     proc_membrane = (
            #         membranes[0, :, :, :, :3].mean(-1)).transpose(ffn_transpose)
            # elif membrane_type == 'threshold':
            #     print 'Membrane: %s' % membrane_type
            #     proc_membrane = (
            #         membranes[0, :, :, :, :3].mean(-1) > 0.5).astype(
            #             membranes.dtype).transpose(ffn_transpose)
            # else:
            #     raise NotImplementedError

        vol = vol.transpose(ffn_transpose)  # ).astype(np.uint8)
        membranes = np.stack(
            (vol, membranes), axis=-1).astype(np.float32) * 255.
        if rotate:
            membranes = np.rot90(membranes, k=1, axes=(1, 2))
        np.save(mpath, membranes)
        print 'Saved membrane volume to %s' % mpath
        if predict_membranes:
            del bump_map
        del vol  # Garbage collect
    if not membrane_only:
        del membranes
    if membrane_only:
        for z in range(path_extent[0]):
            for y in range(path_extent[1]):
                for x in range(path_extent[2]):
                    path = config.nii_mem_str % (
                        pad_zeros(seed[0] + x, 4),
                        pad_zeros(seed[1] + y, 4),
                        pad_zeros(seed[2] + z, 4),
                        pad_zeros(seed[0] + x, 4),
                        pad_zeros(seed[1] + y, 4),
                        pad_zeros(seed[2] + z, 4))
                    mem = membranes[
                        z * config.shape[0]: z * config.shape[0] + config.shape[0],  # noqa
                        y * config.shape[1]: y * config.shape[1] + config.shape[1],  # noqa
                        x * config.shape[2]: x * config.shape[2] + config.shape[2], 1]  # noqa
                    recursive_make_dir(path)
                    img = nib.Nifti1Image(mem, np.eye(4))
                    nib.save(img, path)
        return True, True, True
    mpath = '%s.npy' % mpath

    # 4. Start FFN
    if prev_coordinate is not None:
        # Compute shift offset from previous segmentation coord
        prev_coordinate = np.array(prev_coordinate)
        shifts = (seed - prev_coordinate) * config.shape
        shift_x, shift_y, shift_z = shifts
        seg_vol = os.path.join(
            config.ffn_formatted_output % (
                pad_zeros(prev_coordinate[0], 4),
                pad_zeros(prev_coordinate[1], 4),
                pad_zeros(prev_coordinate[2], 4),
                0),
            '0',
            '0',
            'seg-0_0_0.npz')

    if validate:
        seed = [99, 99, 99]
    seg_dir = config.ffn_formatted_output % (
        pad_zeros(seed[0], 4),
        pad_zeros(seed[1], 4),
        pad_zeros(seed[2], 4),
        idx)
    recursive_make_dir(seg_dir)

    # Ran into an error with the 0/0 folders not being made sometimes -- Why?
    t_seg_dir = os.path.join(seg_dir, '0', '0')
    recursive_make_dir(t_seg_dir)

    # PASS FLAG TO CHOOSE WHETHER OR NOT TO SAVE SEGMENTATIONS
    print 'Saving segmentations to: %s' % seg_dir
    if seg_vol is not None:
        ffn_config = '''image {hdf5: "%s"}
            image_mean: 128
            image_stddev: 33
            seed_policy: "%s"
            model_checkpoint_path: "%s"
            model_name: "%s.ConvStack3DFFNModel"
            model_args: "{\\"depth\\": 12, \\"fov_size\\": [64, 64, 16], \\"deltas\\": %s, \\"shifts\\": [%s, %s, %s]}"
            init_segmentation: {hdf5: "%s"}
            segmentation_output_dir: "%s"
            inference_options {
                init_activation: 0.95
                pad_value: 0.05
                move_threshold: %s
                min_boundary_dist { x: 1 y: 1 z: 1}
                segment_threshold: %s
                min_segment_size: 1000
            }''' % (
            mpath,
            seed_policy,
            config.ffn_ckpt,
            config.ffn_model,
            deltas,
            shift_z, shift_y, shift_x,
            seg_vol,
            seg_dir,
            move_threshold,
            segment_threshold)
    else:
        ffn_config = '''image {hdf5: "%s"}
            image_mean: 128
            image_stddev: 33
            seed_policy: "%s"
            model_checkpoint_path: "%s"
            model_name: "%s.ConvStack3DFFNModel"
            model_args: "{\\"depth\\": 12, \\"fov_size\\": [64, 64, 16], \\"deltas\\": %s}"
            segmentation_output_dir: "%s"
            inference_options {
                init_activation: 0.95
                pad_value: 0.05
                move_threshold: %s
                min_boundary_dist { x: 1 y: 1 z: 1}
                segment_threshold: %s
                min_segment_size: 1000
            }''' % (
            mpath,
            seed_policy,
            config.ffn_ckpt,
            config.ffn_model,
            deltas,
            seg_dir,
            move_threshold,
            segment_threshold)
    req = inference_pb2.InferenceRequest()
    _ = text_format.Parse(ffn_config, req)
    runner = inference.Runner()
    runner.start(req, tag='_inference')
    _, segments, probabilities = runner.run(
        (0, 0, 0),
        (model_shape[0], model_shape[1], model_shape[2]))

    # Copy the nii file to the appropriate path
    # Try to pull segments and probability from runner
    # segments = np.load(
    #     os.path.join(seg_dir, '0', '0', 'seg-0_0_0.npz'))['segmentation']
    for z in range(path_extent[0]):
        for y in range(path_extent[1]):
            for x in range(path_extent[2]):
                path = config.nii_path_str % (
                    pad_zeros(seed[0] + x, 4),
                    pad_zeros(seed[1] + y, 4),
                    pad_zeros(seed[2] + z, 4),
                    pad_zeros(seed[0] + x, 4),
                    pad_zeros(seed[1] + y, 4),
                    pad_zeros(seed[2] + z, 4))
                seg = segments[
                    z * config.shape[0]: z * config.shape[0] + config.shape[0],
                    y * config.shape[1]: y * config.shape[1] + config.shape[1],
                    x * config.shape[2]: x * config.shape[2] + config.shape[2]]
                recursive_make_dir(path)

                if savetype == '.nii':
                    img = nib.Nifti1Image(seg, np.eye(4))
                    nib.save(img, path)
                elif savetype == '.sz':
                    raise NotImplementedError
                    # img = snappy.compress(img)
    if debug_nii:
        # Reconstruct from .nii files
        vol = np.zeros((np.array(config.shape) * path_extent))
        for z in range(path_extent[0]):
            for y in range(path_extent[1]):
                for x in range(path_extent[2]):
                    path = config.nii_path_str % (
                        pad_zeros(seed[0] + x, 4),
                        pad_zeros(seed[1] + y, 4),
                        pad_zeros(seed[2] + z, 4),
                        pad_zeros(seed[0] + x, 4),
                        pad_zeros(seed[1] + y, 4),
                        pad_zeros(seed[2] + z, 4))
                    v = nib.load(path).get_data()
                    vol[
                        z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                        y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                        x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v  # nopep8
        assert np.all(vol == segments), 'Mismatch in .nii reconstruction.'
        print('.nii reconstruction matches segments from FFN.')
        out_path = os.path.join(
            seg_dir,
            '0',
            '0',
            'reconstruction')
        np.save(out_path, vol)
    return True, segments, probabilities  # Success!


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
        default='1,1,1',  # '3,3,3',
        help='Provide extent of segmentation in 128^3 volumes.')
    parser.add_argument(
        '--move_threshold',
        dest='move_threshold',
        type=float,
        default=0.7,
        help='Movement threshold. Higher is more likely to move.')
    parser.add_argument(
        '--segment_threshold',
        dest='segment_threshold',
        type=float,
        default=0.5,
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
    args = parser.parse_args()
    start = time.time()
    get_segmentation(**vars(args))
    end = time.time()
    print('Segmentation took {}'.format(end - start))
