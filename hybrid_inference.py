import os
import argparse
import numpy as np
from google.protobuf import text_format
from ffn.inference import inference
from ffn.inference import inference_pb2
import nibabel as nib
from membrane.models import l3_fgru_constr as fgru
import logging
from config import Config


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def recursive_make_dir(path, s=3):
    """Recursively build output paths."""
    split_path = path.split(os.path.sep)
    for idx, p in enumerate(split_path):
        if idx > s:
            d = '/'.join(split_path[:idx])
            if not os.path.exists(d):
                os.makedirs(d)
                # print('Created: %s' % d)
            else:
                # print('Reusing: %s' % d)
                pass


def pad_zeros(x, total):
    """Pad x with zeros to total digits."""
    if not isinstance(x, basestring):
        x = str(x)
    total = total - len(x)
    for idx in range(total):
        x = '0' + x
    return x


def make_dir(d):
    """Make directory d if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


def rdirs(coors, path, its=3):
    """Recursively make paths."""
    paths = path.split('/')
    for idx in reversed(range(its)):
        if idx == 2:
            it_path = '/'.join(paths[:-(idx + 1)]) % (pad_zeros(coors[0], 4))
        elif idx == 1:
            it_path = '/'.join(paths[:-(idx + 1)]) % (
                pad_zeros(coors[0], 4), pad_zeros(coors[1], 4))
        elif idx == 0:
            it_path = '/'.join(paths[:-(idx + 1)]) % (
                pad_zeros(coors[0], 4),
                pad_zeros(coors[1], 4),
                pad_zeros(coors[2], 4))
        make_dir(it_path)
        print 'Made: %s' % it_path


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
        seg_vol=None,
        deltas='[15, 15, 3]',  # '[27, 27, 6]'
        seed_policy='PolicyMembrane',  # 'PolicyPeaks'
        debug=True,
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
                            v = np.fromfile(path, dtype='uint8').reshape(config.shape)
                            vol[
                                z * config.shape[0]: z * config.shape[0] + config.shape[0],
                                y * config.shape[1]: y * config.shape[1] + config.shape[1],
                                x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v
        else:
            data = np.load(config.test_segmentation_path)
            vol = data['volume'][:model_[0]]
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
        vol_ = vol.shape
        print('seed: %s' % seed)
        print('mpath: %s' % mpath)
        print('volume size: (%s, %s, %s)' % (
            vol_[0],
            vol_[1],
            vol_[2]))

        # 2. Predict its membranes
        membranes = fgru.main(
            test=vol,
            evaluate=True,
            adabn=True,
            gpu_device='/gpu:0',  # '/cpu:0',
            test_input_shape=np.concatenate((model_shape, [1])).tolist(),
            test_label_shape=np.concatenate((model_shape, [3])).tolist(),
            checkpoint=config.membrane_ckpt)

        # 3. Concat the volume w/ membranes and pass to FFN
        if membrane_type == 'probability':
            print 'Membrane: %s' % membrane_type
            proc_membrane = (
                membranes[0, :, :, :, :3].mean(-1)).transpose(ffn_transpose)
        elif membrane_type == 'threshold':
            print 'Membrane: %s' % membrane_type
            proc_membrane = (
                membranes[0, :, :, :, :3].mean(-1) > 0.5).astype(
                    int).transpose(ffn_transpose)
        else:
            raise NotImplementedError
        vol = vol.transpose(ffn_transpose)  # ).astype(np.uint8)
        membranes = np.stack(
            (vol, proc_membrane), axis=-1).astype(np.float32) * 255.
        if rotate:
            membranes = np.rot90(membranes, k=1, axes=(1, 2))
        np.save(mpath, membranes)
        print 'Saved membrane volume to %s' % mpath
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
     
    # PASS FLAG TO CHOOSE WHETHER OR NOT TO SAVE SEGMENTATIONS
    print 'Saving segmentations to: %s' % seg_dir
    # seg_vol = '/media/data_cifs/cluster_projects/ffn_membrane_v2/
    # ding_segmentations/x0015/y0015/z0018/v3/0/0/seg-0_0_0.npz'
    # shift_z, shift_y, shift_x = 0, 0, 256
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
                min_segment_size: 100
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
                min_segment_size: 100
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
                    img = snappy.compress(img)
    if debug:
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
                    v = nib.load(path).get_fdata()
                    vol[
                        z * config.shape[0]: z * config.shape[0] + config.shape[0],
                        y * config.shape[1]: y * config.shape[1] + config.shape[1],
                        x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v
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
        '--validate',
        dest='validate',
        action='store_true',
        help='Force berson validation dataset.')
    parser.add_argument(
        '--rotate',
        dest='rotate',
        action='store_true',
        help='Rotate the input data.')
    args = parser.parse_args()
    get_segmentation(**vars(args))

