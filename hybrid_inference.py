import os
import argparse
import numpy as np
from google.protobuf import text_format
from ffn.inference import inference
from ffn.inference import inference_pb2
import nibabel as nib
from membrane.models import l3_fgru_constr as fgru
import logging

# DEFAULTS
SHAPE = np.array([128, 128, 128])
# CONF = [4992, 16000, 10112]
PATH_STR = '/media/data/connectomics/mag1/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.raw'  # nopep8
NII_PATH_STR = '/media/data/connectomics/mag1_segs/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.nii'  # nopep8
MEM_STR = '/media/data/membranes/mag1/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.raw'  # nopep8

# OPTIONS
MODEL = 'feedback_hgru_v5_3l_notemp_f_v4'
CKPT = '/media/data_cifs/connectomics/ffn_ckpts/64_fov/feedback_hgru_v5_3l_notemp_f_v4_berson4x_w_inf_memb_r0/model.ckpt-225915'  # nopep8
# MODEL = 'feedback_hgru_v5_3l_notemp_f_v5'
# CKPT= '/media/data_cifs/connectomics/ffn_ckpts/64_fov/feedback_hgru_v5_3l_notemp_f_v5_berson4x_w_inf_memb_r0/model.ckpt-263215'
# MEMBRANE_MODEL = 'fgru_tmp'  # Allow for dynamic import
MEMBRANE_CKPT = '/media/data_cifs/connectomics/checkpoints/l3_fgru_constr_berson_0_berson_0_2019_02_16_22_32_22_290193/model_137000.ckpt-137000'  # nopep8
# path_extent = [2, 3, 3]  # (256, 384, 384)
FFN_TRANSPOSE = (0, 1, 2)  # 0, 2, 1
# START = [50, 250, 200]
MEMBRANE_TYPE = 'probability'  # 'threshold'

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def recursive_make_dir(path):
    """Recursively build output paths."""
    split_path = path.split(os.path.sep)
    for idx, p in enumerate(split_path):
        if idx > 3:
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
        move_threshold=0.7,
        segment_threshold=0.5,
        validate=False,
        seed='14,15,18',
        shift_z=None,
        shift_y=None,
        shift_x=None,
        x=None,
        y=None,
        z=None,
        prev_coordinate=None,
        seg_vol=None,
        deltas='[15, 15, 3]',  # '[27, 27, 6]'
        seed_policy='PolicyMembrane',  # 'PolicyPeaks'
        debug=False,
        path_extent=[1, 1, 1],
        rotate=False):
    """Apply the FFN routines using fGRUs."""
    if seed is not None:
        SEED = np.array([int(s) for s in seed.split(',')])
    else:
        SEED = np.array([x, y, z])
    rdirs(SEED, MEM_STR)
    model_shape = (SHAPE * path_extent)
    mpath = MEM_STR % (
        pad_zeros(SEED[0], 4),
        pad_zeros(SEED[1], 4),
        pad_zeros(SEED[2], 4),
        pad_zeros(SEED[0], 4),
        pad_zeros(SEED[1], 4),
        pad_zeros(SEED[2], 4))
    if idx == 0:
        # 1. select a volume
        if not validate:
            if np.all(path_extent == 1):
                path = PATH_STR % (
                    pad_zeros(SEED[0], 4),
                    pad_zeros(SEED[1], 4),
                    pad_zeros(SEED[2], 4),
                    pad_zeros(SEED[0], 4),
                    pad_zeros(SEED[1], 4),
                    pad_zeros(SEED[2], 4))
                vol = np.fromfile(path, dtype='uint8').reshape(SHAPE)
            else:
                vol = np.zeros((np.array(SHAPE) * path_extent))
                for z in range(path_extent[0]):
                    for y in range(path_extent[1]):
                        for x in range(path_extent[2]):
                            path = PATH_STR % (
                                pad_zeros(SEED[0] + x, 4),
                                pad_zeros(SEED[1] + y, 4),
                                pad_zeros(SEED[2] + z, 4),
                                pad_zeros(SEED[0] + x, 4),
                                pad_zeros(SEED[1] + y, 4),
                                pad_zeros(SEED[2] + z, 4))
                            v = np.fromfile(path, dtype='uint8').reshape(SHAPE)
                            vol[
                                z * SHAPE[0]: z * SHAPE[0] + SHAPE[0],
                                y * SHAPE[1]: y * SHAPE[1] + SHAPE[1],
                                x * SHAPE[2]: x * SHAPE[2] + SHAPE[2]] = v
        else:
            data = np.load(
                '/media/data_cifs/connectomics/datasets/berson_0.npz')
            vol = data['volume'][:model_shape[0]]
            SEED = [99, 99, 99]
            mpath = MEM_STR % (
                pad_zeros(SEED[0], 4),
                pad_zeros(SEED[1], 4),
                pad_zeros(SEED[2], 4),
                pad_zeros(SEED[0], 4),
                pad_zeros(SEED[1], 4),
                pad_zeros(SEED[2], 4))
            rdirs(SEED, MEM_STR)
        vol = vol.astype(np.float32) / 255.
        vol_shape = vol.shape
        print('seed: %s' % SEED)
        print('mpath: %s' % mpath)
        print('volume size: (%s, %s, %s)' % (
            vol_shape[0],
            vol_shape[1],
            vol_shape[2]))

        # 2. Predict its membranes
        membranes = fgru.main(
            test=vol,
            evaluate=True,
            adabn=True,
            gpu_device='/cpu:0',
            test_input_shape=np.concatenate((model_shape, [1])).tolist(),
            test_label_shape=np.concatenate((model_shape, [12])).tolist(),
            checkpoint=MEMBRANE_CKPT)

        # 3. Concat the volume w/ membranes and pass to FFN
        if MEMBRANE_TYPE == 'probability':
            print 'Membrane: %s' % MEMBRANE_TYPE
            proc_membrane = (
                membranes[0, :, :, :, :3].mean(-1)).transpose(FFN_TRANSPOSE)
        elif MEMBRANE_TYPE == 'threshold':
            print 'Membrane: %s' % MEMBRANE_TYPE
            proc_membrane = (
                membranes[0, :, :, :, :3].mean(-1) > 0.5).astype(
                    int).transpose(FFN_TRANSPOSE)
        else:
            raise NotImplementedError
        vol = vol.transpose(FFN_TRANSPOSE)  # ).astype(np.uint8)
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
        shifts = (SEED - prev_coordinate) * SHAPE
        shift_x, shift_y, shift_z = shifts

    if validate:
        SEED = [99, 99, 99]

    seg_dir = 'ding_segmentations/x%s/y%s/z%s/v%s/' % (
        pad_zeros(SEED[0], 4),
        pad_zeros(SEED[1], 4),
        pad_zeros(SEED[2], 4),
        idx)
    print 'Saving segmentations to: %s' % seg_dir
    # seg_vol = '/media/data_cifs/cluster_projects/ffn_membrane_v2/
    # ding_segmentations/x0015/y0015/z0018/v3/0/0/seg-0_0_0.npz'
    # shift_z, shift_y, shift_x = 0, 0, 256
    if seg_vol is not None:
        config = '''image {hdf5: "%s"}
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
            CKPT,
            MODEL,
            deltas,
            shift_z, shift_y, shift_x,
            seg_vol,
            seg_dir,
            move_threshold,
            segment_threshold)
    else:
        config = '''image {hdf5: "%s"}
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
            CKPT,
            MODEL,
            deltas,
            seg_dir,
            move_threshold,
            segment_threshold)
    req = inference_pb2.InferenceRequest()
    _ = text_format.Parse(config, req)
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
                path = NII_PATH_STR % (
                    pad_zeros(SEED[0] + x, 4),
                    pad_zeros(SEED[1] + y, 4),
                    pad_zeros(SEED[2] + z, 4),
                    pad_zeros(SEED[0] + x, 4),
                    pad_zeros(SEED[1] + y, 4),
                    pad_zeros(SEED[2] + z, 4))
                seg = segments[
                    z * SHAPE[0]: z * SHAPE[0] + SHAPE[0],
                    y * SHAPE[1]: y * SHAPE[1] + SHAPE[1],
                    x * SHAPE[2]: x * SHAPE[2] + SHAPE[2]]
                recursive_make_dir(path)

                # Save as .nii
                img = nib.Nifti1Image(seg, np.eye(4))
                nib.save(img, path)
    if debug:
        # Reconstruct from .nii files
        vol = np.zeros((np.array(SHAPE) * path_extent))
        for z in range(path_extent[0]):
            for y in range(path_extent[1]):
                for x in range(path_extent[2]):
                    path = NII_PATH_STR % (
                        pad_zeros(SEED[0] + x, 4),
                        pad_zeros(SEED[1] + y, 4),
                        pad_zeros(SEED[2] + z, 4),
                        pad_zeros(SEED[0] + x, 4),
                        pad_zeros(SEED[1] + y, 4),
                        pad_zeros(SEED[2] + z, 4))
                    v = nib.load(path).get_fdata()
                    vol[
                        z * SHAPE[0]: z * SHAPE[0] + SHAPE[0],
                        y * SHAPE[1]: y * SHAPE[1] + SHAPE[1],
                        x * SHAPE[2]: x * SHAPE[2] + SHAPE[2]] = v
        assert np.all(vol == segments), 'Mismatch in .nii reconstruction.'
        print('.nii reconstruction matches segments from FFN.')
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
