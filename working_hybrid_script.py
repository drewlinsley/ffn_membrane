import numpy as np
from google.protobuf import text_format
from ffn.inference import inference
from ffn.inference import inference_pb2
from membrane.models import fgru_tmp as fgru
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def pad_zeros(x, total):
    """Pad x with zeros to total digits."""
    if not isinstance(x, basestring):
        x = str(x)
    total = total - len(x)
    for idx in range(total):
        x = '0' + x
    return x


# DEFAULTS
SHAPE = np.array([128, 128, 128])
CONF = [4992, 16000, 10112]
PATH_STR = '/local1/dlinsley/connectomics/mag1/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.raw'
MEM_STR = '/gpfs/data/tserre/data/tmp_ding/mag1/x%s/y%s/z%s/membrane_110629_k0725_mag1_x%s_y%s_z%s.npy'

# OPTIONS
MEMBRANE_MODEL = 'fgru_tmp'  # Allow for dynamic import; not yet implemented
MEMBRANE_CKPT = '/gpfs/data/tserre/data/connectomics/checkpoints/global_2_fb_wide_mini_fb_hgru3d_berson_0_berson_0_2018_10_29_14_58_16_883649/model_63000.ckpt-63000'
SEED = np.array([15, 15, 15])
PATH_EXTENT = [1, 3, 3]
FFN_TRANSPOSE = (0, 2, 1)
MEMBRANE_TYPE = 'probability'
NUM_RESTARTS = 1

# 1. select a volume
if np.all(PATH_EXTENT == 1):
    path = PATH_STR % (
        pad_zeros(SEED[0], 4),
        pad_zeros(SEED[1], 4),
        pad_zeros(SEED[2], 4),
        pad_zeros(SEED[0], 4),
        pad_zeros(SEED[1], 4),
        pad_zeros(SEED[2], 4))
    vol = np.fromfile(path, dtype='uint8').reshape(SHAPE)
else:
    vol = np.zeros((np.array(SHAPE) * PATH_EXTENT))
    for z in range(PATH_EXTENT[0]):
        for y in range(PATH_EXTENT[1]):
            for x in range(PATH_EXTENT[2]):
                path = PATH_STR % (
                    pad_zeros(SEED[0] + x, 4),
                    pad_zeros(SEED[1] + y, 4),
                    pad_zeros(SEED[2] + z, 4),
                    pad_zeros(SEED[0] + x, 4),
                    pad_zeros(SEED[1] + y, 4),
                    pad_zeros(SEED[2] + z, 4))
                v = np.fromfile(path, dtype='uint8').reshape(SHAPE)
                vol[
                    z * SHAPE[0] : z * SHAPE[0] + SHAPE[0],
                    y * SHAPE[1] : y * SHAPE[1] + SHAPE[1],
                    x * SHAPE[2] : x * SHAPE[2] + SHAPE[2]] = v
vol = vol.astype(np.float32) / 255.

# 2. Predict its membranes
model_shape = (SHAPE * PATH_EXTENT)
membranes = fgru.main(
    test=vol,
    evaluate=True,
    adabn=True,
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
vol = vol.transpose(FFN_TRANSPOSE)

# Un-normalize vol and stack
membranes = np.stack((vol * 255., proc_membrane), axis=-1)
mpath = MEM_STR % (
    pad_zeros(SEED[0], 4),
    pad_zeros(SEED[1], 4),
    pad_zeros(SEED[2], 4),
    pad_zeros(SEED[0], 4),
    pad_zeros(SEED[1], 4),
    pad_zeros(SEED[2], 4))
np.save(mpath, membranes)
print 'Saved membrane volume to %s' % mpath

# 4. Start FFN
for idx in range(NUM_RESTARTS):
    if idx == 0:
        seed_policy = 'PolicyPeaks'
    else:
        seed_policy = 'ShufflePolicyPeaks'
    config = '''image {hdf5: "%s"}
        image_mean: 128
        image_stddev: 33
        seed_policy: "%s"
        model_checkpoint_path: "/gpfs/data/tserre/data/connectomics/checkpoints/feedback_hgru_v5_3l_notemp_f_berson2x_w_memb_r0/model.ckpt-44450"
        model_name: "feedback_hgru_v5_3l_notemp_f.ConvStack3DFFNModel"
        model_args: "{\\"depth\\": 12, \\"fov_size\\": [57, 57, 13], \\"deltas\\": [8, 8, 3]}"
        segmentation_output_dir: "ding_segmentations/x%s/y%s/z%s/v%s/"
        inference_options {
            init_activation: 0.95
            pad_value: 0.05
            move_threshold: 0.6  # Used to be 0.7
            min_boundary_dist { x: 1 y: 1 z: 1}
            segment_threshold: 0.6
            min_segment_size: 4096
        }''' % (mpath, seed_policy, pad_zeros(SEED[0], 4), pad_zeros(SEED[1], 4), pad_zeros(SEED[2], 4), idx)

    req = inference_pb2.InferenceRequest()
    _ = text_format.Parse(config, req)
    if idx == 0:
        runner = inference.Runner()
    runner.start(req, tag='_inference')
    runner.run((0, 0, 0),
        (model_shape[0], model_shape[1], model_shape[2]))

