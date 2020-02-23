import os
import time
import logging
import argparse
import numpy as np
import nibabel as nib
from db import db
from config import Config
from membrane.models import seung_unet3d_adabn_small as unet
from utils.hybrid_utils import pad_zeros, make_dir
from skimage.feature import peak_local_max


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_data(config, seed, pull_from_db, return_membrane=False):
    if not pull_from_db:
        seed = seed
    else:
        seed = db.get_next_synapse_coordinate()
        if seed is None:
            raise RuntimeError('No more coordinantes to process!')
    path = config.path_str % (
        pad_zeros(seed['x'], 4),
        pad_zeros(seed['y'], 4),
        pad_zeros(seed['z'], 4),
        pad_zeros(seed['x'], 4),
        pad_zeros(seed['y'], 4),
        pad_zeros(seed['z'], 4))
    vol = np.fromfile(path, dtype='uint8').reshape(config.shape)
    try:
        membrane_p = nib.load(
            path.replace(
                '/mag1/', '/mag1_membranes_nii/').replace(
                '.raw', '.nii'))
    except Exception, e:
        return None, e
    membrane = membrane_p.get_data()
    membrane_p.uncache()
    if return_membrane:
        return membrane
    # Check vol/membrane scale
    # vol = (vol / 255.).astype(np.float32)
    membrane[np.isnan(membrane)] = 0.
    vol = np.stack((vol, membrane), -1)[None] / 255.
    return vol, None


def process_preds(preds, config, thresh=[0.9, 0.8]):
    """Extract likely synapse locations."""
    # Threshold and save results
    # Set threshold. Also potentially set
    # it to be lower for amacrine, with a WTA.
    thresh_preds_r = np.clip(preds[..., 0], thresh[0], 1.1)
    thresh_preds_a = np.clip(preds[..., 1], thresh[1], 1.1)
    thresh_preds_r[thresh_preds_r <= thresh[0]] = 0.
    thresh_preds_a[thresh_preds_a <= thresh[1]] = 0.
    thresh_preds = np.stack((thresh_preds_r, thresh_preds_a), -1)

    # Take max per 3d coordinate
    max_vals = np.max(thresh_preds, -1)
    argmax_vals = np.argmax(thresh_preds, -1)

    # Find peaks
    peaks = peak_local_max(max_vals, min_distance=24)

    # Put back into array
    # fixed_array = np.zeros_like(thresh_preds)
    # fixed_array[..., 0] += (max_vals * (argmax_vals == 0).astype(np.float32))
    # fixed_array[..., 1] += (max_vals * (argmax_vals == 1).astype(np.float32))

    ribbon_coords, amacrine_coords = [], []
    for p in peaks:
        ch = argmax_vals[p[0], p[1], p[2]]
        if ch == 0:
            ribbon_coords += [p]
        else:
            amacrine_coords += [p]
    # # Take local max
    # ribbon_coords = peak_local_max(fixed_preds[..., 0], min_distance=32)
    # amacrine_coords = peak_local_max(fixed_preds[..., 1], min_distance=32)

    # # Add coords to the db
    # adjusted_ribbon_coordinates = [
    #     coord * config.shape for coord in ribbon_coords]
    # adjusted_amacrine_coordinates = [
    #     coord * config.shape for coord in amacrine_coords]
    synapses = []
    for s in ribbon_coords:
        synapses += [{'x': s[0], 'y': s[1], 'z': s[2], 'type': 'ribbon'}]
    for s in amacrine_coords:
        synapses += [{'x': s[0], 'y': s[1], 'z': s[2], 'type': 'amacrine'}]
    return synapses


def test(
        ffn_transpose=(0, 1, 2),
        cube_size=128,
        output_dir='synapse_predictions_v0',
        # ckpt_path='new_synapse_checkpoints_new_dataloader_smaller_weight/30000/30000-30000.ckpt',  # noqa
        ckpt_path='new_synapse_checkpoints_new_dataloader_smaller_weight/-65000.ckpt',  # noqa
        paths='/media/data_cifs/connectomics/membrane_paths.npy',
        pull_from_db=False,
        keep_processing=False,
        path_extent=None,
        save_preds=False,
        seed=(15, 10, 10),
        rotate=False):
    """Apply the FFN routines using fGRUs."""
    config = Config()
    out_path = os.path.join(config.project_directory, output_dir)
    make_dir(out_path)
    model_shape = [128, 128, 128, 2]  # noqa list(np.load(synapse_files[0])['vol'].shape)[1:]
    num_completed = 0
    if keep_processing and pull_from_db:
        while keep_processing:
            seed = db.get_next_synapse_coordinate()
            if seed is None:
                print('No more synapse coordinates to process. Finished!')
                os._exit(1)
            vol, error = get_data(
                seed=seed, pull_from_db=pull_from_db, config=config)
            if vol is None:
                # No membranes found. Push this to DB
                db.missing_membrane([seed])
                print('Failed: {}'.format(error))
                continue
            if num_completed == 0:
                preds, sess, test_dict = unet.main(
                    test=vol,
                    evaluate=True,
                    adabn=True,
                    return_sess=keep_processing,
                    test_input_shape=model_shape,
                    test_label_shape=model_shape,
                    checkpoint=ckpt_path,
                    gpu_device='/gpu:0')
                preds = preds[0].squeeze()
            else:
                feed_dict = {
                    test_dict['test_images']: vol,
                }
                it_test_dict = sess.run(
                    test_dict,
                    feed_dict=feed_dict)
                preds = it_test_dict['test_logits'].squeeze()
            synapses = process_preds(preds, config)

            # Add to DB
            db.add_synapses(synapses)
            print(
                'Finished {}. Found {} synapses.'.format(
                    num_completed, len(synapses)))
            num_completed += 1

            if save_preds:
                # Save raw to file structure
                it_out = out_path.replace(
                    os.path.join(config.project_directory, 'mag1'), out_path)
                it_out = os.path.sep.join(it_out.split(os.path.sep)[:-1])
                np.save(it_out, preds)
    else:
        vol = get_data(seed=seed, pull_from_db=pull_from_db, config=config)
        preds = unet.main(
            test=vol,
            evaluate=True,
            adabn=True,
            return_sess=keep_processing,
            test_input_shape=model_shape,
            test_label_shape=model_shape,
            checkpoint=ckpt_path,
            gpu_device='/gpu:0')
        preds = preds[0].squeeze()
        synapses = process_preds(preds, config)

        # Add to DB
        db.add_synapses(synapses)
        if save_preds:
            # Save raw to file structure
            it_out = out_path.replace(
                os.path.join(config.project_directory, 'mag1'), out_path)
            it_out = os.path.sep.join(it_out.split(os.path.sep)[:-1])
            np.save(it_out, preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_extent',
        dest='path_extent',
        type=str,
        default='3,3,3',
        help='Provide extent of segmentation in 128^3 volumes.')
    parser.add_argument(
        '--keep_processing',
        dest='keep_processing',
        action='store_true',
        help='Get coords.')
    parser.add_argument(
        '--pull_from_db',
        dest='pull_from_db',
        action='store_true',
        help='Get coords.')
    args = parser.parse_args()
    start = time.time()
    test(**vars(args))
    end = time.time()
    print('Testing took {}'.format(end - start))

