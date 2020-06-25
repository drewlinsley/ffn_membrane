import os
import time
import logging
import argparse
import numpy as np
import nibabel as nib
from db import db
from config import Config
from membrane.models import seung_unet3d_adabn_small as unet
from membrane.models import l3_fgru_constr as fgru
from utils.hybrid_utils import pad_zeros, make_dir
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential
from utils.hybrid_utils import recursive_make_dir as rdirs
from pull_and_convert_predicted_synapses import convert_synapse_predictions


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def process_cubes(cubes, debug_coords, debug, num_completed, seed, ribbons, amacrines, keep_processing, ckpt_path, device, config, vol):
    """Get synapse preds for cubes."""
    if 1:
        model_shape = list(cubes[0].shape)
        if debug:
            debug_vol = np.zeros(list(vol.shape))
            # debug_vol = np.zeros(model_shape)
        else:
            debug_vol = None

        synapses = []
        for cube, dcoords in zip(cubes, debug_coords):
            if num_completed == 0:
                preds, sess, test_dict = unet.main(
                    test=cube[None],
                    evaluate=True,
                    adabn=True,
                    return_sess=keep_processing,
                    test_input_shape=model_shape,
                    test_label_shape=model_shape,
                    checkpoint=ckpt_path,
                    gpu_device=device)
                preds = preds[0].squeeze()
            else:
                feed_dict = {
                    test_dict['test_images']: cube[None],
                }
                it_test_dict = sess.run(
                    test_dict,
                    feed_dict=feed_dict)
                preds = it_test_dict['test_logits'].squeeze()
            new_seed = np.array(
                [
                    dcoords['d_s'],
                    dcoords['h_s'],
                    dcoords['w_s']]) + np.array(
                [
                    seed['x'],
                    seed['y'],
                    seed['z']]) * config.shape
            it_synapse, it_ribbons, it_amacrines = process_preds(
                preds, config, offset=new_seed)
            synapses += it_synapse
            ribbons += it_ribbons
            amacrines += it_amacrines
            num_completed += 1
            if debug:
                debug_vol[
                    dcoords['d_s']: dcoords['d_e'],
                    dcoords['h_s']: dcoords['h_e'],
                    dcoords['w_s']: dcoords['w_e']] = preds
    return synapses, ribbons, amacrines, debug_vol


def cube_data(vol, model_shape, divs):
    """Chunk up data into cubes for processing separately."""
    # Reshape vol into 9 cubes and process each
    cubes = []
    assert model_shape[1] / divs[1] == np.round(model_shape[1] / divs[1])
    d_ind_start = np.arange(0, model_shape[0], model_shape[0] / divs[0])
    h_ind_start = np.arange(0, model_shape[1], model_shape[1] / divs[1])
    w_ind_start = np.arange(0, model_shape[2], model_shape[2] / divs[2])
    d_ind_end = d_ind_start + model_shape[0] / divs[0]
    h_ind_end = h_ind_start + model_shape[1] / divs[1]
    w_ind_end = w_ind_start + model_shape[2] / divs[2]
    debug_coords = []
    for d_s, d_e in zip(d_ind_start, d_ind_end):
        for h_s, h_e in zip(h_ind_start, h_ind_end):
            for w_s, w_e in zip(w_ind_start, w_ind_end):
                cubes += [vol[d_s: d_e, h_s: h_e, w_s: w_e]]
                debug_coords += [
                    {
                        'd_s': d_s,
                        'd_e': d_e,
                        'h_s': h_s,
                        'h_e': h_e,
                        'w_s': w_s,
                        'w_e': w_e
                    }
                ]
    return cubes, debug_coords


def get_data_old(config, seed, pull_from_db, return_membrane=False):
    if not pull_from_db:
        seed = seed
    else:
        seed = db.get_next_synapse_coordinate()
        if seed is None:
            raise RuntimeError('No more coordinantes to process!')
    if not pull_from_db:
        path = config.path_str % (
            pad_zeros(seed[0], 4),
            pad_zeros(seed[1], 4),
            pad_zeros(seed[2], 4),
            pad_zeros(seed[0], 4),
            pad_zeros(seed[1], 4),
            pad_zeros(seed[2], 4))
    else:
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


def get_data_working(config, seed, pull_from_db, return_membrane=False):
    if not pull_from_db:
        seed = seed
    else:
        seed = db.get_next_synapse_coordinate()
        if seed is None:
            raise RuntimeError('No more coordinantes to process!')
    path = config.mem_str % (
        pad_zeros(seed['x'], 4),
        pad_zeros(seed['y'], 4),
        pad_zeros(seed['z'], 4),
        pad_zeros(seed['x'], 4),
        pad_zeros(seed['y'], 4),
        pad_zeros(seed['z'], 4))
    path = '{}.npy'.format(path)
    if os.path.exists(path):
        membrane = np.load(path)
        assert membrane.max > 1, 'Membrane is scaled to [0, 1]. Fix this!'
        if return_membrane:
            return membrane
        # Check vol/membrane scale
        # vol = (vol / 255.).astype(np.float32)
        membrane[np.isnan(membrane)] = 0.
        # vol = np.stack((vol, membrane), -1)[None] / 255.
        membrane /= 255.
        return membrane, None
    else:
        return None, None


def get_data(config, seed, pull_from_db, return_membrane=False, path_extent=[3, 9, 9]):
    if not pull_from_db:
        seed = seed
    else:
        seed = db.get_next_synapse_coordinate()
        if seed is None:
            raise RuntimeError('No more coordinantes to process!')
    vol = np.zeros((np.array(config.shape) * path_extent))
    mem = np.zeros((np.array(config.shape) * path_extent))
    for z in range(path_extent[0]):
        for y in range(path_extent[1]):
            for x in range(path_extent[2]):
                vol_path = config.path_str % (
                    pad_zeros(seed['x'] + x, 4),
                    pad_zeros(seed['y'] + y, 4),
                    pad_zeros(seed['z'] + z, 4),
                    pad_zeros(seed['x'] + x, 4),
                    pad_zeros(seed['y'] + y, 4),
                    pad_zeros(seed['z'] + z, 4))
                v = np.fromfile(vol_path, dtype='uint8').reshape(config.shape)
                vol[
                    z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                    y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                    x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v  # nopep8
                mem_path = config.nii_mem_str % (
                    pad_zeros(seed['x'] + x, 4),
                    pad_zeros(seed['y'] + y, 4),
                    pad_zeros(seed['z'] + z, 4),
                    pad_zeros(seed['x'] + x, 4),
                    pad_zeros(seed['y'] + y, 4),
                    pad_zeros(seed['z'] + z, 4))
                h = nib.load(mem_path)
                v = np.array(h.get_data())
                h.uncache()
                mem[
                    z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                    y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                    x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v  # nopep8

    assert mem.max > 1, 'Membrane is scaled to [0, 1]. Fix this!'
    if return_membrane:
        return mem
    # Check vol/membrane scale
    mem[np.isnan(mem)] = 0.
    mem = np.stack((vol, mem), -1)
    mem /= 255.
    return mem, None


def get_data_or_process(config, seed, pull_from_db, return_membrane=False, path_extent=[3, 9, 9], feed_dict=None, sess=None, test_dict=None):
    if not pull_from_db:
        seed = seed
    else:
        seed = db.get_next_synapse_coordinate()
        if seed is None:
            raise RuntimeError('No more coordinantes to process!')
    vol = np.zeros((np.array(config.shape) * path_extent))
    mem = np.zeros((np.array(config.shape) * path_extent))
    for z in range(path_extent[0]):
        for y in range(path_extent[1]):
            for x in range(path_extent[2]):
                mem_path = config.nii_mem_str % (
                    pad_zeros(seed['x'] + x, 4),
                    pad_zeros(seed['y'] + y, 4),
                    pad_zeros(seed['z'] + z, 4),
                    pad_zeros(seed['x'] + x, 4),
                    pad_zeros(seed['y'] + y, 4),
                    pad_zeros(seed['z'] + z, 4))
                vol_path = config.path_str % (
                    pad_zeros(seed['x'] + x, 4),
                    pad_zeros(seed['y'] + y, 4),
                    pad_zeros(seed['z'] + z, 4),
                    pad_zeros(seed['x'] + x, 4),
                    pad_zeros(seed['y'] + y, 4),
                    pad_zeros(seed['z'] + z, 4))
                if not (os.path.exists(mem_path) and os.path.exists(vol_path)):
                    v = np.fromfile(vol_path, dtype='uint8').reshape(config.shape)
                    vol[
                        z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                        y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                        x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v  # nopep8
                    if sess is None:
                        membranes, sess, test_dict = fgru.main(
                            test=v.reshape(np.concatenate([[1], config.shape, [1]])),
                            evaluate=True,
                            adabn=True,
                            gpu_device='/gpu:0',
                            return_sess=True,
                            test_input_shape=np.concatenate((
                                config.shape, [1])).tolist(),
                            test_label_shape=np.concatenate((
                                config.shape, [3])).tolist(),
                            checkpoint=config.membrane_ckpt)
                        mem = membranes[0].squeeze(0).mean(-1)
                        if mem.max() <= 1.:
                            mem = mem * 255.
                        img = nib.Nifti1Image(mem, np.eye(4))
                        rdirs(mem_path)
                        nib.save(img, mem_path)
                    else:
                        feed_dict = {
                            test_dict['test_images']: v.reshape(np.concatenate([[1], config.shape, [1]])),
                        }
                        it_test_dict = sess.run(
                            test_dict,
                            feed_dict=feed_dict)
                        mem = it_test_dict['test_logits'].squeeze(0).mean(-1)
                        if mem.max() <= 1.:
                            mem = mem * 255.
                        img = nib.Nifti1Image(mem, np.eye(4))
                        rdirs(mem_path)
                        nib.save(img, mem_path)
    return sess, feed_dict, test_dict


def process_preds(preds, config, offset, thresh=[0.90, 0.51], so_thresh=27):
    """Extract likely synapse locations."""
    # Threshold and save results
    # Set threshold. Also potentially set
    # it to be lower for amacrine, with a WTA.
    thresh_preds_r = np.clip(preds[..., 0], thresh[0], 1.1)
    thresh_preds_a = np.clip(preds[..., 1], thresh[1], 1.1)
    thresh_preds_r[thresh_preds_r <= thresh[0]] = 0.
    thresh_preds_a[thresh_preds_a <= thresh[1]] = 0.
    thresh_preds = np.stack((thresh_preds_r, thresh_preds_a), -1)
    thresh_pred_mask = remove_small_objects(thresh_preds > 0.5, so_thresh)
    thresh_preds *= thresh_pred_mask

    # Take max per 3d coordinate
    max_vals = np.max(thresh_preds, -1)
    argmax_vals = np.argmax(thresh_preds, -1)

    # Find peaks
    peaks = peak_local_max(max_vals, min_distance=28)
    # ids = relabel_sequential(thresh_pred_mask)[0]

    # Split into ribbon/amacrine
    ribbon_coords, amacrine_coords = [], []
    for p in peaks:
        ch = argmax_vals[p[0], p[1], p[2]]
        adj_p = offset + p
        # size = np.sum(ids[..., ch] == ids[p[0], p[1], p[2], ch])
        size = thresh_preds[p[0], p[1], p[2], ch]
        adj_p = np.concatenate((adj_p, [size]))
        if ch == 0:
            ribbon_coords += [adj_p]
        else:
            amacrine_coords += [adj_p]
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
        synapses += [
            {'x': s[0], 'y': s[1], 'z': s[2], 'size': s[3], 'type': 'ribbon'}]  # noqa
    for s in amacrine_coords:
        synapses += [
            {'x': s[0], 'y': s[1], 'z': s[2], 'size': s[3], 'type': 'amacrine'}]  # noqa
    return synapses, len(ribbon_coords), len(amacrine_coords)


def test(
        ffn_transpose=(0, 1, 2),
        cube_size=128,
        output_dir='synapse_predictions_v0',
        # ckpt_path='new_synapse_checkpoints_new_dataloader_smaller_weight/30000/30000-30000.ckpt',  # noqa
        # ckpt_path='new_synapse_checkpoints_new_dataloader_smaller_weight/-65000.ckpt',  # noqa
        ckpt_path='new_synapse_checkpoints_new_dataloader_smaller_weight/-85000.ckpt',  # noqa
        paths='/media/data_cifs/connectomics/membrane_paths.npy',
        pull_from_db=False,
        keep_processing=False,
        path_extent=None,
        save_preds=False,
        divs=[6, 2, 2],
        debug=False,
        out_dir=None,
        segmentation_path=None,
        finish_membranes=False,
        seed=(15, 10, 10),
        device="/gpu:0",
        rotate=False):
    """Apply the FFN routines using fGRUs."""
    config = Config()
    path_extent = np.array([int(s) for s in path_extent.split(',')])
    out_path = os.path.join(config.project_directory, output_dir)
    make_dir(out_path)
    num_completed, fixed_membranes = 0, 0
    ribbons = 0
    amacrines = 0
    if segmentation_path is not None:
        seed = np.asarray([int(x) for x in segmentation_path.split(",")])
        seed = {"x": seed[0], "y": seed[1], "z": seed[2]}
        try:
            vol, error = get_data(
                seed=seed, pull_from_db=False, config=config)
        except Exception as e:
            print(e)
            import ipdb;ipdb.set_trace()
            sess, feed_dict, test_dict = get_data_or_process(
                seed=seed, pull_from_db=False, config=config)
        model_shape = list(vol.shape)
        """
        preds = unet.main(
            test=vol[None],
            evaluate=True,
            adabn=True,
            return_sess=keep_processing,
            test_input_shape=model_shape,
            test_label_shape=model_shape,
            checkpoint=ckpt_path,
            gpu_device=device)
        preds = preds[0].squeeze()
        synapses = process_preds(preds, config, offset=0)
        """
        cubes, debug_coords = cube_data(vol=vol, model_shape=model_shape, divs=divs)
        synapses, ribbons, amacrines, debug_vol = process_cubes(cubes, debug_coords, debug, num_completed, seed, ribbons, amacrines,  keep_processing, ckpt_path, device, config, vol)
        convert_synapse_predictions(
            offset=seed,
            synapse_list=synapses,
            out_ribbon_file=os.path.join(out_dir, "{}_ribbon.nml".format(segmentation_path)))
        return vol, debug_vol
    elif keep_processing and pull_from_db:
        while keep_processing:
            seed = db.get_next_synapse_coordinate()
            if seed is None:
                print('No more synapse coordinates to process. Finished!')
                os._exit(1)
            # CHECK THIS -- MAKE SURE DATA REFLECTS HYBRID_... IT IS EFFED RIGHT NOW
            # Compare the ding vol to the constituent niis
            try:
                vol, error = get_data(
                    seed=seed, pull_from_db=pull_from_db, config=config)
            except Exception as e:
                print(e)
                if finish_membranes:
                    if fixed_membranes == 0:
                        sess, feed_dict, test_dict = get_data_or_process(
                            seed=seed, pull_from_db=pull_from_db, config=config)
                    else:
                        get_data_or_process(
                            seed=seed, pull_from_db=pull_from_db, config=config, feed_dict=feed_dict, sess=sess, test_dict=test_dict)
                    fixed_membranes += 1
                    print('Fixed {} {} {} membranes.'.format(seed['x'], seed['y'], seed['z']))
                    continue
                else:
                    vol = None
            if finish_membranes:
                continue
            if vol is None:
                # No membranes found. Push this to DB
                db.missing_membrane([seed])
                print('Failed: {}')
                continue
            model_shape = list(vol.shape)

            """"
            # Reshape vol into 9 cubes and process each
            cubes = []
            assert model_shape[1] / divs[1] == np.round(model_shape[1] / divs[1])
            d_ind_start = np.arange(0, model_shape[0], model_shape[0] / divs[0])
            h_ind_start = np.arange(0, model_shape[1], model_shape[1] / divs[1])
            w_ind_start = np.arange(0, model_shape[2], model_shape[2] / divs[2])
            d_ind_end = d_ind_start + model_shape[0] / divs[0]
            h_ind_end = h_ind_start + model_shape[1] / divs[1]
            w_ind_end = w_ind_start + model_shape[2] / divs[2]
            debug_coords = []
            for d_s, d_e in zip(d_ind_start, d_ind_end):
                for h_s, h_e in zip(h_ind_start, h_ind_end):
                    for w_s, w_e in zip(w_ind_start, w_ind_end):
                        cubes += [vol[d_s: d_e, h_s: h_e, w_s: w_e]]
                        debug_coords += [
                            {
                                'd_s': d_s,
                                'd_e': d_e,
                                'h_s': h_s,
                                'h_e': h_e,
                                'w_s': w_s,
                                'w_e': w_e
                            }
                        ]
            if debug:
                debug_vol = np.zeros(model_shape)
            model_shape = list(cubes[0].shape)
            synapses = []
            for cube, dcoords in zip(cubes, debug_coords):
                if num_completed == 0:
                    preds, sess, test_dict = unet.main(
                        test=cube[None],
                        evaluate=True,
                        adabn=True,
                        return_sess=keep_processing,
                        test_input_shape=model_shape,
                        test_label_shape=model_shape,
                        checkpoint=ckpt_path,
                        gpu_device=device)
                    preds = preds[0].squeeze()
                else:
                    feed_dict = {
                        test_dict['test_images']: cube[None],
                    }
                    it_test_dict = sess.run(
                        test_dict,
                        feed_dict=feed_dict)
                    preds = it_test_dict['test_logits'].squeeze()
                new_seed = np.array(
                    [
                        dcoords['d_s'],
                        dcoords['h_s'],
                        dcoords['w_s']]) + np.array(
                    [
                        seed['x'],
                        seed['y'],
                        seed['z']]) * config.shape
                it_synapse, it_ribbons, it_amacrines = process_preds(
                    preds, config, offset=new_seed)
                synapses += it_synapse
                ribbons += it_ribbons
                amacrines += it_amacrines
                num_completed += 1
                if debug:
                    debug_vol[
                        dcoords['d_s']: dcoords['d_e'],
                        dcoords['h_s']: dcoords['h_e'],
                        dcoords['w_s']: dcoords['w_e']] = preds
            """
            cubes, debug_coords = cube_data(vol=vol, model_shape=model_shape, divs=divs)
            synapses, ribbons, amacrines, debug_vol = process_cubes(cubes, debug_coords, debug, num_completed, seed, ribbons, amacrines,  keep_processing, ckpt_path, device, config, vol)
            # Add to DB
            db.add_synapses(synapses)
            print(
                'Finished {}. Found {} ribbons and {} amacrines.'.format(
                    num_completed, ribbons, amacrines))
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
            gpu_device=device)
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
        default='9,9,3',
        help='Provide extent of segmentation in 128^3 volumes.')
    parser.add_argument(
        '--segmentation_path',
        dest='segmentation_path',
        type=str,
        default=None,
        help='Path to existing segmentation file that you want to get synapses for.')
    parser.add_argument(
        '--device',
        dest='device',
        type=str,
        default="/gpu:0",
        help="String for the device to use.")
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
    parser.add_argument(
        '--debug',
        dest='debug',
        action='store_true',
        help='Debug preds.')
    parser.add_argument(
        '--finish_membranes',
        dest='finish_membranes',
        action='store_true',
        help='Finish membrane generation.')
    args = parser.parse_args()
    start = time.time()
    test(**vars(args))
    end = time.time()
    print('Testing took {}'.format(end - start))
