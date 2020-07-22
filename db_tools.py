'''DB tools'''
import os
import argparse
import numpy as np
import pandas as pd
from db import db
from glob2 import glob
from utils import logger
from config import Config
from tqdm import tqdm
from utils import hybrid_utils
from utils.hybrid_utils import pad_zeros


VOLUME = '/media/data_cifs/connectomics/mag1/'
SYNAPSE = '/media/data_cifs/connectomics/mag1_membranes_nii'
GLOB_MATCH = os.path.join(VOLUME, '**', '**', '**', '*.raw')
# GLOB_MATCH_S = os.path.join(SYNAPSE, '**', '**', '**', '*.nii')


def main(
        init_db=False,
        reset_coordinates=False,
        reset_priority=False,
        reset_config=False,
        reset_synapses=False,
        populate_db=False,
        populate_synapses=False,
        get_progress=False,
        berson_correction=True,
        segmentation_grid=None,
        merge_coordinates=False,
        extra_merges=True,
        priority_list=None):
    """Routines for adjusting the DB."""
    config = Config()
    log = logger.get(os.path.join(config.log_dir, 'setup'))
    if init_db:
        # Create the DB from a schema file
        db.initialize_database()
        log.info('Initialized database.')

    if reset_coordinates:
        # Create the DB from a schema file
        db.reset_database()
        log.info('Reset coordinates.')

    if populate_db:
        # Fill the DB with a coordinates + global config
        assert config.coord_path is not None
        if os.path.exists(config.coord_path):
            coords = np.load(config.coord_path)
        else:
            print(
                'Gathering coordinates from: %s '
                '(this may take a while)' % config.coord_path)
            coords = glob(GLOB_MATCH)
            coords = [os.path.sep.join(
                x.split(os.path.sep)[:-1]) for x in coords]
            np.save(config.coord_path, coords)
        if segmentation_grid is not None:
            # Quantize the coordinates according to segmentation_grid
            segmentation_grid = [int(x) for x in segmentation_grid.split(',')]
            ids = []
            for coord in coords:
                sx = coord.split(os.path.sep)[-3].strip('x')
                sy = coord.split(os.path.sep)[-2].strip('y')
                sz = coord.split(os.path.sep)[-1].strip('z')
                count, process = 0, True
                while process:
                    if sx[count] != '0':
                        sx = sx[count:]
                        process = False
                    count += 1
                    if count > 3:
                        process = False
                count, process = 0, True
                while process:
                    if sy[count] != '0':
                        sy = sy[count:]
                        process = False
                    count += 1
                    if count > 3:
                        process = False
                count, process = 0, True
                while process:
                    if sz[count] != '0':
                        sz = sz[count:]
                        process = False
                    count += 1
                    if count > 3:
                        process = False
                if len(sx) == 4:
                    sx = 0
                if len(sy) == 4:
                    sy = 0
                if len(sz) == 4:
                    sz = 0
                ids += [[int(sx), int(sy), int(sz)]]
            ids = np.array(ids)
            mins = ids.min(0)
            maxs = ids.max(0)
            x_range = np.arange(
                mins[0] + segmentation_grid[0], maxs[0], segmentation_grid[0])
            y_range = np.arange(
                mins[1] + segmentation_grid[1], maxs[1], segmentation_grid[1])
            z_range = np.arange(
                mins[2] + segmentation_grid[2], maxs[2], segmentation_grid[2])
            new_coords = []
            for x in x_range:
                for y in y_range:
                    for z in z_range:
                        coord = np.array([x, y, z])
                        ds = np.abs(coord - ids).sum(-1)
                        best_c = ids[np.argmin(ds)]
                        new_coords += [os.path.join(
                            VOLUME,
                            'x{}'.format(hybrid_utils.pad_zeros(best_c[0], 4)),
                            'y{}'.format(hybrid_utils.pad_zeros(best_c[1], 4)),
                            'z{}'.format(hybrid_utils.pad_zeros(best_c[2], 4)))
                        ]
            coords = np.array(new_coords)
            coords = np.unique(coords, axis=0)
        # db.populate_db(coords)

    if reset_synapses:
        print('Deleting from synapses table.')
        db.reset_synapses_table()
        print('Deleting from synapse_list table.')
        db.reset_synapse_list_table()

    if populate_synapses:
        # Fill the DB with a coordinates + global config
        assert config.synapse_coord_path is not None
        if os.path.exists(config.synapse_coord_path):
            coords = np.load(config.synapse_coord_path)
        else:
            print(
                'Gathering coordinates from: %s '
                '(this may take a while)' % config.synapse_coord_path)
            # coords = glob(GLOB_MATCH_S)
            # coords = [os.path.sep.join(
            #     x.split(os.path.sep)[:-1]) for x in coords]
            all_segment_coords = np.array(db.pull_membrane_coors())
            db_paths = []
            for row in all_segment_coords:
                db_paths += [[row['x'], row['y'], row['z']]]
            coords = np.unique(np.array(db_paths), axis=0)
            # np.save(config.synapse_coord_path, coords)
        coords = np.unique(coords, axis=0)
        db.populate_synapses(coords, str_input=False)

    if merge_coordinates:
        raw_offsets = np.array([3, 9, 9])[::-1]  # Hardcoded for the segs
        offsets = raw_offsets // 2
        coords = db.pull_membrane_coors()
        xyzs = [[d['x'], d['y'], d['z']] for d in coords]
        xyzs = np.unique(xyzs, axis=0)
        # Z is the short dimension
        # Add separate translations in the -x/-y/-z
        new_xyzs = []
        for xyz in tqdm(xyzs, desc='Checking offset paths', total=len(xyzs)):
            xyz0 = np.copy(xyz)
            xyz1 = np.copy(xyz)
            xyz2 = np.copy(xyz)
            xyz01 = np.copy(xyz)
            xyz02 = np.copy(xyz)
            xyz12 = np.copy(xyz)
            xyz0[0] -= offsets[0]
            xyz1[1] -= offsets[1]
            xyz2[2] -= offsets[2]
            xyz01[0] -= offsets[0]
            xyz02[0] -= offsets[0]
            xyz01[1] -= offsets[1]
            xyz12[1] -= offsets[1]
            add_xyzs = [xyz0, xyz1, xyz2, xyz01, xyz02, xyz12]
            for new_xyz in add_xyzs:
                path_test = []
                for tx in range(new_xyz[0], new_xyz[0] + raw_offsets[0]):
                    for ty in range(new_xyz[1], new_xyz[1] + raw_offsets[1]):
                        for tz in range(new_xyz[2], new_xyz[2] + raw_offsets[2]):
                            tp = config.path_str % (
                                pad_zeros(tx, 4),
                                pad_zeros(ty, 4),
                                pad_zeros(tz, 4),
                                pad_zeros(tx, 4),
                                pad_zeros(ty, 4),
                                pad_zeros(tz, 4))
                            if os.path.exists(tp):
                                path_test.append(True)
                if np.all(path_test):
                    new_xyzs += [new_xyz]
        new_xyzs = np.unique(np.array(new_xyzs), axis=0)
        debug_merge = False
        if debug_merge:
            # Check that all connected raw files exist
            missing_raws = []
            for xyz in tqdm(new_xyzs, desc='Checking RAW files', total=len(new_xyzs)):
                for xo in range(raw_offsets[0]):
                    for yo in range(raw_offsets[1]):
                        for zo in range(raw_offsets[2]):
                            new_xyz = xyz + np.array([xo, yo, zo])
                            test_path = config.path_str % (
                                pad_zeros(new_xyz[0], 4),
                                pad_zeros(new_xyz[1], 4),
                                pad_zeros(new_xyz[2], 4),
                                pad_zeros(new_xyz[0], 4),
                                pad_zeros(new_xyz[1], 4),
                                pad_zeros(new_xyz[2], 4))
                            if os.path.exists(test_path):
                                missing_raws += [xyz]
            # Check that all connected membrane files exist
            missing_mems = []
            for xyz in tqdm(new_xyzs, desc='Checking MEM files', total=len(new_xyzs)):
                for xo in range(raw_offsets[0]):
                    for yo in range(raw_offsets[1]):
                        for zo in range(raw_offsets[2]):
                            new_xyz = xyz + np.array([xo, yo, zo])
                            test_path = config.nii_mem_str % (
                                pad_zeros(new_xyz[0], 4),
                                pad_zeros(new_xyz[1], 4),
                                pad_zeros(new_xyz[2], 4),
                                pad_zeros(new_xyz[0], 4),
                                pad_zeros(new_xyz[1], 4),
                                pad_zeros(new_xyz[2], 4))
                            if os.path.exists(test_path):
                                missing_mems += [xyz]
            np.savez('debug', missing_raws=missing_raws, missing_mems=missing_mems)
            os._exit(1)

        # Get existing coords and push the exclusive ones
        existing_merges = db.pull_merge_membrane_coors()
        existing_merges = np.asarray([[d['x'], d['y'], d['z']] for d in existing_merges])
        final_xyzs = []
        for r in new_xyzs:
            # check = r - existing_merges
            # check = (check != 0).sum(-1) == 0
            check = np.abs(r - existing_merges).sum(-1) == 0
            if not np.any(check):  # noqa If this coord has not yet been run -- no matches
                final_xyzs.append(r)
        if len(final_xyzs):
            db.populate_db(coords=np.asarray(final_xyzs), merge_coordinates=True)
        else:
            print("No remaining merge coordinates to add to the DB.")

    if extra_merges:
        em = np.load('final_merges.npy')[:, :-1]
        ems = []
        for e in em:
            ems.append({'x': e[0], 'y': e[1], 'z': e[2]})
        # db.populate_db(coords=em, merge_coordinates=True)
        em = db.reset_merges(coords=ems)

    if reset_priority:
        # Create the DB from a schema file
        db.reset_priority()
        log.info('Reset priorities.')

    if reset_config:
        # Create the global config to starting values
        db.reset_config()
        log.info('Reset config.')

    if get_progress:
        # Return the status of segmentation
        db.get_progress()

    if priority_list is not None:
        # Add coordinates to the DB priority list
        assert '.csv' in priority_list, 'Priorities must be a csv.'
        priorities = pd.read_csv(priority_list)
        if berson_correction:
            max_chain = db.get_max_chain_id() + 1
            chains = range(max_chain, max_chain + len(priorities))
            priorities.x //= 128
            priorities.y //= 128
            priorities.z //= 128
            priorities['prev_chain_idx'] = 0
            priorities['chain_id'] = chains
            priorities['processed'] = False
            priorities['force'] = True
            db.update_max_chain_id(np.max(chains))
        db.add_priorities(priorities)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--init_db',
        dest='init_db',
        action='store_true',
        help='Initialize DB from schema (must be run at least once).')
    parser.add_argument(
        '--populate_db',
        dest='populate_db',
        action='store_true',
        help='Add all coordinates to the DB.')
    parser.add_argument(
        '--populate_synapses',
        dest='populate_synapses',
        action='store_true',
        help='Add all coordinates to the DB.')
    parser.add_argument(
        '--reset_coordinates',
        dest='reset_coordinates',
        action='store_true',
        help='Reset coordinate progress.')
    parser.add_argument(
        '--reset_priority',
        dest='reset_priority',
        action='store_true',
        help='Reset coordinate progress.')
    parser.add_argument(
        '--reset_config',
        dest='reset_config',
        action='store_true',
        help='Reset global config.')
    parser.add_argument(
        '--reset_synapses',
        dest='reset_synapses',
        action='store_true',
        help='Reset synapse tables.')
    parser.add_argument(
        '--get_progress',
        dest='get_progress',
        action='store_true',
        help='Return proportion-finished of total segmentation.')
    parser.add_argument(
        '--berson_correction',
        dest='berson_correction',
        action='store_false',
        help='No berson coordinate correction.')
    parser.add_argument(
        '--priority_list',
        dest='priority_list',
        type=str,
        default=None,  # 'db/priorities.csv'
        help='Path of CSV with priority coordinates '
        '(see db/priorities.csv for example).')
    parser.add_argument(
        '--segmentation_grid',
        dest='segmentation_grid',
        type=str,
        default=None,  # 'db/priorities.csv'
        help='Quantize the coordinate space.')
    parser.add_argument(
        '--add_merge_coordinates',
        dest='merge_coordinates',
        action='store_true',
        help='Add merge coordinates.')
    parser.add_argument(
        '--extra_merges',
        dest='extra_merges',
        action='store_true',
        help='Add merge coordinates.')
    args = parser.parse_args()
    main(**vars(args))

