'''DB tools'''
import os
import argparse
import numpy as np
import pandas as pd
from db import db
from glob2 import glob
from utils import logger
from config import Config
from utils import hybrid_utils


VOLUME = '/media/data/connectomics/mag1/'
GLOB_MATCH = os.path.join(VOLUME, '**', '**', '**', '*.raw')


def main(
        init_db=False,
        reset_coordinates=False,
        reset_priority=False,
        reset_config=False,
        populate_db=False,
        get_progress=False,
        berson_correction=True,
        segmentation_grid=None,
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
            x_range = np.arange(mins[0] + segmentation_grid[0], maxs[0], segmentation_grid[0])
            y_range = np.arange(mins[1] + segmentation_grid[1], maxs[1], segmentation_grid[1])
            z_range = np.arange(mins[2] + segmentation_grid[2], maxs[2], segmentation_grid[2])
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
        db.populate_db(coords)

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
    args = parser.parse_args()
    main(**vars(args))

