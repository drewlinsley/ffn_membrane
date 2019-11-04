'''DB tools'''
import os
import argparse
import numpy as np
import pandas as pd
from db import db
from glob2 import glob
from utils import logger
from config import Config


VOLUME = '/media/data/connectomics/mag1/'
GLOB_MATCH = os.path.join(VOLUME, '**', '**', '**', '*.raw')


def main(
        init_db=False,
        populate_db=False,
        priority_list=None):
    """Routines for adjusting the DB."""
    config = Config()
    log = logger.get(os.path.join(config.log_dir, 'setup'))
    if init_db:
        # Create the DB from a schema file
        db.initialize_database()
        log.info('Initialized database.')

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
        db.populate_db(coords)

    if priority_list is not None:
        # Add coordinates to the DB priority list
        assert '.csv' in priority_list, 'Priorities must be a csv.'
        priorities = pd.read_csv(priority_list)
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
        '--priority_list',
        dest='priority_list',
        type=str,
        default=None,  # 'db/priorities.csv'
        help='Path of CSV with priority coordinates '
        '(see db/priorities.csv for example).')
    args = parser.parse_args()
    main(**vars(args))
