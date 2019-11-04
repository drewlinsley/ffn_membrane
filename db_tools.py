'''DB tools'''
import os
import numpy as np
from db import db
from glob2 import glob


VOLUME = '/media/data/connectomics/mag1/'
GLOB_MATCH = os.path.join(VOLUME, '**', '**', '**')


def main(
        init_db=False,
        populate_db=False,
        priority_list=None):
    """Routines for adjusting the DB."""
    if init_db:
        # Create the DB from a schema file
        db.initialize_database()
    if populate_db:
        # Fill the DB with a coordinates + global config
        coord_path = os.path.join(db, 'coordinates.npy')
        if os.path.exists(coord_path):
            coords = np.load(coord_path)
        else:
            print(
                'Gathering coordinates from: %s '
                '(this may take a while)' % coord_path)
            coords = glob(GLOB_MATCH)
            np.save(coord_path, coords)
        db.populate_db(coords)
    if priority_list is not None:
        # Add coordinates to the DB priority list
        pass
        db.add_priorities(priority_list)

