import os
import time
import logging
import argparse
import numpy as np
import nibabel as nib
from db import db
from config import Config
from utils.hybrid_utils import pad_zeros, make_dir


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
    except Exception as e:
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



