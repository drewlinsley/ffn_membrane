import logging
# import sys
# import os
import numpy as np
from db import db
from cell_type_test import get_segmentation
from skimage import measure
import argparse
import pandas as pd
# from config import Config


def get_new_coors(x, y, z, next_direction, stride):
    """Consolidate this ugly conditional for iterating coordinates."""
    if next_direction == 0:
        x -= stride[0]
    elif next_direction == 1:
        y -= stride[1]
    elif next_direction == 2:
        z -= stride[2]
    elif next_direction == 3:
        x += stride[0]
    elif next_direction == 4:
        y += stride[1]
    elif next_direction == 5:
        z += stride[2]
    return x, y, z


def main(
        segment_threshold=0.8,
        idx=0,
        membrane_only=False,
        segment_only=False,
        merge_segment_only=False,
        path_extent=[9, 9, 3],  # 9,9,3  x/y/z 128 voxel cube extent
        seg_ordering=[2, 1, 0],  # transpose to z/y/x for segmentation
        offset=[32, 32, 8],  # Should be 1/2 FOV in FFN
        stride=[5, 5, 2]):  # [3, 3, 3]  # x/y/z
    """Run a worker by pulling volume info from the DB."""
    # config = Config()
    path_extent = np.array(path_extent)
    stride = np.array(stride)
    next_coordinate = db.get_next_muller_coordinate()
    if next_coordinate is None:
        # No need to process this point
        logging.exception('No more coordinates found!')
        return
    x, y, z = next_coordinate['x'], next_coordinate['y'], next_coordinate['z']  # noqa
    prev_coordinate = None

    # Run segmentation
    try:
        success = get_segmentation(
            idx=idx,  # Force membrane detection
            seed=None,
            segment_threshold=segment_threshold,
            x=x,
            y=y,
            z=z,
            path_extent=path_extent[[seg_ordering]])
    except Exception as e:
        logging.exception('Failed segmentation: {}'.format(e))
        print('Failed segmentation: {}'.format(e))
        success = False

    # Update DB with results
    if success:
        d = [{"x": x, "y": y, "z": z}]
        db.finish_coordinate_muller(d)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--idx',
        dest='idx',
        type=int,
        default=0,
        help='Segmentation version.')
    parser.add_argument(
        '--membrane_only',
        dest='membrane_only',
        action='store_true',
        help='Only process membranes.')
    parser.add_argument(
        '--segment_only',
        dest='segment_only',
        action='store_true',
        help='Only segment cells.')
    parser.add_argument(
        '--merge_segment_only',
        dest='merge_segment_only',
        action='store_true',
        help='Only segment merge locations.')
    args = parser.parse_args()
    main(**vars(args))
