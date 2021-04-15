import sys
import time
import os
import gzip
import fastremap
import numpy as np
from glob import glob
# from ffn.inference import segmentation
from skimage.segmentation import relabel_sequential as rfo
from tqdm import tqdm
from config import Config
import nibabel as nib
from scipy.spatial import distance
from utils.hybrid_utils import rdirs
from utils.hybrid_utils import pad_zeros
from utils.hybrid_utils import recursive_make_dir
from skimage import measure
# from numba import njit, jit, prange
from joblib import Parallel, delayed, parallel_backend

out_dir = '/gpfs/data/tserre/data/final_merge/'  # /localscratch/merge/'


try:
    from db import db
except:
    print("Failed to load db.")
try:
    import wkw
    NOWKW = False
except:
    print("Failed to load wkw.")
    NOWKW = True

# @jit(parallel=True, fastmath=True)
def npad_zeros(x, total):
    """Pad x with zeros to total digits."""
    if not isinstance(x, str):
        x = str(x)
    total = total - len(x)
    for idx in range(total):
        x = '0' + x
    return x


# @jit(parallel=True, fastmath=True)
def nrecursive_make_dir(path, s=3):
    """Recursively build output paths."""
    split_path = path.split(os.path.sep)
    for idx, p in enumerate(split_path):
        if idx > s:
            d = '/'.join(split_path[:idx])
            if not os.path.exists(d):
                os.makedirs(d)
                # print('Created: %s' % d)
            else:
                # print('Reusing: %s' % d)
                pass


def diff_data(zidx, z, unique_z, coordinates, config, mins):
    """Cube data in z."""

    # This plane
    z_sel_coors = coordinates[coordinates[:, 2] == z]
    z_sel_coors = np.unique(z_sel_coors, axis=0)

    # Next plane information
    if zidx < len(unique_z) - 1:
        z_next = unique_z[zidx + 1]
        max_z = z_next - z
    else:
        max_z = path_extent[-1]
        z_next = None

    # Allow for fast loading for debugging
    if os.path.exists(os.path.join(out_dir, 'plane_z{}.npy'.format(z))):
        # Load the current slab
        main = np.load(os.path.join(out_dir, 'plane_z{}.npy'.format(z)))
        # Combine the two into a merged slab
        merge_to = (path_extent[-1] - max_z) * config.shape[-1]  # ((z + path_extent[-1]) - z_next) * config.shape[-1]
        if merge_to < (config.shape[-1] * 3) and z_next is not None:
            # Load the next slab
            main_b = np.load(os.path.join(out_dir, 'plane_z{}.npy'.format(z_next)))
            main_b = main_b[..., merge_to // 2 : merge_to]  # noqa case of 1-offset: 128:256
            main_comp_a = main[..., -(merge_to // 2):]
            diff = np.maximum((main_comp_a > 0).astype(np.float32) - (main_b > 0).astype(np.float32), 0)

            # Store how much was missing previously in an array
            prop_miss.append(diff.sum() / float(main_b.size))

            # Remove regions smaller than a threshold
            diff = measure.label(diff)
            threshold = 512
            diff = morphology.remove_small_objects(diff, min_size=threshold)

            # Keep all main_b regions, propogate main_a regions but add an offset
            main_b += diff + max_vox

            # Now move main_b over
            main[..., -(merge_to // 2):] = main_b  # noqa case of 1-offset: 256: 

        if zidx < len(unique_z) - 1:
            max_z = max_z + 1  # Process until +1 min overlap
        if zidx > 0:
            min_z = 1  # Avoid the boundary FX on the 0th volume
        else:
            min_z = 0  # Unless we are on the 0th iteration


path_extent = [9, 9, 3]
glob_debug = False  # True
save_cubes = False
merge_debug = False
remap_labels = False
in_place = False
z_max = 384
bu_margin = 2
config = Config()

# Get list of coordinates
if NOWKW:
    db_og_coordinates = db.pull_membrane_coors()
    """
    if glob_debug:
        new_og_coordinates = []
        for r in db_og_coordinates:
            sel_coor = [r['x'], r['y'], r['z']]
            check = glob(os.path.join('/media/data_cifs/connectomics/ding_segmentations/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4))))
            if len(check):
                new_og_coordinates.append(sel_coor)
        og_coordinates = np.array(new_og_coordinates)
    else:
        og_coordinates = np.array([[r['x'], r['y'], r['z']] for r in db_og_coordinates if r['processed_segmentation']])
    """
    og_coordinates = np.array([[r['x'], r['y'], r['z']] for r in db_og_coordinates if r['processed_segmentation']])
    db_merges = db.pull_merge_membrane_coors()
    """
    if glob_debug:
        new_merges = []
        for r in db_merges:
            sel_coor = [r['x'], r['y'], r['z']]
            check = glob(os.path.join('/media/data_cifs/connectomics/ding_segmentations_merge/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4))))
            if len(check):
                new_merges.append(sel_coor)
        merges = np.array(new_merges)
    else:
        merges = np.array([[r['x'], r['y'], r['z']] for r in db_merges if r['processed_segmentation']])
    """
    merges = np.array([[r['x'], r['y'], r['z']] for r in db_merges if r['processed_segmentation']])
    np.savez("wkw_db_data", og_coordinates=og_coordinates, merges=merges)  # db_og_coordinates=db_og_coordinates, db_merges=db_merges, allow_pickle=False)
    os._exit(1)
else:
    db_data = np.load("wkw_db_data.npz")
    og_coordinates = db_data["og_coordinates"]
    merges = db_data["merges"]

# Loop over coordinates
coordinates = np.concatenate((og_coordinates, np.zeros_like(og_coordinates)[:, 0][:, None]), 1)
merges = np.concatenate((merges, np.ones_like(merges)[:, 0][:, None]), 1)
coordinates = np.concatenate((coordinates, merges))
unique_z = np.unique(coordinates[:, -2])
print(unique_z)
np.save('unique_zs_for_merge', unique_z)

# Compute extents
path_extent = np.array(path_extent)
mins = np.min(coordinates[:, :-1], axis=0)
maxs = np.max(coordinates[:, :-1], axis=0)  # Add extent to this
print("Bounding-box TL corner is: {}".format(mins * 128))

mins_vs = mins * config.shape  # (config.shape * np.array(path_extent))
maxs_vs = maxs * config.shape  # (config.shape * np.array(path_extent))
maxs_vs += config.shape * path_extent  # np.array(path_extent)
diffs = (maxs_vs - mins_vs)
xoff, yoff, zoff = path_extent * config.shape  # [:2]

# Loop through x-axis
use_parallel = True
dtype = np.uint32
max_vox, count, prev = 0, 0, None
slice_shape = np.concatenate((diffs[:-1], [z_max]))

for zidx, z in tqdm(enumerate(unique_z), total=len(unique_z), desc="Z-slice main clock"):
    diff_data(zidx, z, unique_z, coordinates, config, mins)
