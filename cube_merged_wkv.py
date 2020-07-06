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


# @jit(parallel=True, fastmath=True)
def convert_save_cubes(coords, data, cifs_path, mins, max_z, config, dataset, corner):
    """All coords come from the same z-slice. Save these as npys to cifs."""
    # for x in range(path_extent[0]):
    #     for y in range(path_extent[1]):
    #          for z in range(max_z):
    # from skimage.segmentation import relabel_sequential as rs;from matplotlib import pyplot as plt;
    # plt.imshow(rs(seg.astype(np.uint32)[64])[0]);plt.savefig("512_512_0.png")
    for z in range(max_z):
        for y in range(path_extent[1]):
            for x in range(path_extent[0]):
                seg = data[
                    x * config.shape[0]: x * config.shape[0] + config.shape[0],
                    y * config.shape[1]: y * config.shape[1] + config.shape[1],
                    z * config.shape[2]: z * config.shape[2] + config.shape[2]]
                """
                it_corner = np.asarray([(x + corner[0]) * config.shape[0], (y + corner[1]) * config.shape[1], (z + corner[2]) * config.shape[2]])
                dataset.write(it_corner, seg)
                """
                seg = data[
                    z * config.shape[0]: z * config.shape[0] + config.shape[0],
                    y * config.shape[1]: y * config.shape[1] + config.shape[1],
                    x * config.shape[2]: x * config.shape[2] + config.shape[2]]
                it_corner = np.asarray([(x + corner[0]) * config.shape[0], (y + corner[1]) * config.shape[1], (z + corner[2]) * config.shape[2]])
                dataset.write(it_corner, seg)


path_extent = [9, 9, 3]
glob_debug = True
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
    if glob_debug:
        new_og_coordinates = []
        for r in db_og_coordinates:
            sel_coor = [r['x'], r['y'], r['z']]
            check = glob(os.path.join('/media/data_cifs/connectomics/ding_segmentations/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4))))
            if len(check):
                new_og_coordinates.append(sel_coor)
        og_coordinates = np.array(new_og_coordinates)
    else:
        og_coordinates = np.array([[r['x'], r['y'], r['z']] for r in og_coordinates if r['processed_segmentation']])

    db_merges = db.pull_merge_membrane_coors()
    if glob_debug:
        new_merges = []
        for r in db_merges:
            sel_coor = [r['x'], r['y'], r['z']]
            check = glob(os.path.join('/media/data_cifs/connectomics/ding_segmentations_merge/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4))))
            if len(check):
                new_merges.append(sel_coor)
        merges = np.array(new_merges)
    else:
        merges = np.array([[r['x'], r['y'], r['z']] for r in merges if r['processed_segmentation']])
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
mins_vs = mins * config.shape  # (config.shape * np.array(path_extent))
maxs_vs = maxs * config.shape  # (config.shape * np.array(path_extent))
maxs_vs += config.shape * path_extent  # np.array(path_extent)
diffs = (maxs_vs - mins_vs)
xoff, yoff, zoff = path_extent * config.shape  # [:2]

# Loop through x-axis
max_vox, count, prev = 0, 0, None
slice_shape = np.concatenate((diffs[:-1], [z_max]))
dataset = wkw.Dataset.open(
    "/media/data_cifs/connectomics/merge_data_wkw/1",
    wkw.Header(np.uint32))
cifs_stem = '/media/data_cifs/connectomics/merge_data_nii_raw_v2/'
cifs_path = '{}/1/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.raw'.format(cifs_stem)
out_dir = '/gpfs/data/tserre/data/final_merge/'  # /localscratch/merge/'
for zidx, z in tqdm(enumerate(unique_z), total=len(unique_z), desc="Z-slice main clock"):
    # # Allocate tensor
    # main = np.zeros(slice_shape, np.uint32)

    # This plane
    z_sel_coors = coordinates[coordinates[:, 2] == z]
    # sort_idx = np.argsort(z_sel_coors, -1)[::-1]
    # z_sel_coors = z_sel_coors[sort_idx]
    z_sel_coors = np.unique(z_sel_coors, axis=0)

    # Next plane information
    if zidx < len(unique_z):
        z_next = unique_z[zidx + 1]
        max_z = z_next - z
        # max_z = dz * config.shape[-1]
    else:
        max_z = path_extent[-1]
        # max_z = path_extent[-1] * config.shape[-1]
    # Allow for fast loading for debugging
    if os.path.exists(os.path.join(out_dir, 'plane_z{}.npy'.format(z))):
        main = np.load(os.path.join(out_dir, 'plane_z{}.npy'.format(z)))
        main = main.transpose((2, 1, 0))  # Needed to go from ZYX (segs) -> XYZ (raws)
        for sel_coor in tqdm(z_sel_coors, desc='Z (saving): {}'.format(z)):
            adj_coor = (sel_coor[:-1] - mins) * config.shape
            vol = main[
                :,
                adj_coor[1]: adj_coor[1] + yoff,
                adj_coor[0]: adj_coor[0] + xoff]
            # vol = vol.transpose((2, 1, 0))  # Needed to go from ZYX (segs) -> XYZ (raws)
            convert_save_cubes(
                dataset=dataset,
                # corner=adj_coor,
                corner=sel_coor[:-1],
                data=vol,
                coords=sel_coor,
                cifs_path=cifs_path,
                mins=mins,
                max_z=max_z,
                config=config)

