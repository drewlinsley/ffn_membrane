import sys
import os
import numpy as np
from db import db
from glob import glob
from utils.hybrid_utils import pad_zeros
from ffn.inference import segmentation
from skimage.segmentation import relabel_sequential as rfo
from tqdm import tqdm
from config import Config
import nibabel as nib
from scipy.spatial import distance
from utils.hybrid_utils import rdirs, pad_zeros
import scipy.sparse
from skimage import measure


def load_npz(sel_coor):
    """First try loading from main segmentations, then the merges.

    Later, add loading for nii as the fallback."""
    path = os.path.join('/media/data_cifs/connectomics/ding_segmentations/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4)))
    merge = False
    if not os.path.exists(path):
        path = os.path.join('/media/data_cifs/connectomics/ding_segmentations_merge/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4)))
        merge = True
    if not os.path.exists(path):
        raise RuntimeError('Path not found: %s' % path)
        # Add nii loading here...
    zp = np.load(path)  # ['segmentation']
    vol = zp['segmentation']
    del zp.f
    zp.close()
    return vol


path_extent = [9, 9, 3]
config = Config()

# Get list of coordinates
og_coordinates = db.pull_membrane_coors()
og_coordinates = np.array([[r['x'], r['y'], r['z']] for r in og_coordinates if r['processed_segmentation']])

# Get list of merges
merges = db.pull_merge_membrane_coors()
merges = np.array([[r['x'], r['y'], r['z']] for r in merges if r['processed_segmentation']])

# Loop over coordinates
coordinates = np.concatenate((og_coordinates, np.zeros_like(og_coordinates)[:, 0][:, None]), 1)
merges = np.concatenate((merges, np.ones_like(merges)[:, 0][:, None]), 1)
coordinates = np.concatenate((coordinates, merges))
unique_z = np.unique(coordinates[:, -2])
print(unique_z)
np.save('unique_zs_for_merge', unique_z)

# Compute extents
mins = np.min(coordinates[:, :-1], axis=0)
maxs = np.max(coordinates[:, :-1], axis=0)  # Add extent to this
mins_vs = mins * config.shape  # (config.shape * np.array(path_extent))
maxs_vs = maxs * config.shape  # (config.shape * np.array(path_extent))
maxs_vs += config.shape * np.array(path_extent)
diffs = (maxs_vs - mins_vs)
xoff, yoff, zoff = np.array(path_extent) * config.shape  # [:2]

# Loop through x-axis
debug = -1  # 2
max_vox, count = 0, 0
slice_shape = np.concatenate((diffs[:-1], [384]))
# out_dir = '/media/data_cifs/connectomics/merge_data/'
out_dir = '/localscratch/merge/'
for zidx, z in tqdm(enumerate(unique_z), total=len(unique_z), desc="z-slice"):
    # Allocate tensor
    if zidx == 0:
        main = np.zeros(slice_shape, np.uint32)
    elif zidx > 24:
        out_dir = '/media/data/merge/'
    # else:
    #     main = merge

    # This plane
    z_sel_coors = coordinates[coordinates[:, 2] == z]

    if debug > 1:
        z_sel_coors = z_sel_coors[:debug]
        z_plus_sel_coors = z_plus_sel_coors[:debug]

    # Load this plane
    for sel_coor in tqdm(z_sel_coors, desc='Z: {}'.format(z)):
        vol = load_npz(sel_coor).transpose((2, 1, 0))
        adj_coor = (sel_coor[:-1] - mins) * config.shape
        try:
            main[
                adj_coor[0]: adj_coor[0] + xoff,
                adj_coor[1]: adj_coor[1] + yoff,
                :] = vol  # rfo(vol)[0]
        except Exception as e:
            import ipdb;ipdb.set_trace()
            raise RuntimeError(e)
            # print(sel_coor, (e))
    np.save(os.path.join(out_dir, 'plane_x{}'.format(z)), main)

