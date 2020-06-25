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


path_extent = [9, 9, 3]
config = Config()

# Get list of coordinates
coordinates = db.pull_membrane_coors()
coordinates = np.array([[r['x'], r['y'], r['z']] for r in coordinates if r['processed_segmentation']])

# Get list of merges
merges = db.pull_merge_membrane_coors()
merges = np.array([[r['x'], r['y'], r['z']] for r in merges if r['processed_segmentation']])

# Loop over coordinates
coordinates = np.concatenate((coordinates, np.zeros_like(coordinates)[:, 0][:, None]), 1)
merges = np.concatenate((merges, np.ones_like(merges)[:, 0][:, None]), 1)
coordinates = np.concatenate((coordinates, merges))
scoords = np.unique(coordinates, axis=0)
# scoords = np.concatenate((scoords, np.ones_like(scoords)[:, 0][:, None]), axis=-1)
used = np.ones_like(scoords)[:, 0]
rng =  np.arange(len(scoords))
idx = 0
coord = scoords[0]
while used.sum() > 0:
# for idx, coord in tqdm(enumerate(scoords), total=len(scoords)):
    vol = np.zeros((np.array(config.shape) * path_extent))
    num_coord = 0
    num_merge = 0
    if coord[3] == 0:
        try:
            for z in range(path_extent[0]):
                for y in range(path_extent[1]):
                    for x in range(path_extent[2]):
                        path = '/media/data_cifs/connectomics/mag1_segs/x{}/y{}/z{}/*.nii'.format(
                            pad_zeros(coord[0] + x, 4),
                            pad_zeros(coord[1] + y, 4),
                            pad_zeros(coord[2] + z, 4))
                        v = np.zeros(config.shape)
                        try:
                            path = glob(path)[0]
                            h = nib.load(path)
                            v = h.get_fdata()
                            h.uncache()
                        except:
                            pass
                        vol[
                            z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                            y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                            x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v  # nopep8
        except Exception as e:
            print('Failed COORD load: {}'.format(e))
            """
                    path = '/media/data_cifs/connectomics/mag1_merge_segs/x{}/y{}/z{}/*.nii'.format(
                        pad_zeros(coord[0] + x, 4),
                        pad_zeros(coord[1] + y, 4),
                        pad_zeros(coord[2] + z, 4))
                    try:
                        path = glob(path)[0]
                        # v = np.fromfile(
                        #     path, dtype='uint8').reshape(config.shape)
                        h = nib.load(path)
                        v = h.get_fdata()
                        h.uncache()
                        vol[
                            z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                            y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                            x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v  # nopep8
                        num_merge += 1
                    except Exception as e:
                        pass
            """
    else:
        """
        path = os.path.join('/media/data_cifs/connectomics/ding_segmentations_merge/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4)))
        vol = np.load(path)['segmentation']
        """
        try:
            for z in range(path_extent[0]):
                for y in range(path_extent[1]):
                    for x in range(path_extent[2]):
                        path = '/media/data_cifs/connectomics/mag1_merge_segs/x{}/y{}/z{}/*.nii'.format(
                            pad_zeros(coord[0] + x, 4),
                            pad_zeros(coord[1] + y, 4),
                            pad_zeros(coord[2] + z, 4))
                        v = np.zeros(config.shape)
                        try:
                            path = glob(path)[0]
                            h = nib.load(path)
                            v = h.get_fdata()
                            h.uncache()
                        except:
                            pass
                        vol[
                            z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                            y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                            x * config.shape[2]: x * config.shape[2] + config.shape[2]] = v  # nopep8
        except Exception as e:
            print('Failed MERGE load: {}'.format(e))
    if idx == 0:
        max_count = vol.max() + 1
    if idx == 0:
        pv = rfo(vol)[0]
    else:
        
        # tv = np.load(path)['segmentation']
        diff = coord - prev_coord
        direction = np.argmax(diff)
        tv = rfo(vol)[0] + max_seg
        import ipdb;ipdb.set_trace()
        print(diff)
        if diff[0] < 9 and diff[0] > 0 and diff[1] < 9 and diff[1] > 0 and diff[2] < 3 and diff[2] > 0:
            # import ipdb;ipdb.set_trace()
            pv = segmentation.drew_consensus(tv, pv)
        else:
            pv = tv
    prev_coord = coord
    prev_path = path

    # Find closest coord
    # scoords[idx, -2] = 0
    used[idx] = 0
    keep = used > 0
    import ipdb;ipdb.set_trace()
    keep = np.logical_and(keep, scoords[:, 2] == 36)
    sel = np.argmin(distance.cdist(coord[None, :3], scoords[keep, :3])[0])
    sel = rng[np.where(keep)[0][sel]]
    coord = scoords[sel, :]
    max_seg = pv.max() + 1
    idx = sel
    # idx += 1

"""
get_coord_paths = os.path.exists('coord_paths_pp.npy')
get_merge_paths = os.path.exists('merge_paths_pp.npy')
if get_coord_paths:
    coord_paths = glob('/media/data_cifs/connectomics/mag1_segs/x00**/y00**/z00**/*.nii')
    np.save('coord_paths_pp', coord_paths)
else:
    coord_paths = np.load('coord_paths_pp.npy')
if get_merge_paths:
    merge_paths = glob('/media/data_cifs/connectomics/mag1_merge_segs/x00**/y00**/z00**/*.nii')
    np.save('merge_paths_pp', coord_paths)
else:
    merge_paths = np.load('merge_paths_pp.npy')
scoords = np.unique(coordinates, axis=0)

# Create grid from 1st coordinate then greedily merge
x_max, y_max, z_max = scoords.max(0)
count = 0
for xs in range(x_max):
    for ys in range(y_max):
        for zs in range(z_max):
            path = os.path.join('/media/data_cifs/connectomics/mag1_segs/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(x, 4), pad_zeros(y, 4), pad_zeros(z, 4)))
            merge_path = os.path.join('/media/data_cifs/connectomics/mag1_merge_segs/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(x, 4), pad_zeros(y, 4), pad_zeros(z, 4)))
            if os.path.exists(path):
                if count == 0:
                    
"""


# Get first coordinate then find closest membrane and keep riding

# Save in webknossos format!
