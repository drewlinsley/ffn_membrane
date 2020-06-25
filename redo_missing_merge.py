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
to_add = []
while 1:  # used.sum() > 0:
# for idx, coord in tqdm(enumerate(scoords), total=len(scoords)):
    vol = np.zeros((np.array(config.shape) * path_extent))
    num_coord = 0
    num_merge = 0
    if idx != 0: 
        # tv = np.load(path)['segmentation']
        diff = coord - prev_coord
        new_coord = np.round(coord - diff * .5)
        print(diff, coord, new_coord)
        to_add += [new_coord.astype(int)]
        direction = np.argmax(diff)
    prev_coord = coord

    # Find closest coord
    # scoords[idx, -2] = 0
    used[idx] = 0
    keep = used > 0
    if keep.sum() == 0:
        break
    sel = np.argmin(distance.cdist(coord[None, :3], scoords[keep, :3])[0])
    sel = rng[np.where(keep)[0][sel]]
    coord = scoords[sel, :]
    idx = sel
    # idx += 1
# np.save('final_merges', to_add)

