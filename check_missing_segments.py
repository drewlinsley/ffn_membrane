import re
import os
import numpy as np
from db import db
from glob2 import glob
from utils.hybrid_utils import pad_zeros
from tqdm import tqdm


all_segment_coords = db.pull_membrane_coors()
if os.path.exists('progress/finished_segs.npy'):
    finished_segments = np.load('progress/finished_segs.npy')
else:
    finished_segments = glob(
        '/media/data_cifs/connectomics/ding_segmentations/**/**/**/v0/0/0/seg-0_0_0.npz')
    np.save('progress/finished_segs.npy', finished_segments)

finished_paths = []
for p in finished_segments:
    splits = p.split('/')
    x = int(re.split('[a-z].[0]*', splits[-7])[1])
    y = int(re.split('[a-z].[0]*', splits[-6])[1])
    z = int(re.split('[a-z].[0]*', splits[-5])[1])
    finished_paths += [[x, y, z]]

db_paths = []
for row in all_segment_coords:
    x = row['x']
    y = row['y']
    z = row['z']
    _id = row['_id']
    db_paths += [[x, y, z, _id]]


# Get overlaps
print('Getting overlaps')
db_paths = np.array(db_paths)
db_paths = np.sort(db_paths, axis=0)
finished_paths = np.array(finished_paths)
path_sort_idx = np.argsort(finished_paths[:, 0], axis=0)
finished_paths = finished_paths[path_sort_idx]
finished_segments = np.array(finished_segments)[path_sort_idx]
min_db = db_paths.sum(-1).min()
max_db = db_paths.sum(-1).max()
mask = np.logical_and(finished_paths.sum(-1) > min_db, finished_paths.sum(-1) < max_db)
finished_paths = finished_paths[mask]
finished_segments = finished_segments[mask]
if os.path.exists('progress/intermediate_segs.npy'):
    db_paths = np.load('progress/intermediate_segs.npy')
else:
    for p in tqdm(finished_paths, total=len(finished_paths)):
        for il, d in enumerate(db_paths):
            if np.all(p == d[:-1]):
                # pop the element so we dont compare anymore
                np.delete(db_paths, il, axis=0)
                break
    np.save('progress/intermediate_segs.npy', db_paths)

checks = []
finished = []
missing = []
coords = []
_, unique_path_idx = np.unique(db_paths[:, :3], axis=0, return_index=True)
unique_paths = db_paths[unique_path_idx]
for row in tqdm(unique_paths, total=len(unique_paths)):
    path = '/media/data_cifs/connectomics/ding_segmentations/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(  # noqa
        pad_zeros(row[0], 4),
        pad_zeros(row[1], 4),
        pad_zeros(row[2], 4))
    check = os.path.exists(path)
    if check:
        checks += [True]
        finished += [path]
    else:
        checks += [False]
        missing += [path]
        coords += [[row[0], row[1], row[2], row[3]]]
checks = np.array(checks)
finished = np.array(finished)
missing = np.array(missing)

# Make missing viable for processing again
fixes = []
for coord in coords:
    fix_row = {
        'x': coord[0],
        'y': coord[1],
        'z': coord[2],
        '_id': coord[3],
    }
    fixes += [fix_row]

db.process_segmentation_rows(fixes)

