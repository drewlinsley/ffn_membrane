import os
import numpy as np
from db import db
from glob2 import glob
from utils.hybrid_utils import pad_zeros
from tqdm import tqdm


all_membrane_coords = db.pull_membrane_coors()
if os.path.exists('finished_mems.npy'):
    finished_membranes = np.load('finished_mems.npy')
else:
    finished_membranes = glob(
        '/media/data_cifs/connectomics/mag1_membranes_nii/**/**/**/*.nii')

checks = []
finished = []
missing = []
coords = []
for row in tqdm(all_membrane_coords, total=len(all_membrane_coords)):
    path = '/media/data_cifs/connectomics/mag1_membranes/x{}/y{}/z{}/*.npy'.format(  # noqa
        pad_zeros(row['x'], 4),
        pad_zeros(row['y'], 4),
        pad_zeros(row['z'], 4))
    check = glob(path)
    if len(check):
        checks += [True]
        finished += [path]
    else:
        checks += [False]
        missing += [path]
        coords += [[row['x'], row['y'], row['z'], row['_id']]]
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

print fixes
# db.process_rows(fixes)

