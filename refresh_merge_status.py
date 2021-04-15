import os
import numpy as np
from db import db
from joblib import Parallel, delayed
from utils.hybrid_utils import pad_zeros
import joblib
from tqdm import tqdm
import contextlib


CHECK_DIRS = [
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics_data/mag1_merge_segs",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics_data_v0/mag1_merge_segs",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics_data_v0/mag1_segs",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/mag1_merge_segs",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/mag1_segs",
    # "/gpfs/data/tserre/data/tmp_ding_segmentations/mag1_merge_segs/mag1_merge_segs",
    # "/gpfs/data/tserre/data/tmp_ding_segmentations/mag1_segs/mag1_merge_segs",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics/mag1_merge_segs"
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics/mag1_segs",

    "/cifs/data/tserre/CLPS_Serre_Lab/connectomics/mag1_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/connectomics/mag1_merge_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data/mag1_merge_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data/mag1_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch/mag1_merge_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch/mag1_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_v0/mag1_merge_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_v0/mag1_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v1/mag1_merge_segs"
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v1/mag1_segs"


]
BU_DIRS = [
    "/gpfs/data/tserre/data/tmp_ding_segmentations/mag1_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/connectomics/ding_segmentations",
    "/cifs/data/tserre/CLPS_Serre_Lab/connectomics/ding_segmentations_merge",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics/ding_segmentations",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics/ding_segmentations_merge",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics_data_v0/ding_segmentations_merge",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data/ding_segmentations",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data/ding_segmentations_merge",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch/ding_segmentations",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch/ding_segmentations_merge",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_v0/ding_segmentations",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_v0/ding_segmentations_merge",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v1/ding_segmentations_merge"
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v1/ding_segmentations"
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v2/ding_segmentations"
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v2/ding_segmentations_merge"
]


def check_path(sel_coor, CHECK_DIRS, BU_DIRS, path_extent=[9, 9, 3]):
    for di in BU_DIRS:
        path = os.path.join(di, 'x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4)))
        if os.path.exists(path):
            return 2
    coords = []
    for x in range(path_extent[0]):
        for y in range(path_extent[1]):
            for z in range(path_extent[2]):
                coords.append((sel_coor[0] + x, sel_coor[1] + y, sel_coor[2] + z))
    for di in CHECK_DIRS:
        checks = np.zeros(len(coords), dtype=bool)
        for idx, sel_coor in enumerate(coords):
            path = os.path.join(di, 'x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.nii'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4), pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4)))
            if os.path.exists(path):
                checks[idx] = True
        if np.all(checks):
            return 1
    return 0


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()    


compute = False
merges = db.pull_all_merge_membrane_coors()
merges = np.asarray([[d['x'], d['y'], d['z']] for d in merges])
print(len(merges))
if compute:
    with Parallel(backend='loky', n_jobs=-16, max_nbytes=None) as parallel:  # n_jobs=32
        with tqdm_joblib(tqdm(desc="My calculation",total=len(merges))) as progress_bar:
            results = Parallel()(delayed(check_path)(coord, CHECK_DIRS, BU_DIRS) for coord in merges)
    np.save("coord_status", results)
else:
    results = np.load("coord_status.npy")

# Now update all the 1s and 2s to finished and leave everything else
update_idx = np.asarray(results) > 0
updates = merges[update_idx]
d = []
for idx, coords in enumerate(updates):  # existing_merges):
    coords = [int(x) for x in coords]
    d.append({"x": coords[0], "y": coords[1], "z": coords[2], "run_number": None})
print(len(d))
db.finish_coordinate_merge(d)

