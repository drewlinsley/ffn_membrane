import sys
import os
import numpy as np
import fastremap
import numba as nb
from glob import glob
from skimage.segmentation import relabel_sequential as rfo
from tqdm import tqdm
from shutil import copyfile
from scipy.sparse import csr_matrix, dok_matrix
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def find_and_update(Q, T, V):
    for i in prange(Q.shape[0]):
        if Q[i] == T:
            V[i] = T
    return V


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))


def consensus(segs, olds, segs_areas, segs_ids, olds_areas, olds_ids, max_id, plane_a, plane_b, min_size=1000):
    """Return consensus between seg and original."""
    # Add a stride
    plane_diff = plane_b - plane_a
    plane_diff = (plane_diff * 128) // 2
    if plane_diff > 256:
        return None
    olds = olds[..., plane_diff:]
    olds = np.concatenate((olds, np.zeros_like(olds)), -1)

    # Add some reshapes for speed
    seg_shape = segs.shape
    olds = olds.reshape(-1)
    segs = segs.reshape(-1)

    # Do some accounting
    segs_ids, segs_areas = fastremap.unique(segs, return_counts=True) # may be much faster than np.unique
    olds_ids, olds_areas = fastremap.unique(olds, return_counts=True) # may be much faster than np.uniqu

    # Get the docket on all labels
    both_ids = np.concatenate((segs_ids, olds_ids), 0)
    both_groups = np.concatenate((
      np.zeros_like(segs_areas),
      np.ones_like(olds_areas)))
    both_areas = np.concatenate((segs_areas, olds_areas), 0)
    X = np.stack((both_ids, both_groups, both_areas), -1)
    X = X[X[:, 0] != 0]
    X_idx = np.argsort(X[..., -1])[::-1]
    X = X[X_idx]
    X = np.concatenate((X, np.zeros_like(X)[:, 0].reshape(-1, 1)), -1)
    # Sparsify segs/olds for faster indexing
    # segs = segs.ravel()
    # olds = olds.ravel()
    # segs = compute_M(segs)
    # olds = compute_M(olds)
    # segs = dok_matrix(segs.reshape(1, -1))
    # olds = dok_matrix(olds.reshape(1, -1))
    # segs = segs.ravel()
    # olds = olds.ravel()
    # unique_segs, segs_inv = np.unique(segs, return_inverse=True)
    # unique_olds, olds_inv = np.unique(olds, return_inverse=True)

    # # Get mapping
    # uniq_segs = fastremap.unique(segs, return_counts=False) # may be much faster than np.unique
    # segs, segs_remapping = fastremap.renumber(segs, in_place=True) # relabel values from 1 and refit data type
    # uniq_olds = fastremap.unique(olds, return_counts=False) # may be much faster than np.unique
    # olds, olds_remapping = fastremap.renumber(olds, in_place=True) # relabel values from 1 and refit data type

    # # Build hashes
    # segs_hash = {}
    # olds_hash = {}

    # New volume + loop
    new_vol = np.zeros_like(segs)
    dtype = new_vol.dtype
    # del segs, olds
    # for idx, rw in tqdm(
    #       enumerate(X),
    #       total=len(X),
    #       desc='Updating ids for consensus'):
    for idx, rw in enumerate(X):
        # ids/new|old/areas/book
        is_duplicate = X[:, 0] == rw[0]
        dup_sum = np.sum(is_duplicate) > 1
        if not rw[-1]:  # If we haven't bookkeeped this location
          # Copy segment into new_vol
          if rw[1] == 1:
            # # This is an old
            # mask = olds == rw[0]
            # mask = fastremap.mask_except(olds, [rw[0]])
            # idx = np.where(olds == rw[0])
            find_and_update(Q=olds, V=new_vol, T=rw[0])
          else:
            # This is a new
            # mask = segs == rw[0]
            # mask = fastremap.mask_except(segs, [rw[0]])
            # idx = np.where(segs == rw[0])
            if dup_sum:
                find_and_update(Q=segs, V=new_vol, T=rw[0])
            else:
                find_and_update(Q=segs, V=new_vol, T=rw[0] + max_id)
          # if not dup_sum and rw[1] == 0:
            # Ensure that this is a globally unique id
            # Only apply for "new" segs
            # seg_id = max_id + rw[0]
            # mask = fastremap.remap(mask, {rw[0]: seg_id}, in_place=True, preserve_missing_labels=True)
          if dup_sum:
            X[is_duplicate, -1] = 1  # Skip the next time this idx comes up
          # new_vol = new_vol + mask
          # new_vol[idx] = seg_id
        else:
          # This is a duplicate so we skip
          pass
        X[idx, -1] = 1  # Not necessary, but we will bookkeep
    return new_vol.reshape(seg_shape)


def merge(fn_a, rp_a, fn_b, rp_b, out_b, plane_a, plane_b, merged):
    """Given volumes + region props run a merge."""
    if not os.path.exists('running_seg_count.npy'):
        max_id = 0
    else:
        max_id = np.load('running_seg_count.npy')
    if merged is None:
        segs_a = np.load(fn_a)  # np.load('/localscratch/plane_xplus9.npy')
    else:
        segs_a = merged
    rp_a = np.load(rp_a)
    segs_b = np.load(fn_b)
    rp_b = np.load(rp_b)
    areas_a = rp_a['segs_areas']
    ids_a = rp_a['segs_ids']
    areas_b = rp_b['segs_areas']
    ids_b = rp_b['segs_ids']

    # Run the merge
    merged = consensus(
        segs=segs_b,
        olds=segs_a,
        segs_areas=areas_b,
        segs_ids=ids_b,
        olds_areas=areas_a,
        olds_ids=ids_a,
        plane_a=plane_a,
        plane_b=plane_b,
        max_id=max_id)

    # Save the merged volume
    if merged is not None:
        np.save(out_b, merged)

        # Update the running seg count
        # _, max_id = fastremap.minmax(merged) + 1
        max_id = merged.max() + 1
        np.save('running_seg_count.npy', max_id)
        return merged
    else:
        print('Gap is too large')
        np.save(out_b, segs_b)


def run_merge(files, rps, outdir, plane):
    """Run merges between all successive pairs of files."""
    merged = None
    for target in tqdm(range(1, len(files)), total=len(files) - 1, desc='Merging'):
        fn_a = files[target - 1]
        rp_a = rps[target - 1]
        plane_a = plane[target - 1]
        fn_b = files[target]
        rp_b = rps[target]
        plane_b = plane[target]
        out_b = os.path.join(outdir, fn_b.split(os.path.sep)[-1].split('.')[0])
        print('Merging {} with {} into volume {}'.format(fn_b, fn_a, out_b))
        merged = merge(fn_a=fn_a, rp_a=rp_a, fn_b=fn_b, rp_b=rp_b, out_b=out_b, plane_a=plane_a, plane_b=plane_b, merged=merged)

if __name__ == '__main__':
    # fn_a, rp_a, fn_b, rp_b, out_b = sys.argv[1:]
    # print('Merging {} with {} into volume {}'.format(fn_b, fn_a, out_b))
    # merge(fn_a=fn_a, rp_a=rp_a, fn_b=fn_b, rp_b=rp_b, out_b=out_b)
    print('Trying single-python session processing.')
    print('If this fails uncomment the above and run through bash.')
    files_0 = glob('/localscratch/merge/*.npy')
    files_1 = glob('/media/data/merge/*.npy')
    files = files_0 + files_1
    files = np.array(files)
    plane = [int(f.split('x')[1].split('.')[0]) for f in files]
    files = files[np.argsort(plane)]
    # rps = np.array(glob('/localscratch/rps/*.npz'))
    rps = np.array(
        ['/localscratch/rps/plane_x{}.npy.npz'.format(p) for p in plane])
    outdir = '/gpfs/data/tserre/data/final_merge'
    copyfile(files[0], os.path.join(outdir, files[0].split(os.path.sep)[-1].split('.')[0]))
    run_merge(files=files, rps=rps, outdir=outdir, plane=plane)

