import gc
import sys
import time
import os
import gzip
import fastremap
import numpy as np
from db import db
from glob import glob
from ffn.inference import segmentation
from skimage.segmentation import relabel_sequential as rfo
from skimage import morphology
from tqdm import tqdm
from config import Config
import nibabel as nib
import pandas as pd
from scipy.spatial import distance
from utils.hybrid_utils import rdirs
from utils.hybrid_utils import pad_zeros
from utils.hybrid_utils import recursive_make_dir
from utils.hybrid_utils import make_dir
from skimage import measure
from numba import njit, jit, prange  # autojit
from joblib import Parallel, delayed
# from list_to_array import toarr
# import pyximport; pyximport.install()
from list_to_array_skel import toarr
import wknml
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import cc3d
from pympler import tracker
# from memory_profiler import profile


"""
# LRS directories
CHECK_DIRS = [
    "/users/dlinsley/scratch/connectomics_data/mag1_merge_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/connectomics/mag1_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/connectomics/mag1_merge_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data/mag1_merge_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch/mag1_merge_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_v0/mag1_merge_segs",
]
BU_DIRS = [
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data/ding_segmentations_merge",
    "/cifs/data/tserre/CLPS_Serre_Lab/connectomics/ding_segmentations_merge",
    "/users/dlinsley/scratch/connectomics_data/ding_segmentations",
    "/users/dlinsley/scratch/connectomics_data/ding_segmentations_merge",
    "
]
"""
CHECK_DIRS = [
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics_data/mag1_merge_segs",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics_data_v0/mag1_merge_segs",
    # "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics_data_v0/mag1_segs",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/mag1_merge_segs",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/mag1_segs",
    # "/gpfs/data/tserre/data/tmp_ding_segmentations/mag1_merge_segs/mag1_merge_segs",
    # "/gpfs/data/tserre/data/tmp_ding_segmentations/mag1_segs/mag1_merge_segs",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics/mag1_merge_segs",
    # "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics/mag1_segs",

    "/cifs/data/tserre/CLPS_Serre_Lab/connectomics/mag1_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/connectomics/mag1_merge_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data/mag1_merge_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data/mag1_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch/mag1_merge_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch/mag1_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_v0/mag1_merge_segs",
    # "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_v0/mag1_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v1/mag1_merge_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v1/mag1_segs"
]

BU_DIRS = [
    # "/gpfs/data/tserre/data/tmp_ding_segmentations/mag1_segs",
    "/cifs/data/tserre/CLPS_Serre_Lab/connectomics/ding_segmentations",
    "/cifs/data/tserre/CLPS_Serre_Lab/connectomics/ding_segmentations_merge",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics/ding_segmentations",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics/ding_segmentations_merge",
    "/gpfs/data/tserre/data/tmp_ding_segmentations/connectomics_data_v0/ding_segmentations_merge",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data/ding_segmentations",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data/ding_segmentations_merge",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch/ding_segmentations",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch/ding_segmentations_merge",
    # "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_v0/ding_segmentations",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_v0/ding_segmentations_merge",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v1/ding_segmentations_merge",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v1/ding_segmentations",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v2/connectomics_data/ding_segmentations",
    "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v2/connectomics_data/ding_segmentations_merge"
]


@njit(parallel=True, fastmath=True)
def direct_overlaps(main_margin, merge_margin, um):
    overlaps = np.zeros_like(merge_margin)
    for h in prange(main_margin.shape[0]):
        if main_margin[h] == um:
            overlaps[h] = merge_margin[h]
    return overlaps


def remap_loop(um, tc, main_margin, merge_margin):
    masked_plane = main_margin == um
    overlap = merge_margin[masked_plane]
    overlap = overlap[overlap != 0]
    overlap_check = len(overlap)  # overlap.sum()
    update = False
    transfer = False
    remap = []
    if not overlap_check:
        # In this case, there's a segment in main that overlaps with empty space in merge. Let's propagate main.
        transfer = um
        update = True
    else:
        # In this case, there's overlap between the merge and the main. Let's pass the main label to all merges that are touched (This used to be an argmax).
        uni_over, counts = fastremap.unique(overlap, return_counts=True)
        # uni_over = uni_over[uni_over > 0]
        for ui, uc in zip(uni_over, counts):
            remap.append([ui, um, uc])  # Append merge ids for the overlap
    return remap, transfer, update


# @profile
def get_remapping(main_margin, merge_margin, parallel, use_numba=False, merge_wiggle=0.8):
    """Determine where to merge.

    Potential problem is that this assumes both main/merge are fully segmented. If this is not the
    case, and we have a non-zero main segmentation but a zero merge segmentation, the zeros
    will overwrite the mains segmentation.

    Shitty but not sure what to do yet or if it's a real issue.
    """
    # Loop through the margin in main, to find per-segment overlaps with merge
    if not len(main_margin):
        return None
    if not len(merge_margin):
        return None
    unique_main, unique_main_counts = fastremap.unique(main_margin, return_counts=True)
    unique_main_mask = unique_main > 0
    unique_main = unique_main[unique_main_mask]
    unique_main_counts = unique_main_counts[unique_main_mask]
    if not len(unique_main):
        return [], merge_margin, [], False
    remap = []
    transfers = []
    update = False
    # For each segment in main, find the corresponding seg in margin. Transfer the id over, or transfer the bigger segment over (second needs to be experimental).
    # Use parallel on this loop
    info = parallel(delayed(remap_loop)(um, tc, main_margin, merge_margin) for um, tc in zip(unique_main, unique_main_counts))
    updates, transfers, remaps = [], [], []
    for r in info:
        updates.append(r[2])
        transfers.append(r[1])
        remaps.append(r[0])

    remaps = [x for x in remaps if len(x)]
    try:
        if len(remaps):
            remaps = np.concatenate(remaps, 0)
    except:
        import pdb;pdb.set_trace()
    # transfers = np.concatenate(transfers, 0)
    updates = np.max(updates)

    """
    for um, tc in zip(unique_main, unique_main_counts):  # Package this as a function
        masked_plane = main_margin == um  # fastremap.mask_except(h_plane, um)
        overlap = merge_margin[masked_plane]
        overlap = overlap[overlap != 0]
        overlap_check = len(overlap)  # overlap.sum()
        if not overlap_check:
            # In this case, there's a segment in main that overlaps with empty space in merge. Let's propagate main.
            transfers.append(um)
            update = True
        else:
            # In this case, there's overlap between the merge and the main. Let's pass the main label to all merges that are touched (This used to be an argmax).
            uni_over, counts = fastremap.unique(overlap, return_counts=True)
            # uni_over = uni_over[uni_over > 0]
            for ui, uc in zip(uni_over, counts):
                remap.append([ui, um, uc])  # Append merge ids for the overlap
    """
    if 0:  # len(transfers):
        # Transfer all over in a single C++ optimized call
        merge_margin += fastremap.mask_except(main_margin, transfers)
    return remaps, merge_margin, transfers, updates


# @autojit(parallel=True, fastmath=True)
@jit(parallel=True, fastmath=True)
def npad_zeros(x, total):
    """Pad x with zeros to total digits."""
    if not isinstance(x, str):
        x = str(x)
    total = total - len(x)
    for idx in range(total):
        x = '0' + x
    return x


# @autojit(parallel=True, fastmath=True)
@jit(parallel=True, fastmath=True)
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

# @autojit(parallel=True, fastmath=True)
# @jit(parallel=True, fastmath=True)
def convert_save_cubes(coords, data, cifs_path, mins, config, xoff, yoff, path_extent):
    """All coords come from the same z-slice. Save these as npys to cifs."""
    for idx in prange(len(coords)):
    # for seed in coords:
        seed = coords[idx]
        adj_coor = (seed[:-1] - mins) * config.shape
        segments = data[
            adj_coor[0]: adj_coor[0] + xoff,
            adj_coor[1]: adj_coor[1] + yoff]
        for x in range(path_extent[0]):
            for y in range(path_extent[1]):
                for z in range(path_extent[2]):
                    path = cifs_path % (
                        npad_zeros(seed[0] + x, 4),
                        npad_zeros(seed[1] + y, 4),
                        npad_zeros(seed[2] + z, 4),
                        npad_zeros(seed[0] + x, 4),
                        npad_zeros(seed[1] + y, 4),
                        npad_zeros(seed[2] + z, 4))
                    seg = segments[
                        x * config.shape[0]: x * config.shape[0] + config.shape[0],
                        y * config.shape[1]: y * config.shape[1] + config.shape[1],
                        z * config.shape[2]: z * config.shape[2] + config.shape[2]]
                    nrecursive_make_dir(path)
                    np.save(path, seg)


# @njit(parallel=False, fastmath=True, nogil=True)
def build_vol_v0(vol, vols, coords, shape):
    """Insert d into vol."""
    for idx in range(len(vols)):
        x, y, z = coords[idx]
        vol[
            x * shape[0]: x * shape[0] + shape[0],  # nopep8
            y * shape[1]: y * shape[1] + shape[1],  # nopep8
            z * shape[2]: z * shape[2] + shape[2]] = vols[idx]
    return vol


# @jit('u4[:, :, :](u4[:, :, :], u4[:, :, :, :], u4[:], u4[:])', parallel=True, fastmath=True, nogil=True)
# @njit(parallel=True, fastmath=True, nogil=True)
# @jit(parallel=False, fastmath=True, nogil=True)
def build_vol_v1(vol, vols, coords, shape):
    """Insert d into vol."""
    vol_shape = vol.shape
    # for idx in prange(len(vols)):
    for idx in range(len(vols)):
        x, y, z = coords[idx]
        x_slice = np.arange(x * shape[0], x * shape[0] + shape[0])
        y_slice = np.arange(y * shape[1], y * shape[1] + shape[1])
        z_slice = np.arange(z * shape[2], z * shape[2] + shape[2])
        ix_ = np.ix_(x_slice, y_slice, z_slice)
        # vol[
        #     x * shape[0]: x * shape[0] + shape[0],  # nopep8
        #     y * shape[1]: y * shape[1] + shape[1],  # nopep8
        #     z * shape[2]: z * shape[2] + shape[2]] = vols[idx]
        vol[ix_] = vols[idx]
    return vol


def check_backup(sel_coor):
    """Check backup paths."""
    for d in BU_DIRS:
        path = os.path.join(d, "x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz".format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4)))
        if os.path.exists(path):
            data = np.load(path)
            v = data["segmentation"].transpose(2, 1, 0)
            del data.f
            data.close()
            return v, True
    return False, False


# @profile
def load_npz(sel_coor, shape, dtype, path_extent, parallel=None, verbose=False, debug=False):
    """First try loading from main segmentations, then the merges.

    Later, add loading for nii as the fallback."""
    # empty = np.zeros(path_extent)
    coords, idxs = [], []
    empty = False
    for x in range(path_extent[0]):
        for y in range(path_extent[1]):
            for z in range(path_extent[2]):
                coords.append((sel_coor[0] + x, sel_coor[1] + y, sel_coor[2] + z))
                idxs.append((x, y, z))
    if verbose:
        print('Loading niftis')
        elapsed = time.time()
    if debug:
        vols = []
        for coord, idx in zip(coords, idxs):
            vols.append(execute_load(coord, idx, config, dtype, shape=shape))
        proc_per_vol = True
    else:
        # First try ding_segmentations
        vols, _, success = execute_ding_load(sel_coor, idxs, config, shape=shape)

        # Then load cubes independently
        proc_per_vol = False
        if not success:
            # vols = parallel(delayed(execute_load)(coord, idx, config, dtype, shape=config.shape) for coord, idx in zip(coords, idxs))
            vols = []
            for coord, idx in zip(coords, idxs):
                vols.append(execute_load(coord, idx, config, dtype, shape=config.shape))
            proc_per_vol = True
    if proc_per_vol:
        proc_vols, idxs, success = [], [], []
        for v, i, s in vols:
            proc_vols.append(v)
            idxs.append(i)
            success.append(s)
        # success = np.asarray(success)
        # print("Failed on {}/{} loads from {}, {}, {}.".format(len(success) - success.sum(), len(success), sel_coor[0], sel_coor[1], sel_coor[2]))
        # np.savez("fails/fail_{}_{}_{}".format(sel_coor[0], sel_coor[1], sel_coor[2]), coords=coords, sel_coor=sel_coor)
        # np.save("fails/fail_{}_{}_{}".format(sel_coor[0], sel_coor[1], sel_coor[2]), sel_coor)

        vols = proc_vols
        if verbose:
            print('Finished: {}'.format(time.time() - elapsed))
        if verbose:
            print('Converting to array')
            elapsed = time.time()
        vols = toarr(vols)
        if verbose:
            print('Finished: {}'.format(time.time() - elapsed))
        if verbose:
            print('Building vol')
        # vols = np.array(vols)
        vol = np.zeros((np.array(shape)), dtype=dtype)  # shape * path_extent
        # vol = None
        # empty = True
        try:
            # import time
            # start = time.time()
            vol = build_vol_v0(vol=vol, vols=vols, coords=idxs, shape=config.shape)
            # end = time.time()
            # print("TIMING: {}".format(end-start))
            # os._exit(1)
        except:
            import pdb;pdb.set_trace()
        del vols, proc_vols
        return vol, empty
    if verbose:
        print('Finished: {}'.format(time.time() - elapsed))
    return vols, empty


# @profile
def execute_load(sel_coor, idx, config, dtype=np.uint32, shape=(128, 128, 128), dc_path="/cifs/data/tserre/CLPS_Serre_Lab/connectomics"):
    """Load a single nii."""
    for di in CHECK_DIRS:
        path = os.path.join(di, 'x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.nii'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4), pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4)))
        if os.path.exists(path):
            zp = nib.load(path)
            h = zp.dataobj
            v = h.get_unscaled()
            # v = zp.get_data()  # get_unscaled()
            zp.uncache()
            # del zp, v, h
            del zp, h
            v = np.asarray(v.transpose((2, 1, 0)).astype(dtype))
            # print("*Found* {}".format(path))
            return v, idx, True  # sel_coor
    # print("Failed to find {}".format(path))
    return np.zeros(shape, dtype), idx, False


def check_location(sel_coor, idx):
    """Check if it's npz or npy."""
    for di in BU_DIRS:
        path = os.path.join(di, 'x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4)))
        if os.path.exists(path):
            return 2  # , idx
    for di in CHECK_DIRS:
        path = os.path.join(di, 'x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.nii'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4), pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4)))
        if os.path.exists(path):
            return 1  # , idx
    return 0  # , idx


# @profile
def execute_ding_load(sel_coor, idx, config, dtype=np.uint32, shape=(128, 128, 128), dc_path="/cifs/data/tserre/CLPS_Serre_Lab/connectomics"):
    """Load a single nii."""
    for di in BU_DIRS:
        path = os.path.join(di, 'x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4)))
        if os.path.exists(path):
            z = np.load(path)
            v = z["segmentation"].transpose(2, 1, 0).astype(dtype)
            # Make ssure v.shape matches what we're looking for
            del z.f
            z.close()
            if np.all(np.asarray(v.shape) == shape):
                # v = np.asarray(v.transpose((2, 1, 0)).astype(dtype))
                return v, idx, True  # sel_coor
    # print("\nFailed to find {}".format(path))
    return np.zeros(shape, dtype), idx, False


def skeleton_merge(main, slice_skels, max_vox, z, mins_vs):
    """Merge segments in main according to slice_skels."""
    true_z = z * 128
    change_dict = {}
    fails = []
    for k, skel in slice_skels.items():
        changeid = None
        for sid, node in enumerate(skel):
            node = node[0]
            node = [int(x) for x in node]
            adjusted_z = node[2] - true_z
            adjusted_y = node[1] - mins_vs[1]
            adjusted_x = node[0] - mins_vs[0]
            try:
                main_shape = main.shape
                if adjusted_x < main_shape[0] and adjusted_y < main_shape[1] and adjusted_z < main_shape[2]:
                    tochange = main[adjusted_x, adjusted_y, adjusted_z]
                    if tochange != 0:
                        if changeid is None:
                            changeid = tochange
                        """
                        if sid == 0:  # Get a "global" skeleton id for all nodes in this slice
                            if tochange > 0:
                                changeid = tochange
                            else:
                                changeid = max_vox + 1
                        """
                        change_dict[tochange] = changeid
            except Exception as e:
                print("Failed on skel {}, {}, {}".format(adjusted_x, adjusted_y, adjusted_z))
                print(e)
                fails.append([node[0], node[1], node[2], adjusted_x, adjusted_y, adjusted_z])
    if len(change_dict):
        print("Remapping")
        main = fastremap.remap(main, change_dict, preserve_missing_labels=True, in_place=True)
    return main, max_vox, fails


# @profile
def remap(vol, remap_type, min_vol_size, in_place, max_vox, connectivity=6, disable_max_vox=True):  # Moved max_vox to main loop
    """Run a remapping and iterate by max_vox."""
    if remap_type == "seung":
        # vol = cc3d.connected_components(vol, connectivity=connectivity)
        if min_vol_size > 0:
            vol = morphology.remove_small_objects(vol, min_size=min_vol_size, in_place=in_place).astype(dtype)
        vol, _ = fastremap.renumber(vol, in_place=in_place, preserve_zero=True)  # .astype(np.uint32)
        vol = vol.astype(dtype)
    elif remap_type == "seung_no_small":
        # vol = cc3d.connected_components(vol, connectivity=connectivity)
        vol, _ = fastremap.renumber(vol, in_place=in_place, preserve_zero=True)  # .astype(np.uint32)
        vol = vol.astype(dtype)
    elif remap_type == "skimage":
        vol = rfo(vol)[0]
    else:
        raise NotImplementedError(remap_type)
    # _, max_vox = fastremap.minmax(vol)
    if not disable_max_vox:
        it_max_vox = vol.max()
        zeros = (vol > 0).astype(dtype)
        vol += max_vox  # Update to max_vox
        vol *= zeros
        max_vox = max_vox + it_max_vox + 1  # Increment
    return vol, max_vox


def get_margins(margin_idx, margin_end, margin_offset, main_margin_offset, edge_offset, candidate_diffs, div, adj_coor, idx):
    margin_start = np.abs(div)
    if np.any(margin_idx):
        # There is a candidate!!
        sel_idx = candidate_diffs[margin_idx][0]
        # directed_sel_idx = sel_idx[idx]
        margin_end = margin_start + margin_offset
        edge_start = adj_coor[idx] + margin_start - main_margin_offset
        merge = True
    else:
        edge_start = adj_coor[idx] + margin_start - 1
        merge = False
    edge_start = np.maximum(edge_start, 0)
    edge_end = edge_start + edge_offset
    return merge, edge_start, edge_end, margin_start, margin_end


# @profile
def get_merge_coords(
        candidate_diffs,
        main_margin_offset,
        edge_offset,
        margin_start,
        margin_end,
        margin_offset,
        vs,
        adj_coor,
        main,
        vol,
        parallel,
        div=10):
    """Find overlapping coordinates in top/bottom/left/right planes."""
    midpoints = vs // 2

    ## Top
    top_margin_idx = candidate_diffs[:, 0] < midpoints[0]
    bottom_margin_idx = candidate_diffs[:, 0] > midpoints[0]
    left_margin_idx = candidate_diffs[:, 1] < midpoints[1]
    right_margin_idx = candidate_diffs[:, 1] > midpoints[1]

    top_merge, top_edge_start, top_edge_end, top_margin_start, top_margin_end = get_margins(
        top_margin_idx, margin_end, margin_offset, main_margin_offset, edge_offset, candidate_diffs, div, adj_coor, idx=0)
    bottom_merge, bottom_edge_start, bottom_edge_end, bottom_margin_start, bottom_margin_end = get_margins(
        bottom_margin_idx, margin_end, margin_offset, main_margin_offset, edge_offset, candidate_diffs, div, adj_coor, idx=0)
    left_merge, left_edge_start, left_edge_end, left_margin_start, left_margin_end = get_margins(
        left_margin_idx, margin_end, margin_offset, main_margin_offset, edge_offset, candidate_diffs, div, adj_coor, idx=1)
    right_merge, right_edge_start, right_edge_end, right_margin_start, right_margin_end = get_margins(
        right_margin_idx, margin_end, margin_offset, main_margin_offset, edge_offset, candidate_diffs, div, adj_coor, idx=1)

    if top_merge:
        main_top_face = main[  # Should this be expanded to a matrix instead of a vector?? so noisy
            top_edge_start: top_edge_end,
            adj_coor[1]: adj_coor[1] + vs[1],
            :]
    if bottom_merge:
        main_bottom_face = main[
            bottom_edge_start: bottom_edge_end,
            adj_coor[1]: adj_coor[1] + vs[1],
            :]

    if left_merge:
        main_left_face = main[
            adj_coor[0]: adj_coor[0] + vs[0],
            left_edge_start: left_edge_end,
            :]
    if right_merge:
        main_right_face = main[
            adj_coor[0]: adj_coor[0] + vs[0],
            right_edge_start: right_edge_end,
            :]
    remap_top, remap_left, remap_right, remap_bottom = [], [], [], []
    bottom_trans, top_trans, right_trans, left_trans = [], [], [], []

    # Get remapping for each face
    if top_merge:
        merge_top_face = vol[top_margin_start: top_margin_end, :, :]
        remap_top, merge_top_face, top_trans, update = get_remapping(
            main_margin=main_top_face,
            merge_margin=merge_top_face, parallel=parallel)
    if left_merge:
        merge_left_face = vol[:, left_margin_start: left_margin_end, :]
        remap_left, merge_left_face, left_trans, update = get_remapping(
            main_margin=main_left_face,
            merge_margin=merge_left_face, parallel=parallel)
    if right_merge:
        merge_right_face = vol[:, -right_margin_end + 1: -right_margin_start + 1]
        remap_right, merge_right_face, right_trans, update = get_remapping(
            main_margin=main_right_face,
            merge_margin=merge_right_face, parallel=parallel)
    if bottom_merge:
        merge_bottom_face = vol[-bottom_margin_end + 1: -bottom_margin_start + 1]
        remap_bottom, merge_bottom_face, bottom_trans, update = get_remapping(
            main_margin=main_bottom_face,
            merge_margin=merge_bottom_face, parallel=parallel)
    return np.array(remap_top + remap_left + remap_right + remap_bottom), bottom_margin_start, top_margin_start, right_margin_start, left_margin_start, bottom_trans, top_trans, right_trans, left_trans


def par_trans(tr, sel_vol, sel_main):
    it_trans_remap = {}
    to_map = np.unique(sel_vol[tr == sel_main])
    to_map = to_map[to_map != 0]
    for m in to_map:
        it_trans_remap[m] = tr
    return it_trans_remap


# @profile
def process_merge(main, vol, sel_coor, mins, config, path_extent, parallel, max_vox=None, margin_start=0, margin_end=1, test=0.50, prev=None, plane_coors=None, verbose=False, main_margin_offset=1, edge_offset=1, margin_offset=1, min_vol_size=256):  # margin_end=1 main_margin_offset=1,edge_offset=1
    """Handle merge volumes.
    TODO: Just load section that could be used for merging two existing mains.
    """
    # Merge function:
    # i. Take a local neighborhood of voxels around a merge
    # ii. reduce across orthogonal dimensions
    # iii. Compute correlations per segment
    # iv. if correlation > threshold, pass the label from main -> merge

    # # For horizontal
    # Insert all merges that will not colide with mains (add these to the plane_coors list as mains)
    # Find the collisions between merges + mains. Double check all 4 sides of the merge volume. Apply merge to the sides that have verified collisions.

    if not len(vol):
        return main, max_vox
    # Vol size
    vs = config.shape * path_extent

    # Get centroid
    adj_coor = (sel_coor[:-1] - mins) * config.shape
    center = adj_coor + (vs // 2)
    center = center[:-1]
    if prev is None:  # direction == 'horizontal':
        # Signed distance between corners of adj_coor and other planes 
        adj_planes = ((plane_coors[:, :-1] - mins) * config.shape)

        # Drop T/B/L/R planes into main. See if there's anything there. Brute force.
        # adj_coor is TL corner of the merge we're looking at.
        # We also wan these planes to be a bit offset from boundaries because of sparse segs there
        margin = 25
        top_plane_hs = [adj_coor[0] + margin, adj_coor[0] + margin + 1]
        top_plane_ws = [adj_coor[1] + margin, adj_coor[1] + vs[1] - margin]
        bottom_plane_hs = [adj_coor[0] + vs[0] - margin - 1, adj_coor[0] + vs[0] - margin]
        bottom_plane_ws = [adj_coor[1] + margin, adj_coor[1] + vs[1] - margin]
        left_plane_hs = [adj_coor[0] + margin, adj_coor[0] + vs[0] - margin]
        left_plane_ws = [adj_coor[1] + margin, adj_coor[1] + margin + 1]
        right_plane_hs = [adj_coor[0] + margin, adj_coor[0] + vs[0] - margin]
        right_plane_ws = [adj_coor[1] + vs[1] - margin - 1, adj_coor[1] + vs[1] - margin]
        top_plane = main[top_plane_hs[0]: top_plane_hs[1], top_plane_ws[0]: top_plane_ws[1]]
        bottom_plane = main[bottom_plane_hs[0]: bottom_plane_hs[1], bottom_plane_ws[0]: bottom_plane_ws[1]]
        left_plane = main[left_plane_hs[0]: left_plane_hs[1], left_plane_ws[0]: left_plane_ws[1]]
        right_plane = main[right_plane_hs[0]: right_plane_hs[1], right_plane_ws[0]: right_plane_ws[1]]
        top_check, bottom_check, left_check, right_check = top_plane.sum(), bottom_plane.sum(), left_plane.sum(), right_plane.sum()
        trimmed = False  # Try leaving this on as default. The boundary stuff sucks.
        if np.any([top_check, bottom_check, left_check, right_check]):
            # Trim vol to match the margin-adjusted size
            trimmed = True
        vol = vol[margin: -margin, margin: -margin]  # Was previously a contingency in above if

        bottom_trans, top_trans, right_trans, left_trans = [], [], [], []
        all_remaps = []
        if top_check:  # Merge here
            merge_top_face = vol[0, :]
            remap_top, merge_top_face, top_trans, update = get_remapping(
                main_margin=top_plane.squeeze(),
                merge_margin=merge_top_face, parallel=parallel)
            all_remaps.append(remap_top)
        if left_check: 
            merge_left_face = vol[:, 0]
            remap_left, merge_left_face, left_trans, update = get_remapping(
                main_margin=left_plane.squeeze(),
                merge_margin=merge_left_face, parallel=parallel)
            all_remaps.append(remap_left)
        if right_check:
            merge_right_face = vol[:, -1]
            remap_right, merge_right_face, right_trans, update = get_remapping(
                main_margin=right_plane.squeeze(),
                merge_margin=merge_right_face, parallel=parallel)
            all_remaps.append(remap_right)
        if bottom_check:
            merge_bottom_face = vol[-1]
            remap_bottom, merge_bottom_face, bottom_trans, update = get_remapping(
                main_margin=bottom_plane.squeeze(),
                merge_margin=merge_bottom_face, parallel=parallel)
            all_remaps.append(remap_bottom)

        all_remaps = [x for x in all_remaps if len(x)]
        if len(all_remaps):
            all_remaps = np.concatenate(all_remaps, 0)
        # all_trans = np.concatenate([bottom_trans, top_trans, right_trans, left_trans])

        # Get sizes and originals for every remap. Sort these for the final remap
        if len(all_remaps):
            remap_idx = np.argsort(all_remaps[:, -1])[::-1]  # Sort by sizes
            all_remaps = all_remaps[remap_idx]
            unique_remaps = fastremap.unique(all_remaps[:, 0], return_counts=False) 
            fixed_remaps = {}
            for ur in unique_remaps:  # , rc in zip(unique_remaps, remap_counts):
                mask = all_remaps[:, 0] == ur
                fixed_remaps[ur] = all_remaps[mask][0][1]  # Change all to the biggest
            vol = fastremap.remap(vol, fixed_remaps, preserve_missing_labels=True, in_place=True)
            if verbose:
                print('Finished: {}'.format(time.time() - elapsed))

            # Insert vol into main
            # if trimmed:
            main[top_plane_hs[0]: bottom_plane_hs[1], left_plane_ws[0]: right_plane_ws[1]] = vol
            # else:
            #     main[adj_coor[0]: adj_coor[0] + vs[1], adj_coor[1]: adj_coor[1] + vs[1]] = vol
        else:
            # if remap_labels:
            #     # Only add to non-zeros
            #     vol += (vol != 0).astype(vol.dtype) * max_vox
            #     mv, mxv = fastremap.minmax(vol)
            #     max_vox += mxv + 1
            # adj_coor = (sel_coor[:-1] - mins) * config.shape

            """
            main_shape = main.shape
            main_x = np.arange(adj_coor[0], adj_coor[0] + xoff)
            main_y = np.arange(adj_coor[1], adj_coor[1] + yoff)
            main_z = np.arange(main_shape[-1])
            main_ix_ = np.ix_(main_x, main_y, main_z)
            main[main_ix_] = vol
            """
            # main[
            #     adj_coor[0]: adj_coor[0] + xoff,
            #     adj_coor[1]: adj_coor[1] + yoff,
            #     :] = vol  # rfo(vol)[0]
            main[
                adj_coor[0] + margin: adj_coor[0] + xoff - margin,
                adj_coor[1] + margin: adj_coor[1] + yoff - margin,
                :] = vol  # rfo(vol)[0]
        # del vol
        return main, max_vox
    elif prev is not None:  #  == 'bottom-up':
        # Get distance in z
        fos = 32  # Fixed offset between the planes
        dz = sel_coor[2] - plane_coors[:, 2][0] 
        adj_dz = int(dz * config.shape[-1])
        # curr_bottom_face = main[..., fos]
        # prev_top_face = prev[..., fos + adj_dz]
        curr_bottom_face = main[
            adj_coor[0]: adj_coor[0] + vs[0],
            adj_coor[1]: adj_coor[1] + vs[1],
            fos]  # -1
        prev_vol = prev[
            adj_coor[0]: adj_coor[0] + vs[0],
            adj_coor[1]: adj_coor[1] + vs[1]]
        prev_top_face = prev_vol[..., fos + adj_dz] #  -adj_dz]
        # import pdb;pdb.set_trace()
        # from matplotlib import pyplot as plt;plt.subplot(121);plt.imshow(rfo(curr_bottom_face)[0]);plt.subplot(122);plt.imshow(rfo(prev_top_face)[0]);plt.show()
        if not prev_top_face.sum():
            # Prev doesn't have any voxels, pass the original
            return main, {}
        if verbose:
            print('Running bottom-up remap')
            elapsed = time.time()
        all_remaps, _, transfers, update = get_remapping(
            main_margin=prev_top_face,
            merge_margin=curr_bottom_face,  # mapping from prev -> main
            parallel=parallel,
            use_numba=False)

        # Get sizes and originals for every remap. Sort these for the final remap
        all_remaps = np.array(all_remaps)
        fixed_remaps = {}
        if len(all_remaps):
            remap_idx = np.argsort(all_remaps[:, -1])[::-1]
            all_remaps = all_remaps[remap_idx]
            unique_remaps, remap_counts = fastremap.unique(all_remaps[:, 0], return_counts=True)
            for ur, rc in zip(unique_remaps, remap_counts):
                if ur != 0:
                    mask = all_remaps[:, 0] == ur
                    fixed_remaps[ur] = all_remaps[mask][0][1]  # Change all to the biggest
            if verbose:
                print('Finished: {}'.format(time.time() - elapsed))
        # del curr_bottom_face

        # Also overwrite the sparse bottom-facing edge in main with the segs in prev
        # main[..., :fos] = prev[..., adj_dz: fos + adj_dz]

        return main, fixed_remaps  # main, max_vox
    else:
        raise RuntimeError('Something fucked up.')


def add_to_main(sel_coor, config, path_extent, dtype, remap_labels, remap_type, min_vol_size, in_place, max_vox, mins, main, trim=0, vol=None):
    # Load mains in this plane
    if vol is None:
        vol, empty = load_npz(sel_coor, shape=config.shape * path_extent, dtype=dtype, path_extent=path_extent)  # , parallel=parallel)  # .transpose((2, 1, 0))
    else:
        empty = False
    if not empty:
        if remap_labels:
            vol, it_max_vox = remap(vol=vol, remap_type=remap_type, min_vol_size=min_vol_size, in_place=in_place, max_vox=max_vox)
            max_vox = it_max_vox
        adj_coor = (sel_coor[:-1] - mins) * config.shape
        try:
            if trim:
                main[
                    adj_coor[0] + trim: adj_coor[0] + xoff - trim,
                    adj_coor[1] + trim: adj_coor[1] + yoff - trim,
                    :] = vol[trim:-trim, trim:-trim]  # rfo(vol)[0]
            else:
                main[
                    adj_coor[0]: adj_coor[0] + xoff,
                    adj_coor[1]: adj_coor[1] + yoff,
                    :] = vol  # rfo(vol)[0]
        except:
            print(vol.sum())
            import pdb;pdb.set_trace()


def batch_load(sel_coor, config, path_extent, dtype, remap_labels, remap_type, min_vol_size, in_place, max_vox, mins):
    # Load mains in this plane
    vol, empty = load_npz(sel_coor, shape=config.shape * path_extent, dtype=dtype, path_extent=path_extent)  # , parallel=parallel)  # .transpose((2, 1, 0))
    if not empty:
        if remap_labels:
            vol, max_vox = remap(vol=vol, remap_type=remap_type, min_vol_size=min_vol_size, in_place=in_place, max_vox=max_vox)
        # adj_coor = (sel_coor[:-1] - mins) * config.shape
    return [vol, max_vox, sel_coor]


def increment_max_vox(vol, max_vox):
    it_max_vox = vol.max()
    zeros = (vol > 0).astype(dtype)
    vol += max_vox  # Update to max_vox
    vol *= zeros
    max_vox = max_vox + it_max_vox + 1  # Increment
    return vol, max_vox


path_extent = [9, 9, 3]
glob_debug = False
save_cubes = False
load_processed = True
remap_labels = True
remap_type = "seung_no_small"  # "seung"
in_place = True
merge_skeletons = False
z_max = 384
bu_margin = 2
magic_merge_number_max = 7  # 10
magic_merge_number_min = 4  # 6
# magic_merge_number = 3
dtype = np.uint32
config = Config()
min_vol_size = 256  # 512  # 2048  # Making this bigger to cut down on total IDs  # 1024

# Load skeleton data
skeletons = "All_Skels_to_Stitch_Segs.nml"
if merge_skeletons:
    with open(skeletons, "rb") as f:
        nml = wknml.parse_nml(skeletons)

    # Extract skeletons
    skel_dict = {}
    z_dict = {}  # For a given z-slice, list of dicts, mapping positions for individual skel-ids
    count = 0
    for skel in range(len(nml[1])):
        skel_dict[skel] = []
        for idx in range(len(nml[1][skel][3])):
            position = nml[1][skel][3][idx][1][::-1]  # Reverse for merging
            skel_dict[skel].append([position])
            proc_position = int(position[-1] // 128)
            if proc_position not in z_dict.keys():
                z_dict[proc_position] = {}
            if skel not in z_dict[proc_position].keys():
                z_dict[proc_position][skel] = []
            z_dict[proc_position][skel].append([position])
            count += 1
    print("Found {} nodes for merging.".format(count))
    print(z_dict.keys())

# Get list of coordinates
og_coordinates = db.pull_main_seg_coors()
if glob_debug:
    raise NotImplementedError("Conflict with the new files.")
    new_og_coordinates = []
    for r in og_coordinates:
        sel_coor = [r['x'], r['y'], r['z']]
        check = glob(os.path.join('/media/data_cifs/connectomics/ding_segmentations/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4))))
        if len(check):
            new_og_coordinates.append(sel_coor)
    og_coordinates = np.array(new_og_coordinates)
else:
    og_coordinates = np.array([[r['x'], r['y'], r['z']] for r in og_coordinates if r['processed_segmentation']])

# Get list of merges
merges = db.pull_merge_seg_coors()
if glob_debug:
    new_merges = []
    for r in merges:
        sel_coor = [r['x'], r['y'], r['z']]
        check = glob(os.path.join('/media/data_cifs/connectomics/ding_segmentations_merge/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4))))
        if len(check):
            new_merges.append(sel_coor)
        else:
            print(sel_coor)
    merges = np.array(new_merges)
else:
    merges = np.array([[r['x'], r['y'], r['z']] for r in merges if r['processed_segmentation']])

# Loop over coordinates
coordinates = np.concatenate((og_coordinates, np.zeros_like(og_coordinates)[:, 0][:, None]), 1)
merges = np.concatenate((merges, np.ones_like(merges)[:, 0][:, None]), 1)
coordinates = np.concatenate((coordinates, merges))
unique_z = np.unique(coordinates[:, -2])
print(unique_z)
# np.save('unique_zs_for_merge', unique_z)

# Compute extents
path_extent = np.array(path_extent)
mins = np.min(coordinates[:, :-1], axis=0)
maxs = np.max(coordinates[:, :-1], axis=0)  # Add extent to this
mins_vs = mins * config.shape  # (config.shape * np.array(path_extent))
maxs_vs = maxs * config.shape  # (config.shape * np.array(path_extent))
maxs_vs += config.shape * path_extent  # np.array(path_extent)
diffs = (maxs_vs - mins_vs)
xoff, yoff, zoff = path_extent * config.shape  # [:2]

# Check that directories exist before running
for bud in BU_DIRS:
    if not os.path.isdir(bud):
        print("{} does not exist -- fix this!!!\r".format(bud))
for bud in CHECK_DIRS:
    if not os.path.isdir(bud):
        print("{} does not exist -- fix this!!!\r".format(bud))

# Loop through x-axis
max_vox, count, prev = 1, 0, None
slice_shape = np.concatenate((diffs[:-1], [z_max]))
cifs_path = None  # Force a fail for this. TODO: build a clean API '/media/data_cifs/connectomics/merge_data/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.npy'
out_dir = '/gpfs/data/tserre/data/final_merge/'  # /localscratch/merge/'
# out_dir = '/users/dlinsley/scratch/final_merge/'
make_dir(out_dir)
all_failed_skeletons = []
pbar = tqdm(enumerate(unique_z), total=len(unique_z), desc="Z-slice main clock")
# tr = tracker.SummaryTracker()
with Parallel(max_nbytes=None, n_jobs=128, require='sharedmem') as parallel:  # n_jobs=32
    for zidx, z in pbar:  # tqdm(enumerate(unique_z), total=len(unique_z), desc="Z-slice main clock"):
        # Allocate tensor
        main = np.zeros(slice_shape, dtype)

        # This plane
        z_sel_coors = coordinates[coordinates[:, 2] == z]
        # sort_idx = np.argsort(z_sel_coors, -1)[::-1]
        # z_sel_coors = z_sel_coors[sort_idx]
        z_sel_coors = np.unique(z_sel_coors, axis=0)

        # Split into merge + mains
        z_sel_coors_main = z_sel_coors[z_sel_coors[..., -1] == 0]
        z_sel_coors_merge = z_sel_coors[z_sel_coors[..., -1] == 1]

        # Check each to determine if we have the npz or just npys
        pre_all_coords = np.concatenate((z_sel_coors_main, z_sel_coors_merge), 0)
        npz_or_npy = parallel(delayed(check_location)(sel_coor, idx) for idx, sel_coor in enumerate(pre_all_coords))
        npz_or_npy = np.asarray(npz_or_npy)
        all_coords = pre_all_coords[npz_or_npy == 2]
        # NEXT STEP IS INTRODUCE THE npz_or_npy==1 AS NEEDED

        """
        # Sort into list of non-colliding mains and colliding merges here.
        collisions = []
        dm = distance.squareform(distance.pdist(all_coords[:, :2], "cityblock"))
        dm = dm + np.eye(dm.shape[0]) * 1000
        new_main = {}
        # new_merge = []
        main_dist = magic_merge_number_max  # np.sqrt(2. * magic_merge_number_max ** 2)
        for rid, rw in enumerate(dm):
            check = rw < main_dist  # Any neighbors we want to set to exclude
            exclude = rw == 1000  # What is already excluded
            if check.sum() and not np.all(exclude):
                # Put rid into the mains, add its subthreshold neighbors to the merges
                argids = np.argsort(rw)
                sortvals = np.sort(rw)
                argids = argids[sortvals < main_dist]  # These are close volumes and potential merges
                dm[check] = 1000  # Make these untouchable
                dm[:, check] = 1000  # Make these untouchable
                dm[rid, :] = 1000
                dm[:, rid] = 1000  # Dont include this coordinate again
                # new_merge.append(all_coords[argids])
                new_main[rid] = all_coords[rid]  # append(all_coords[rid])
        z_sel_coors_merge = all_coords[np.asarray(list(set(range(len(dm))) - set(list(new_main.keys()))))]
        z_sel_coors_main = np.stack([v for k, v in new_main.items()], 0)
        z_sel_coors_main = np.unique(z_sel_coors_main, axis=0)
        z_sel_coors_merge = np.unique(z_sel_coors_merge, axis=0)
        print("Using {}/{} npzs for main.".format(len(z_sel_coors_main), len(z_sel_coors_merge)))
        # from matplotlib import pyplot as plt
        # plt.plot(z_sel_coors_main[:,1], z_sel_coors_main[:,0], "bo")
        # plt.plot(z_sel_coors_merge[:,1], z_sel_coors_merge[:,0], "ro")
        # plt.show()

        if magic_merge_number_max > 0:
            # Now, if you want, select merges so that there's at least a distance of X between the merges
            main_merge = np.concatenate((z_sel_coors_main[:, :2], z_sel_coors_merge[:, :2]), 0)
            idxs = np.ones_like(main_merge)[:, 0].reshape(-1, 1)
            idxs[:len(z_sel_coors_main)] = 0
            main_merge = np.concatenate((main_merge, idxs), 1)
            dm = distance.squareform(distance.pdist(main_merge[:, :2], "cityblock"))
            new_merge = []
            for rid, rw in enumerate(dm):
                if main_merge[rid, -1] == 1:  # Only look at coords if they are merges
                    check = rw < magic_merge_number_min # This is now our threshold for finding good merges
                    # main_dist_check = distance.cdist(z_sel_coors_merge[rid, :2][None], z_sel_coors_main[:, :2], "cityblock") <= 3  # Make sure we aren't overlapping with a main for some reason
                    exclude = rw == 1000
                    dm[rid, :] = 1000
                    dm[:, rid] = 1000  # Dont include this coordinate again
                    added = False
                    if not np.all(exclude):  #  and not np.any(main_dist_check):  # Add to the hopper if no reason to exclude
                        # Add to the merge list
                        new_merge.append(main_merge[rid])  # z_sel_coors_merge[rid]) 
                        added = True
                    if check.sum() and added:  # IF we've added remove neighbors from contention
                        dm[check] = 1000  # Make these untouchable
                        dm[:, check] = 1000  # Make these untouchable
            print("Excluded {}/{}".format(len(z_sel_coors_merge) - len(new_merge), len(z_sel_coors_merge)))
            z_sel_coors_merge = np.stack(new_merge, 0)
        print("Finished loading")
        # tr.print_diff()

        # Run one more search with the non-npz segs
        # z_sel_coors_merge = np.concatenate([z_sel_coors_merge, pre_all_coords[npz_or_npy != 2]], 0)
        if 0:  # (npz_or_npy != 2).sum():  # magic_merge_number_max > 0 and (npz_or_npy != 2).sum():
            all_coords = np.concatenate((main_merge, pre_all_coords[npz_or_npy != 2]), 0)
            idxs = np.ones_like(all_coords)[:, 0].reshape(-1, 1)
            idxs[:len(main_merge)] = 0
            # Now, if you want, select merges so that there's at least a distance of X between the merges
            dm = distance.squareform(distance.pdist(all_coords[:, :2], "cityblock"))
            new_merge = []
            for rid, rw in enumerate(dm):
                if all_coords[rid, -1] == 1:
                    check = rw < magic_merge_number_min # This is now our threshold for finding good merges
                    exclude = rw == 1000
                    dm[rid, :] = 1000
                    dm[:, rid] = 1000  # Dont include this coordinate again
                    added = False
                    if not np.all(exclude):  #  and not np.any(main_dist_check):  # Add to the hopper if no reason to exclude
                        # Add to the merge list
                        new_merge.append(all_coords[rid])
                        added = True
                    if check.sum() and added:  # IF we've added remove neighbors from contention
                        dm[check] = 1000  # Make these untouchable
                        dm[:, check] = 1000  # Make these untouchable
                print("Excluded {}/{}".format((npz_or_npy != 2).sum() - len(new_merge), (npz_or_npy != 2).sum()))
            subpar_z_sel_coors_merge = np.stack(new_merge, 0)
            z_sel_coors_merge = np.concatenate((z_sel_coors_merge, subpar_z_sel_coors_merge), 0)
        print("Finished loading")

        # Add a dummy column to z_sel_coors_merge
        z_sel_coors_merge = np.concatenate((z_sel_coors_merge, np.zeros_like(z_sel_coors_merge)[:, 0][:, None]), 1)
        """
        dm = squareform(pdist(pre_all_coords[:, :2], "cityblock"))
        dm = dm + np.eye(dm.shape[0]) * 1000
        keeps = []
        for rid, rw in enumerate(dm):
            check = rw == 1000
            if len(rw) - check.sum() > 0:
                keeps.append(rid)
                mask = rw < 5
                dm[mask] = 1000
                dm[:, mask] = 1000
                dm[rid] = 1000
                dm[:, rid] = 1000
        z_sel_coors_main = pre_all_coords[np.asarray(keeps)]
        z_sel_coors_merge = []
        for zm in pre_all_coords:
            if not np.any((zm == z_sel_coors_main).sum(-1) == 4):
                z_sel_coors_merge.append(zm)
        z_sel_coors_merge = np.asarray(z_sel_coors_merge)
        z_sel_coors_main[:, -1] = 0
        z_sel_coors_merge[:, -1] = 1
        # if zidx > -1:
        #     from matplotlib import pyplot as plt
        #     plt.subplot(121)
        #     plt.plot(z_sel_coors_main[:,1], z_sel_coors_main[:,0], "bo")
        #     plt.plot(z_sel_coors_merge[:,1], z_sel_coors_merge[:,0], "ro")
        #     plt.subplot(122)
        #     plt.plot(pre_all_coords[:,1], pre_all_coords[:,0], "go")
        #     plt.show()

        # Allow for fast loading for debugging
        skip_processing = False
        if load_processed:
            if os.path.exists(os.path.join(out_dir, 'plane_z{}.npy'.format(z))):
                print("Slice already processed. Loading...")
                prev = np.load(os.path.join(out_dir, 'plane_z{}.npy'.format(z)))
                skip_processing = True

        # print('Wherever you dont have mains, see if you can insert a merge (non conflicts with mains), and promote it to a main')
        if not skip_processing:
            # TODO: get ALL coordinates in this plane, then split them into main/merge sets
            start = time.time()
            if len(z_sel_coors_main):
                # # Old way was to add directly into main.
                # parallel(delayed(add_to_main)(sel_coor, config, path_extent, dtype, remap_labels, remap_type, min_vol_size, in_place, max_vox, mins, main) for sel_coor in tqdm(z_sel_coors_main, desc='Z (mains): {}'.format(z)))
                # Now lets parload the data, but sequentially iterate the maxvox, then parload into main
                vols_vox = parallel(delayed(batch_load)(sel_coor, config, path_extent, dtype, remap_labels, remap_type, min_vol_size, in_place, max_vox, mins) for sel_coor in tqdm(z_sel_coors_main, desc='Z (loading mains): {}'.format(z)))
                main_vols, max_voxes, new_sel_coors = [], [], []

                # Update max_vox in this loop
                for vm in vols_vox:
                    it_vol = vm[0]
                    it_vol, max_vox = increment_max_vox(vol=it_vol, max_vox=max_vox)
                    main_vols.append(it_vol), max_voxes.append(vm[1]), new_sel_coors.append(vm[2])
                parallel(delayed(add_to_main)(sel_coor, config, path_extent, dtype, False, remap_type, min_vol_size, in_place, max_vox, mins, main, vol=main_vol) for main_vol, sel_coor in tqdm(zip(main_vols, new_sel_coors), desc='Z (Adding mains to main): {}'.format(z)))
            else:
                raise RuntimeError("Should not have 0 mains.")
                # parallel(delayed(add_to_main)(sel_coor, config, path_extent, dtype, remap_labels, remap_type, min_vol_size, in_place, max_vox, mins, main) for sel_coor in tqdm(z_sel_coors_merge, desc='Z (mains): {}'.format(z)))
                # z_sel_coors_main = np.copy(z_sel_coors_merge)
                # z_sel_coors_merge = []
            del main_vols, it_vol
            gc.collect()
            end = time.time()
            print("Main parloop load time: {}".format(end-start))

            # Par load all the merges
            start = time.time()
            vols_vox = parallel(delayed(batch_load)(sel_coor, config, path_extent, dtype, remap_labels, remap_type, min_vol_size, in_place, max_vox, mins) for sel_coor in tqdm(z_sel_coors_merge, desc='Z (loading merges): {}'.format(z)))
            merge_vols, max_voxes, new_sel_coors = [], [], []
            for vm in vols_vox:
                it_vol = vm[0]
                it_vol, max_vox = increment_max_vox(vol=it_vol, max_vox=max_vox)
                merge_vols.append(it_vol), max_voxes.append(vm[1]), new_sel_coors.append(vm[2])
            gc.collect()
            end = time.time()
            print("Merge parloop load time: {}".format(end-start))
            # max_vox = max(max_voxes)

            # Perform horizontal merge if there's admixed main/merge
            # for sel_coor in tqdm(z_sel_coors_merge, desc='H Merging: {}'.format(z)):
            for idx, sel_coor in tqdm(enumerate(new_sel_coors), desc='H Merging: {}'.format(z)):
                main, max_vox = process_merge(
                    main=main,
                    sel_coor=sel_coor,
                    mins=mins,
                    vol=merge_vols[idx],
                    parallel=parallel,
                    config=config,
                    max_vox=max_vox,
                    plane_coors=z_sel_coors_main,  # np.copy(z_sel_coors_main),
                    min_vol_size=min_vol_size,
                    path_extent=path_extent)
                # print(max_vox)
                pbar.set_description("Z-slice main clock (current max is {})".format(max_vox))
                if max_vox == 1:
                    import pdb;pdb.set_trace()
                z_sel_coors_main = np.concatenate((z_sel_coors_main, [sel_coor]), 0)
            print("Finished h-merge")
            # tr.print_diff()

            # Perform skeleton merge
            if merge_skeletons:
                if z in z_dict.keys():
                    slice_skels = z_dict[z]
                    main, max_vox, failed_skeletons = skeleton_merge(main, slice_skels, max_vox, z, mins_vs)
                    all_failed_skeletons.append(failed_skeletons)

            # Perform bottom-up merge
            if prev is not None:
                margin = config.shape[-1] * (unique_z[zidx] - unique_z[zidx - 1])
                if margin < z_max:
                    """
                    main, all_remaps = process_merge(
                        vol=[1],  # Just pass a dummy for BU
                        main=main,
                        sel_coor=sel_coor,
                        margin_start=margin,
                        margin_end=margin + bu_margin,
                        parallel=parallel,
                        mins=mins,
                        config=config,
                        plane_coors=prev_coords,
                        path_extent=path_extent,
                        min_vol_size=min_vol_size,
                        prev=prev)
                    """
                    all_remaps = {}
                    for sel_coor in tqdm(z_sel_coors_main, desc='BU Merging: {}'.format(z)):
                        main, remaps = process_merge(
                            vol=[1],  # Just pass a dummy for BU
                            main=main,
                            sel_coor=sel_coor,
                            margin_start=margin,
                            margin_end=margin + bu_margin,
                            parallel=parallel,
                            mins=mins,
                            config=config,
                            plane_coors=prev_coords,
                            path_extent=path_extent,
                            min_vol_size=min_vol_size,
                            prev=prev)
                        if len(remaps):
                            all_remaps.update(remaps)
                    if len(all_remaps):
                        # Perform a single remapping
                        print('Performing BU remapping of {} ids'.format(len(all_remaps)))
                        main = fastremap.remap(main, all_remaps, preserve_missing_labels=True)

            print("Finished b-merge")
            # tr.print_diff()

            # Save the current main and retain info for the next slice
            if save_cubes:
                raise RuntimeError("Depreciated save_cubes function.")
                convert_save_cubes(
                    data=main,
                    coords=z_sel_coors_main,
                    cifs_path=cifs_path,
                    mins=mins,
                    config=config,
                    path_extent=path_extent,
                    xoff=xoff,
                    yoff=yoff)
            else:
                np.save(os.path.join(out_dir, 'plane_z{}'.format(z)), main)

            if prev is not None:
                del prev
            # prev = np.copy(main)
            prev = main
        else:
            # Slice is already processed, skip this one.
            if not len(z_sel_coors_main):
                z_sel_coors_main = np.copy(z_sel_coors_merge)
            # _, mv = fastremap.minmax(prev)
            mv = prev.max() + 1
            max_vox += mv
            print('Skipping plane {}. Current max: {}'.format(z, max_vox))
        unique_mains = np.unique(np.concatenate((z_sel_coors_main[:, :-1], z_sel_coors_merge[:, :-1]), 0), axis=0)  # ADDED TO ENSURE BU-prop AT ALL LOCATIONS
        z_sel_coors_main = np.concatenate((unique_mains, np.zeros((len(unique_mains), 1))), 1)  # ADDED TO ENSURE BU-prop AT ALL LOCATIONS
        z_sel_coors_main = z_sel_coors_main.astype(int)
        # prev_coords = np.copy(z_sel_coors_main)
        prev_coords = z_sel_coors_main

        # Now save this layer's coordinates
        np.savez("merge_coordinates/{}".format(z), main=z_sel_coors_main, merge=z_sel_coors_merge)

        try:
            del vols_vox
        except:
            pass
        try:
            del merge_vol
        except:
            pass
        try:
            del merge_vols, it_vol
        except:
            pass
        gc.collect()
    ###### ENDING EARLY
    if merge_skeletons:
        np.save("all_failed_skeletons", all_failed_skeletons)

