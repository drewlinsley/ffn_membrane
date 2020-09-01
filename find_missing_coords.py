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
from numba import njit, jit, autojit, prange
from joblib import Parallel, delayed
from list_to_array import toarr


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
    "/media/data_cifs/connectomics/ding_segmentations"
]

@njit(parallel=True, fastmath=True)
def direct_overlaps(main_margin, merge_margin, um):
    overlaps = np.zeros_like(merge_margin)
    for h in prange(main_margin.shape[0]):
        if main_margin[h] == um:
            overlaps[h] = merge_margin[h]
    return overlaps


def get_remapping(main_margin, merge_margin, use_numba=False, merge_wiggle=0.5):
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
        return [], merge_margin, False
    remap = []
    transfers = []
    update = False
    # For each segment in main, find the corresponding seg in margin. Transfer the id over, or transfer the bigger segment over (second needs to be experimental).
    for um, tc in zip(unique_main, unique_main_counts):  # Package this as a function
        if use_numba:
            overlap = direct_overlaps(main_margin.reshape(-1), merge_margin.reshape(-1), um)
        else:
            masked_plane = main_margin == um  # fastremap.mask_except(h_plane, um)
            overlap = merge_margin[masked_plane]
        overlap = overlap[overlap != 0]
        overlap_check = overlap.sum()
        if not overlap_check:
            # merge_margin += masked_plane.astype(merge_margin.dtype) * um
            transfers.append(um)
            update = True
        # If overlap is a large enough proportion, propogate the main-id to merge
        # prop = float(overlap.sum()) / float(masked_plane.sum())
        # if prop >= test:
        else:
            uni_over, counts = fastremap.unique(overlap, return_counts=True)
            # uni_over = uni_over[uni_over > 0]
            cidx = np.argmax(counts)  # Just transfer largest -> largest
            # for ui, uc in zip(uni_over, counts):
            #     remap.append([ui, um, uc])  # Append merge ids for the overlap
            ui, uc = uni_over[cidx], counts[cidx]
            # if float(uc) > (tc * merge_wiggle):  # Only remap if the merge segment is bigger than the main! This controls boundary artifacts
            remap.append([ui, um, uc])
    if 0:  # len(transfers):
        # Transfer all over in a single C++ optimized call
        merge_margin += fastremap.mask_except(main_margin, transfers)
    return remap, merge_margin, update


@autojit(parallel=True, fastmath=True)
def npad_zeros(x, total):
    """Pad x with zeros to total digits."""
    if not isinstance(x, str):
        x = str(x)
    total = total - len(x)
    for idx in range(total):
        x = '0' + x
    return x


@autojit(parallel=True, fastmath=True)
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


@autojit(parallel=True, fastmath=True)
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


# @jit('u4[:, :, :](u4[:, :, :], u4[:, :, :, :], u4[:], u4[:])', parallel=True, fastmath=True, nogil=True)
@njit(parallel=True, fastmath=True, nogil=True)
def build_vol(vol, vols, coords, shape):
    """Insert d into vol."""
    for idx in prange(len(vols)):
        x, y, z = coords[idx]
        vol[
            x * shape[0]: x * shape[0] + shape[0],  # nopep8
            y * shape[1]: y * shape[1] + shape[1],  # nopep8
            z * shape[2]: z * shape[2] + shape[2]] = vols[idx]
    return vol


def check_backup(sel_coor):
    """Check backup paths."""
    for d in BU_DIRS:
        path = os.path.join(d, "x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz".format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4)))
        if os.path.exists(path):
            import ipdb;ipdb.set_trace()
            data = np.load(path)
            return data["segmentation"].transpose(2, 1, 0), True
    return False, False


def load_npz(sel_coor, shape, dtype, path_extent, parallel, verbose=False, debug=False):
    """First try loading from main segmentations, then the merges.

    Later, add loading for nii as the fallback."""
    coords, idxs = [], []
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
            vols.append(execute_load(coord, idx, config))
    else:
        vols = parallel(delayed(execute_load)(coord, idx, config) for coord, idx in zip(coords, idxs))
    proc_vols, idxs, success = [], [], []
    for v, i, s in vols:
        idxs.append(i)
        success.append(s)
    success = np.asarray(success)
    proc_per_vol = True
    if not np.all(success):
        # Try ding_seg backups
        vol, bu_success = check_backup(sel_coor)
        if bu_success:
            sel_coor = None
    else:
        sel_coor = None
    return sel_coor


def execute_load(sel_coor, idx, config, dtype=np.uint32, shape=(128, 128, 128), dc_path="/cifs/data/tserre/CLPS_Serre_Lab/connectomics"):
    """Load a single nii."""
    for di in CHECK_DIRS:
        path = os.path.join(di, 'x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.nii'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4), pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4)))
        if os.path.exists(path):
            return True, idx, True  # sel_coor
    # print("Failed to find {}".format(path))
    return np.zeros(shape, dtype), idx, False


def process_merge(main, sel_coor, mins, config, path_extent, parallel, max_vox=None, margin_start=0, margin_end=1, test=0.50, prev=None, plane_coors=None, verbose=False, main_margin_offset=1):
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

    # Vol size
    vs = config.shape * path_extent

    # Upper corner
    adj_coor = (sel_coor[:-1] - mins) * config.shape
            
    # Get midpoints for checking horizontal merges (diamond)
    lx = ((sel_coor[:-1] - mins) * config.shape)[0]
    mx = lx + vs[0] // 2
    rx = lx + vs[0]
    ty = ((sel_coor[:-1] - mins) * config.shape)[1]
    my = ty + vs[1] // 2  # ((config.shape * np.array(path_extent))[1] // 2)
    by = ty + vs[1]  # ((config.shape * np.array(path_extent))[1])
    uz = ((sel_coor[:-1] - mins) * config.shape)[1]
    mz = uz + vs[2] // 2  # ((config.shape * np.array(path_extent))[2] // 2)
    lz = uz + vs[2]  # ((config.shape * np.array(path_extent))[2])

    if prev is None:  # direction == 'horizontal':
        vol = load_npz(sel_coor, shape=config.shape, dtype=dtype, path_extent=path_extent, parallel=parallel)  # .transpose((2, 1, 0))
        if remap_labels:
            # vol, remapping = fastremap.renumber(vol, in_place=in_place) 
            vol = rfo(vol)[0]
        mp_coor_mx_ty_mz = (mx, ty, mz)  # (sel_coor[:-1] - mins) * config.shape
        mp_coor_lx_my_mz = (lx, my, mz)  # (sel_coor[:-1] - mins) * config.shape
        mp_coor_mx_by_mz = (mx, by, mz)  # (sel_coor[:-1] - mins) * config.shape
        mp_coor_rx_my_mz = (rx, my, mz)  # (sel_coor[:-1] - mins) * config.shape

        # Figure out how much empty space we are covering with this merge
        sel_coor_tr = np.copy(adj_coor)
        sel_coor_tr[1] += vs[1]  # (config.shape * path_extent)[1]
        # sel_coor_br = np.copy(sel_coor[:-1] - mins)
        # sel_coor_br[0] += 1
        # sel_coor_br[1] += 1
        # sel_coor_br *= config.shape

        # Signed distance between corners of adj_coor and other planes 
        adj_planes = ((plane_coors[:, :-1] - mins) * config.shape)
        nns_tl = adj_coor - adj_planes  # Just X/Y
        nns_tr = sel_coor_tr - adj_planes  # Just X/Y
        # nns_br = sel_coor_br[:-1] - plane_coors[:, :-1]  # Just X/Y
        # nns_br = sel_coor_br - (plane_coors[:, :-1] * config.shape)  # Just X/Y

        # Find the closest nearest neightbors by l1
        c_tl = np.abs(nns_tl)  # .sum(-1))
        c_tr = np.abs(nns_tr)  # .sum(-1))
        # c_bl = np.abs(sel_coor[:-1], nns_bl)  # .sum(-1))
        # c_br = np.abs(nns_br)  # .sum(-1))

        # Aggregate across columns and sort to find nearest IDX
        candidates = np.logical_and(np.logical_and(c_tl[:, 0] < vs[0], c_tl[:, 1] < vs[1]), c_tl.sum(-1) > 0)
        candidate_coords = adj_planes[candidates]
        
        if len(candidate_coords):
            # idx_tl = np.argsort(c_tl.sum(-1))
            # idx_tr = np.argsort(c_tr.sum(-1))
            # idx_bl = np.argsort(c_bl.sum(-1))
            # idx_br = np.argsort(c_br.sum(-1))

            # Check each face
            # merge_top_face = vol[:margin, :, :]
            # merge_left_face = vol[:, :margin, :]
            # merge_right_face = vol[:, -margin:, :]
            # merge_bottom_face = vol[-margin:, :, :]
            # main_top_face = main[
            #     # adj_coor[0] - margin: adj_coor[0] + margin,
            #     adj_coor[0]: adj_coor[0] + margin,
            #     adj_coor[1]: adj_coor[1] + vs[1],
            #     :]
            # main_left_face = main[
            #     adj_coor[0]: adj_coor[0] + vs[0],
            #     # adj_coor[1] - margin: adj_coor[1] + margin,
            #     adj_coor[1]: adj_coor[1] + margin,
            #     :]
            # main_right_face = main[
            #     adj_coor[0]: adj_coor[0] + vs[0],
            #     # adj_coor[1] + vs[1] - margin: adj_coor[1] + vs[1] + margin,
            #     adj_coor[1] + vs[1] - margin: adj_coor[1] + vs[1],
            #     :]
            # main_bottom_face = main[
            #     adj_coor[0] + vs[0] - margin: adj_coor[0] + vs[0],
            #     adj_coor[1]: adj_coor[1] + vs[1],
            #     :]
            run_top, run_left, run_right, run_bottom = True, True, True, True
            candidate_diffs = candidate_coords - adj_coor
            ## Merge overlaps with main. Find which 
            ## mains we overlap with. Push the merge to the top of the stack.
            ## Use the full merge - a margin (only on main-facing edges).
            ## Do the merge at the margin point. This means at the main-facing edge,
            ## there is a certain amount of overlap. Crop the merge so that it's less-
            ## half-the distance between the overlap.

            ## For each segment to be merged, measure the size then only swap labels if
            ## Vol size is > than the main size.

            ## Do another merge which is from -1 in the main direction to the merge.
            ## Maybe just make this the merge? That will control the boundary effects

            if 0:  # adj_coor[0] - margin_start < 0:
                run_top = False
            else:
                # Figure out the margin
                top_margin_idx = np.logical_and(np.logical_and(candidate_diffs[:, 0] == 0, candidate_diffs[:, 1] == 0), candidate_diffs[:, 2] == 0)
                if np.any(top_margin_idx):
                    top_margin_start = np.abs(candidate_diffs[top_margin_idx][0][0] // 10)
                    top_margin_end = top_margin_start + 1
                else:
                    top_margin_start = margin_start
                    top_margin_end = margin_end
                top_edge_start = adj_coor[0] + top_margin_start - main_margin_offset
                top_edge_start = np.maximum(top_edge_start, 0)
                top_edge_end = top_edge_start + 1  # top_margin_end
                main_top_face = main[
                    top_edge_start: top_edge_end,
                    # adj_coor[0] + top_margin_start: adj_coor[0] + top_margin_end,
                    adj_coor[1]: adj_coor[1] + vs[1],
                    :]
            if 0:  # adj_coor[1] - margin < 0:
                run_left = False
            else:
                left_margin_idx = np.logical_and(np.logical_and(candidate_diffs[:, 0] == 0, candidate_diffs[:, 1] < 0), candidate_diffs[:, 2] == 0)
                if np.any(left_margin_idx):
                    left_margin_start = np.abs(candidate_diffs[left_margin_idx][0][1] // 10)
                    left_margin_end = left_margin_start + 1
                else:
                    left_margin_start = margin_start
                    left_margin_end = margin_end
                left_edge_start = adj_coor[1] + left_margin_start - main_margin_offset
                left_edge_start = np.maximum(left_edge_start, 0)
                left_edge_end = left_edge_start + 1  # left_margin_end
                main_left_face = main[
                    adj_coor[0]: adj_coor[0] + vs[0],
                    left_edge_start: left_edge_end,
                    # adj_coor[1] + left_margin_start: adj_coor[1] + left_margin_end,
                    :]
            if 0:  # adj_coor[1] + vs[1] + margin > main.shape[1]:
                run_right = False
            else:
                right_margin_idx = np.logical_and(np.logical_and(candidate_diffs[:, 0] == 0, candidate_diffs[:, 1] > 0), candidate_diffs[:, 2] == 0)
                if np.any(right_margin_idx):
                    right_margin_start = np.abs(candidate_diffs[right_margin_idx][0][1] // 10)
                    right_margin_end = right_margin_start + 1
                else:
                    right_margin_start = margin_start
                    right_margin_end = margin_end
                right_edge_start = adj_coor[1] + vs[1] - right_margin_end + main_margin_offset # + left_margin_start
                right_edge_start = np.minimum(right_edge_start, main.shape[1] - 1)
                right_edge_end = right_edge_start + 1  # right_margin_start
                main_right_face = main[
                    adj_coor[0]: adj_coor[0] + vs[0],
                    right_edge_start: right_edge_end,
                    # adj_coor[1] + vs[1] - right_margin_end: adj_coor[1] + vs[1] - right_margin_start,  # end > start
                    :]
            if 0:  # adj_coor[0] + vs[0] + margin > main.shape[0]:
                run_bottom = False
            else:
                bottom_margin_idx = np.logical_and(np.logical_and(candidate_diffs[:, 0] > 0, candidate_diffs[:, 1] == 0), candidate_diffs[:, 2] == 0)
                if np.any(bottom_margin_idx):
                    bottom_margin_start = np.abs(candidate_diffs[bottom_margin_idx][0][0] // 10)
                    bottom_margin_end = bottom_margin_start + 1
                else:
                    bottom_margin_start = margin_start
                    bottom_margin_end = margin_end
                bottom_edge_start = adj_coor[0] + vs[0] - bottom_margin_end + main_margin_offset  # + left_margin_start
                bottom_edge_start = np.minimum(bottom_edge_start, main.shape[0] - 1)
                bottom_edge_end = bottom_edge_start + 1  # bottom_margin_start
                main_bottom_face = main[
                    # adj_coor[0] + vs[0] - bottom_margin_end: adj_coor[0] + vs[0] - bottom_margin_start,  # end > start
                    bottom_edge_start: bottom_edge_end,
                    adj_coor[1]: adj_coor[1] + vs[1],
                    :]
            merge_top_face = vol[top_margin_start: top_margin_end, :, :]
            merge_left_face = vol[:, left_margin_start: left_margin_end, :]
            merge_right_face = vol[:, -right_margin_end + 1: -right_margin_start + 1, :]
            merge_bottom_face = vol[-bottom_margin_end + 1: -bottom_margin_start + 1, :, :]

            # Get remapping for each face
            if verbose:
                print('Running horizontal remap')
                elapsed = time.time()
            if run_top:
                remap_top, merge_top_face, update = get_remapping(
                    main_margin=main_top_face,
                    merge_margin=merge_top_face)
                # if update:
                #     vol[:margin, :, :] = merge_top_face
            else:
                remap_top = []
            if run_left:
                remap_left, merge_left_face, update = get_remapping(
                    main_margin=main_left_face,
                    merge_margin=merge_left_face)
            else:
                remap_left = []
            # if update:
            #     vol[:, :margin, :] = merge_left_face
            if run_right:
                remap_right, merge_right_face, update = get_remapping(
                    main_margin=main_right_face,
                    merge_margin=merge_right_face)
            else:
                remap_right = []
            # if update:
            #     vol[:, -margin:, :] = merge_right_face
            if run_bottom:
                remap_bottom, merge_bottom_face, update = get_remapping(
                    main_margin=main_bottom_face,
                    merge_margin=merge_bottom_face)
            else:
                remap_bottom = []
            # if update:
            #     vol[-margin:, :, :] = merge_bottom_face

            # Get sizes and originals for every remap. Sort these for the final remap
            all_remaps = np.array(remap_top + remap_left + remap_right + remap_bottom)
            import ipdb;ipdb.set_trace()
            remap_idx = np.argsort(all_remaps[:, -1])[::-1]
            all_remaps = all_remaps[remap_idx]
            unique_remaps = fastremap.unique(all_remaps[:, 0], return_counts=False) 
            fixed_remaps = {}
            for ur in unique_remaps:  # , rc in zip(unique_remaps, remap_counts):
                mask = all_remaps[:, 0] == ur
                fixed_remaps[ur] = all_remaps[mask][0][1]  # Change all to the biggest
            vol = fastremap.remap(vol, fixed_remaps, preserve_missing_labels=True)
            if verbose:
                print('Finished: {}'.format(time.time() - elapsed))
            main[
                adj_coor[0] + top_margin_start: adj_coor[0] + vs[0] - bottom_margin_start,
                adj_coor[1] + left_margin_start: adj_coor[1] + vs[1] - right_margin_start,
                :] = vol[
                    top_margin_start: vs[0] - bottom_margin_start,
                    left_margin_start: vs[1] - right_margin_start]
        else:
            if remap_labels:
                # Only add to non-zeros
                vol += np.nonzeros(vol) * max_vox
                mv, mxv = fastremap.minmax(vol)
                max_vox += mxv + 1
            # adj_coor = (sel_coor[:-1] - mins) * config.shape
            main[
                adj_coor[0]: adj_coor[0] + xoff,
                adj_coor[1]: adj_coor[1] + yoff,
                :] = vol  # rfo(vol)[0]
        return main, max_vox
    elif prev is not None:  #  == 'bottom-up':
        # Get distance in z
        dz = sel_coor[2] - plane_coors[:, 2][0] 
        adj_dz = dz * config.shape[-1]
        """
        half_dz = adj_dz // 10
        # Grab main bottom face and prev top face. Then run the same remap merge routine above.
        prev_top = prev[
            adj_coor[0]: adj_coor[0] + xoff,
            adj_coor[1]: adj_coor[1] + yoff,
            vs[2] - half_dz]
        if not prev_top.sum():
            # Empty prev face
            return main, max_vox
        vol = main[
            adj_coor[0]: adj_coor[0] + vs[0],
            adj_coor[1]: adj_coor[1] + vs[1],
            half_dz:]
        curr_bottom_face = vol[..., -1]
        prev_top_face = prev_top[..., 0]
        curr_bottom_face = main[
            adj_coor[0]: adj_coor[0] + vs[0],
            adj_coor[1]: adj_coor[1] + vs[1],
            adj_dz]  # -1 + half_dz
        prev_vol = prev[
            adj_coor[0]: adj_coor[0] + vs[0],
            adj_coor[1]: adj_coor[1] + vs[1]]
        prev_top_face = prev_vol[..., -adj_dz]
        """
        curr_bottom_face = main[
            adj_coor[0]: adj_coor[0] + vs[0],
            adj_coor[1]: adj_coor[1] + vs[1],
            -1]  # -1 + half_dz
        prev_vol = prev[
            adj_coor[0]: adj_coor[0] + vs[0],
            adj_coor[1]: adj_coor[1] + vs[1]]
        prev_top_face = prev_vol[..., -adj_dz]
        if not prev_top_face.sum():
            # Prev doesn't have any voxels, pass the original
            return main, {}
        if verbose:
            print('Running bottom-up remap')
            elapsed = time.time()
        all_remaps, _, update = get_remapping(
            main_margin=prev_top_face,
            merge_margin=curr_bottom_face,  # mapping from prev -> main
            use_numba=False)

        # Get sizes and originals for every remap. Sort these for the final remap
        all_remaps = np.array(all_remaps)
        fixed_remaps = {}
        if len(all_remaps):
            remap_idx = np.argsort(all_remaps[:, -1])[::-1]
            all_remaps = all_remaps[remap_idx]
            unique_remaps, remap_counts = fastremap.unique(all_remaps[:, 0], return_counts=True)
            for ur, rc in zip(unique_remaps, remap_counts):
                mask = all_remaps[:, 0] == ur
                fixed_remaps[ur] = all_remaps[mask][0][1]  # Change all to the biggest
            # vol = fastremap.remap(vol, fixed_remaps, preserve_missing_labels=True)
            # # Remap all at once!
            # main[
            #     adj_coor[0]:adj_coor[0] + vs[0],
            #     adj_coor[1]:adj_coor[1] + vs[1],
            #     adj_dz:] = prev_vol[..., :-adj_dz]
            if verbose:
                print('Finished: {}'.format(time.time() - elapsed))
        return main, fixed_remaps  # main, max_vox
    else:
        raise RuntimeError('Something fucked up.')


path_extent = [9, 9, 3]
glob_debug = False
save_cubes = False
merge_debug = True
remap_labels = False
in_place = False
z_max = 384
bu_margin = 2
dtype = np.uint32
config = Config()

### Make lists of dicts, where the keys are the TL corners, and the vals are the individual NIIs

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

# Loop through x-axis
max_vox, count, prev = 0, 0, None
slice_shape = np.concatenate((diffs[:-1], [z_max]))
cifs_path = None  # Force a fail for this. TODO: build a clean API '/media/data_cifs/connectomics/merge_data/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.npy'
coor_status = []
for zidx, z in tqdm(enumerate(unique_z), total=len(unique_z), desc="Z-slice main clock"):
    # This plane
    z_sel_coors = coordinates[coordinates[:, 2] == z]
    # sort_idx = np.argsort(z_sel_coors, -1)[::-1]
    # z_sel_coors = z_sel_coors[sort_idx]
    z_sel_coors = np.unique(z_sel_coors, axis=0)

    # Split into merge + mains
    z_sel_coors_main = z_sel_coors[z_sel_coors[..., -1] == 0]
    z_sel_coors_merge = z_sel_coors[z_sel_coors[..., -1] == 1]

    # Get list of non-colliding merges here.
    collisions = []
    if len(z_sel_coors_main):
        for midx, sel_coor in enumerate(z_sel_coors_merge):
            dists = sel_coor[:-2] - z_sel_coors_main[:, :-2]
            dist_test = np.logical_and(dists[:, 0] > path_extent[0], dists[:, 1] > path_extent[1])
            if np.all(dist_test):  # If this merge is far enough away from all mains
                sel_coor[-1] = 0
                z_sel_coors_main = np.concatenate((z_sel_coors_main, sel_coor))
                collisions.append(True)
            else:
                collisions.append(False)
        z_sel_coors_merge = z_sel_coors_merge[np.array(collisions) == False]
    else:
        # If there are no mains, get a non-overlapping set of merges
        for sel_coor in z_sel_coors_merge:
            dists = sel_coor[:2] - z_sel_coors_merge[:, :2]
            # dist_test = np.logical_and(dists[:, 0] < path_extent[0], dists[:, 1] < path_extent[1])
            collisions.append(dists)
        collisions = np.array(collisions)
        dm_h = np.abs(collisions[..., 0])
        dm_w = np.abs(collisions[..., 1])
        h_test = np.logical_and(dm_h < path_extent[0], dm_h > 0)
        w_test = np.logical_and(dm_w < path_extent[1], dm_w > 0)

        # Find indices to skip
        collisions = np.full(len(h_test), True, dtype=bool)
        for ridx, rt in enumerate(h_test):  # range(hw_test.shape[0])
            for widx, wt in enumerate(w_test):
                if np.logical_and(rt[ridx], wt[widx]):
                    collisions[ridx] = False
        z_sel_coors_main = z_sel_coors_merge[~collisions]
        z_sel_coors_merge = z_sel_coors_merge[collisions] 

    # print('Wherever you dont have mains, see if you can insert a merge (non conflicts with mains), and promote it to a main')
    coord_status = []
    with Parallel(n_jobs=32, max_nbytes=None) as parallel:
        if 1:  # not skip_processing:
            # Load mains in this plane
            if len(z_sel_coors_main):
                for sel_coor in tqdm(z_sel_coors_main, desc='Z (mains): {}'.format(z)):
                    vol = load_npz(sel_coor, shape=config.shape, dtype=dtype, path_extent=path_extent, parallel=parallel)  # .transpose((2, 1, 0))
            else:
                # If no mains, load merges and continue to BU merge
                for sel_coor in tqdm(z_sel_coors_merge, desc='Z (merges no-mains): {}'.format(z)):
                    vol = load_npz(sel_coor, shape=config.shape, dtype=dtype, path_extent=path_extent, parallel=parallel)  # .transpose((2, 1, 0))
        coor_status.append(vol)
        gc.collect()
    coor_status = [x for x in coor_status if x is not None]
    np.save("data_status", coor_status)

