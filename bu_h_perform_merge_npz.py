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
from scipy.spatial import distance
from utils.hybrid_utils import rdirs
from utils.hybrid_utils import pad_zeros
from utils.hybrid_utils import recursive_make_dir
from skimage import measure
from numba import njit, jit, autojit, prange


@njit(parallel=True, fastmath=True)
def direct_overlaps(main_margin, merge_margin, um):
    overlaps = np.zeros_like(merge_margin)
    for h in prange(main_margin.shape[0]):
        if main_margin[h] == um:
            overlaps[h] = merge_margin[h]
    return overlaps


def get_remapping(main_margin, merge_margin, use_numba=False):
    """Determine where to merge."""
    # Loop through the margin in main, to find per-segment overlaps with merge
    if not len(main_margin):
        return None
    if not len(merge_margin):
        return None
    unique_main = fastremap.unique(main_margin, return_counts=False)
    unique_main = unique_main[unique_main > 0]
    if not len(unique_main):
        return [], merge_margin, False
    remap = []
    transfers = []
    update = False
    # For each segment in main, find the corresponding seg in margin. Transfer the id over, or transfer the bigger segment over (second needs to be experimental).
    for um in unique_main:  # Package this as a function
        if use_numba:
            overlap = direct_overlaps(main_margin.reshape(-1), merge_margin.reshape(-1), um)
        else:
            masked_plane = main_margin == um  # fastremap.mask_except(h_plane, um)
            overlap = merge_margin[masked_plane]
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
            uni_over = uni_over[uni_over > 0]
            for ui, uc in zip(uni_over, counts):
                remap.append([ui, um, uc])  # Append merge ids for the overlap
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


def process_merge(main, sel_coor, mins, config, path_extent, max_vox=None, margin_start=0, margin_end=1, test=0.50, prev=None, plane_coors=None, verbose=False, margin_steps=3, main_offset=5):
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
        vol = load_npz(sel_coor).transpose((2, 1, 0))
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
            candidate_diffs = np.abs(candidate_coords - adj_coor)

            ## Merge overlaps with main. Find which 
            ## mains we overlap with. Push the merge to the top of the stack
            ## 
            # Top crops
            top_margin_idx = np.logical_and(np.logical_and(candidate_diffs[:, 0] == 0, candidate_diffs[:, 1] == 0), candidate_diffs[:, 2] == 0)
            if np.any(top_margin_idx):
                top_margin_start = candidate_diffs[top_margin_idx][0][0] // 2
                top_margin_end = top_margin_start + main_offset
            else:
                top_margin_start = margin_start
                top_margin_end = margin_end
            merge_top_face = vol[top_margin_start: top_margin_end, :, :]

            # Left crops
            left_margin_idx = np.logical_and(np.logical_and(candidate_diffs[:, 0] == 0, candidate_diffs[:, 1] > 0), candidate_diffs[:, 2] == 0)
            if np.any(left_margin_idx):
                left_margin_start = candidate_diffs[left_margin_idx][0][1] // 2
                left_margin_end = left_margin_start + main_offset
            else:
                left_margin_start = margin_start
                left_margin_end = margin_end
            merge_left_face = vol[:, left_margin_start: left_margin_end, :]

            # Right crops
            right_margin_idx = np.logical_and(np.logical_and(candidate_diffs[:, 0] == 0, candidate_diffs[:, 1] > 0), candidate_diffs[:, 2] == 0)
            if np.any(right_margin_idx):
                right_margin_start = candidate_diffs[right_margin_idx][0][1] // 2
                right_margin_end = right_margin_start + main_offset
            else:
                right_margin_start = margin_start
                right_margin_end = margin_end
            merge_right_face = vol[:, -right_margin_end: -right_margin_start, :]

            # Bottom crops
            bottom_margin_idx = np.logical_and(np.logical_and(candidate_diffs[:, 0] > 0, candidate_diffs[:, 1] == 0), candidate_diffs[:, 2] == 0)
            if np.any(bottom_margin_idx):
                bottom_margin_start = candidate_diffs[bottom_margin_idx][0][0] // 2
                bottom_margin_end = bottom_margin_start + main_offset
            else:
                bottom_margin_start = margin_start
                bottom_margin_end = margin_end
            merge_bottom_face = vol[-bottom_margin_end: -bottom_margin_start, :, :]

            # Figure out the margin
            main_top_face = main[
                adj_coor[0] + top_margin_end: adj_coor[0] + top_margin_end + 1,  # use offset to interpolate
                adj_coor[1]: adj_coor[1] + vs[1],
                :]
            main_left_face = main[
                adj_coor[0]: adj_coor[0] + vs[0],
                adj_coor[1] + left_margin_end: adj_coor[1] + left_margin_end + 1,
                :]
            main_right_face = main[
                adj_coor[0]: adj_coor[0] + vs[0],
                adj_coor[1] + vs[1] - right_margin_end: adj_coor[1] + vs[1] - right_margin_end + 1,
                :]
            main_bottom_face = main[
                adj_coor[0] + vs[0] - bottom_margin_end: adj_coor[0] + vs[0] - bottom_margin_end + 1,
                adj_coor[1]: adj_coor[1] + vs[1],
                :]

            # Get remapping for each face
            if verbose:
                print('Running horizontal remap')
                elapsed = time.time()
            import ipdb;ipdb.set_trace()
            remap_top, merge_top_face, update = get_remapping(
                main_margin=main_top_face,
                merge_margin=merge_top_face)
            remap_left, merge_left_face, update = get_remapping(
                main_margin=main_left_face,
                merge_margin=merge_left_face)
            import ipdb;ipdb.set_trace()
            remap_right, merge_right_face, update = get_remapping(
                main_margin=main_right_face,
                merge_margin=merge_right_face)
            remap_bottom, merge_bottom_face, update = get_remapping(
                main_margin=main_bottom_face,
                merge_margin=merge_bottom_face)

            # Get sizes and originals for every remap. Sort these for the final remap
            all_remaps = np.array(remap_top + remap_left + remap_right + remap_bottom)
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
            import ipdb;ipdb.set_trace()
            main[
                adj_coor[0]: adj_coor[0] + vs[0],
                adj_coor[1]: adj_coor[1] + vs[1],
                :] = vol
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
        # Grab main bottom face and prev top face. Then run the same remap merge routine above.
        prev_top = prev[
            adj_coor[0]: adj_coor[0] + xoff,
            adj_coor[1]: adj_coor[1] + yoff]
        if not prev_top.sum():
            # Empty prev face
            return main, max_vox
        vol = main[
            adj_coor[0]: adj_coor[0] + vs[0],
            adj_coor[1]: adj_coor[1] + vs[1]]
        curr_bottom_face = vol[..., -margin_end: -margin_start]
        prev_top_face = prev_top[..., margin_start: margin_end]
        if verbose:
            print('Running bottom-up remap')
            elapsed = time.time()
        all_remaps, _, update = get_remapping(
            main_margin=prev_top_face,
            merge_margin=curr_bottom_face,  # mapping from prev -> main
            use_numba=False)

        # Get sizes and originals for every remap. Sort these for the final remap
        all_remaps = np.array(all_remaps)
        remap_idx = np.argsort(all_remaps[:, -1])[::-1]
        all_remaps = all_remaps[remap_idx]
        unique_remaps, remap_counts = fastremap.unique(all_remaps[:, 0], return_counts=True)
        fixed_remaps = {}
        for ur, rc in zip(unique_remaps, remap_counts):
            mask = all_remaps[:, 0] == ur
            fixed_remaps[ur] = all_remaps[mask][0][1]  # Change all to the biggest
        vol = fastremap.remap(vol, fixed_remaps, preserve_missing_labels=True)
        if verbose:
            print('Finished: {}'.format(time.time() - elapsed))
        main[
            adj_coor[0]:adj_coor[0] + vs[0],
            adj_coor[1]:adj_coor[1] + vs[1],
            :] = vol
        if verbose:
            print('Finished: {}'.format(time.time() - elapsed))
        return main, max_vox
    else:
        raise RuntimeError('Something fucked up.')


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
og_coordinates = db.pull_membrane_coors()
if glob_debug:
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
merges = db.pull_merge_membrane_coors()
if glob_debug:
    new_merges = []
    for r in merges:
        sel_coor = [r['x'], r['y'], r['z']]
        check = glob(os.path.join('/media/data_cifs/connectomics/ding_segmentations_merge/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4))))
        if len(check):
            new_merges.append(sel_coor)
    merges = np.array(new_merges)
else:
    merges = np.array([[r['x'], r['y'], r['z']] for r in merges if r['processed_segmentation']])

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
cifs_path = '/media/data_cifs/connectomics/merge_data/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.npy'
out_dir = '/gpfs/data/tserre/data/final_merge/'  # /localscratch/merge/'
# unique_z = unique_z[35:]
for zidx, z in tqdm(enumerate(unique_z), total=len(unique_z), desc="Z-slice main clock"):
    # Allocate tensor
    main = np.zeros(slice_shape, np.uint32)

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

    # Allow for fast loading for debugging
    skip_processing = False
    if merge_debug:
        if os.path.exists(os.path.join(out_dir, 'plane_z{}.npy'.format(z))):
            prev = np.load(os.path.join(out_dir, 'plane_z{}.npy'.format(z)))
            skip_processing = True
    # print('Wherever you dont have mains, see if you can insert a merge (non conflicts with mains), and promote it to a main')
    if not skip_processing:
        # Load mains in this plane
        if len(z_sel_coors_main):
            for sel_coor in tqdm(z_sel_coors_main, desc='Z (mains): {}'.format(z)):
                vol = load_npz(sel_coor).transpose((2, 1, 0))
                if remap_labels:
                    # vol, remapping = fastremap.renumber(vol, in_place=in_place) 
                    vol = rfo(vol)[0]
                    vol += np.nonzeros(vol) * max_vox
                    mv, mxv = fastremap.minmax(vol)
                    max_vox += mxv + 1
                adj_coor = (sel_coor[:-1] - mins) * config.shape
                main[
                    adj_coor[0]: adj_coor[0] + xoff,
                    adj_coor[1]: adj_coor[1] + yoff,
                    :] = vol  # rfo(vol)[0]
        else:
            # If no mains, load merges and continue to BU merge
            for sel_coor in tqdm(z_sel_coors_merge, desc='Z (merges no-mains): {}'.format(z)):
                vol = load_npz(sel_coor).transpose((2, 1, 0))
                if remap_labels:
                    # vol, remapping = fastremap.renumber(vol, in_place=in_place) 
                    vol = rfo(vol)[0]
                    vol += np.nonzeros(vol) * max_vox
                    mv, mxv = fastremap.minmax(vol)
                    max_vox += mxv + 1
                adj_coor = (sel_coor[:-1] - mins) * config.shape
                main[
                    adj_coor[0]: adj_coor[0] + xoff,
                    adj_coor[1]: adj_coor[1] + yoff,
                    :] = vol  # rfo(vol)[0]
            z_sel_coors_main = np.copy(z_sel_coors_merge)
            z_sel_coors_merge = []

        # Perform horizontal merge if there's admixed main/merge
        for sel_coor in tqdm(z_sel_coors_merge, desc='H Merging: {}'.format(z)):
            main, max_vox = process_merge(
                main=main,
                sel_coor=sel_coor,
                mins=mins,
                config=config,
                max_vox=max_vox,
                plane_coors=z_sel_coors_main,  # np.copy(z_sel_coors_main),
                path_extent=path_extent)
            z_sel_coors_main = np.concatenate((z_sel_coors_main, [sel_coor]), 0)
        # Perform bottom-up merge
        # if len(z_sel_coors_merge):  This happens in the above loop
        #     z_sel_coors_main = np.concatenate((z_sel_coors_main, z_sel_coors_merge), 0)
        if prev is not None:
            margin = config.shape[-1] * (unique_z[zidx] - unique_z[zidx - 1])
            if margin < z_max:
                for sel_coor in tqdm(z_sel_coors_main, desc='BU Merging: {}'.format(z)):
                    main, max_vox = process_merge(
                        main=main,
                        sel_coor=sel_coor,
                        margin_start=margin,
                        margin_end=margin + bu_margin,
                        mins=mins,
                        config=config,
                        plane_coors=prev_coords,
                        path_extent=path_extent,
                        prev=prev)
        # Save the current main and retain info for the next slice
        if save_cubes:
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
            # f = gzip.GzipFile(os.path.join(out_dir, 'plane_z{}.npy.gz'.format(z)), 'w')
            # np.save(file=f, arr=main)
            # f.close()
    else:
        print('Skipping plane {}'.format(z))
    prev = np.copy(main)
    prev_coords = np.copy(z_sel_coors_main)

