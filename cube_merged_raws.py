import sys
import time
import os
import gzip
import shutil
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
def convert_save_cubes(coords, data, cifs_path, mins, max_z, config):
    """All coords come from the same z-slice. Save these as npys to cifs."""
    # for idx in range(len(coords)):
    # for seed in coords:
    #     seed = coords[idx]
    # adj_coor = (coords[:-1] - mins) * config.shape
    # segments = data[
    #     adj_coor[0]: adj_coor[0] + xoff,
    #     adj_coor[1]: adj_coor[1] + yoff]
    for x in range(path_extent[0]):
        for y in range(path_extent[1]):
            for z in range(max_z):
                path = cifs_path % (
                    npad_zeros(coords[0] + x, 4),
                    npad_zeros(coords[1] + y, 4),
                    npad_zeros(coords[2] + z, 4),
                    npad_zeros(coords[0] + x, 4),
                    npad_zeros(coords[1] + y, 4),
                    npad_zeros(coords[2] + z, 4))
                seg = data[
                    x * config.shape[0]: x * config.shape[0] + config.shape[0],
                    y * config.shape[1]: y * config.shape[1] + config.shape[1],
                    z * config.shape[2]: z * config.shape[2] + config.shape[2]]
                if seg.sum() > 0:
                    nrecursive_make_dir(path)
                    # np.save(path, seg)
                    seg.flatten().tofile(path)
                # with open("{}.raw".format(path), "wb") as raw_file:
                #     raw_file.write(bytearray(seg.astype(seg.dtype).flatten()))


def process_merge(main, sel_coor, mins, config, path_extent, max_vox=None, margin_start=0, margin_end=1, test=0.50, prev=None, plane_coors=None, verbose=False, main_margin_offset=1):
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
cifs_stem = '/media/data_cifs/connectomics/merge_data_nii_raw_v2/'
cifs_path = '{}/1/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.raw'.format(cifs_stem)
out_dir = '/gpfs/data/tserre/data/final_merge/'  # /localscratch/merge/'

# Make the cifs path
recursive_make_dir(cifs_stem)
shutil.copyfile('/media/data_cifs/connectomics/mag1_images/Knossos.conf', os.path.join(cifs_stem, 'Knossos.conf'))
shutil.copyfile('/media/data_cifs/connectomics/mag1_images/datasource-properties.json', os.path.join(cifs_stem, 'datasource-properties.json'))

# Start loop
for zidx, z in tqdm(enumerate(unique_z), total=len(unique_z), desc="Z-slice main clock"):
    # Allocate tensor
    main = np.zeros(slice_shape, np.uint32)

    # This plane
    z_sel_coors = coordinates[coordinates[:, 2] == z]
    # sort_idx = np.argsort(z_sel_coors, -1)[::-1]
    # z_sel_coors = z_sel_coors[sort_idx]
    z_sel_coors = np.unique(z_sel_coors, axis=0)

    # Next plane information
    if zidx < len(unique_z):
        z_next = unique_z[zidx + 1]
        max_z = z_next - z
        # max_z = dz * config.shape[-1]
    else:
        max_z = path_extent[-1]
        # max_z = path_extent[-1] * config.shape[-1]
    # Allow for fast loading for debugging
    if os.path.exists(os.path.join(out_dir, 'plane_z{}.npy'.format(z))):
        main = np.load(os.path.join(out_dir, 'plane_z{}.npy'.format(z)))
        for sel_coor in tqdm(z_sel_coors, desc='Z (saving): {}'.format(z)):
            adj_coor = (sel_coor[:-1] - mins) * config.shape
            vol = main[
                adj_coor[0]: adj_coor[0] + xoff,
                adj_coor[1]: adj_coor[1] + yoff]
            convert_save_cubes(
                data=vol,
                coords=sel_coor,
                cifs_path=cifs_path,
                mins=mins,
                max_z=max_z,
                config=config)

