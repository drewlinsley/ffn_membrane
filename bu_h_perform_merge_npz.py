import sys
import os
import fastremap
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
from utils.hybrid_utils import rdirs, pad_zeros
from skimage import measure


def get_remapping(main_margin, merge_margin):
    """Determine where to merge."""
    # Loop through the margin in main, to find per-segment overlaps with merge
    unique_main = fastremap.unique(main_margin, return_counts=False)
    remap = []
    for um in unique_main:  # Package this as a function
        masked_plane = main_margin == um  # fastremap.mask_except(h_plane, um)
        overlap = merge_margin[masked_bu]

        # If overlap is a large enough proportion, propogate the main-id to merge
        prop = float(overlap.sum()) / float(masked_bu.sum())
        if prop >= test:
            uni_over = fastremap.unique(merge_margin, return_counts=False)
            for ui in uni_over:
                remap.append({ui: um})  # Append merge ids for the overlap
    return remap


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


def process_merge(main, sel_coor, mins, config, path_extent, margin=5, test=0.66, prev=None, plane_coors=None):
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
            vol, remapping = fastremap.renumber(vol, in_place=in_place)
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

        import ipdb;ipdb.set_trace()
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

            # Find tl/tr/bl/br to pass
            crop_top, crop_left, crop_bottom, crop_right = 0, 0, path_extent[0], path_extent[1]
            if np.any(c_tl[idx_tl]) > path_extent[0]:
                tl_d = nns_tl[idx_tl] 
                crop_top = tl_d[0] * config.shape[0]
                crop_left = tl_d[1] * config.shape[1]

            if np.any(c_tr[idx_tr]) > path_extent[0]:
                tr_d = nns_tr[idx_tr]
                crop_top = tr_d[0] * config.shape[0]
                crop_right = tr_d[1] * config.shape[1] + vol.shape[1]
                # crop_bottom = br_d[0] * config.shape[0]
                # crop_right = br_d[1] * config.shape[1]

            # # Crop the volume
            # vol = vol[crop_top:crop_bottom, crop_left:crop_right]

            # Check horizontal plane
            mx_ty_mz = main[mx, ty, :]
            mx_by_mz = main[mx, by, :]
            lx_my_mz = main[lx, my, :]
            rx_my_mz = main[rx, my, :]

            # If there's content in a horizontal plane we can merge            
            mx_ty_mz_sum = mx_ty_mz.sum()
            mx_by_mz_sum = mx_by_mz.sum()
            lx_my_mz_sum = lx_my_mz.sum()
            rx_my_mz_sum = rx_my_mz.sum()
            # Grab some margin and then hunt for overlaps
            import ipdb;ipdb.set_trace()
            if mx_ty_mz_sum:
                # Work on (0, 1)
                main_margin = main[:, :margin, :]
                merge_margin = vol[:, -margin:, :]
                remap = get_remapping(
                    main_margin=main_margin,
                    merge_margin=merge_margin)
            if mx_by_mz_sum:
                # Work on (0, -1)
                main_margin = main[:, -margin:, :]
                merge_margin = vol[:, :margin, :]
                remap = get_remapping(
                    main_margin=main_margin,
                    merge_margin=merge_margin)
            if lx_my_mz_sum:
                # Work on (-1, 0)
                main_margin = main[:margin, :, :]
                merge_margin = vol[-margin:, :, :]
                remap = get_remapping(
                    main_margin=main_margin,
                    merge_margin=merge_margin)
            if rx_my_mz_sum:
                # Work on (1, 0)
                main_margin = main[-margin:, :, :]
                merge_margin = vol[:margin, :, :]
                remap = get_remapping(
                    main_margin=main_margin,
                    merge_margin=merge_margin)

            # Insert the merge volume at the adjusted coordinates
            crop_adj_coors = adj_coor
            crop_adj_coor[0] += crop_top
            crop_adj_coor[1] += crop_left
            main[
                crop_adj_coors[0]:crop_adj_coors[0] + crop_bottom,
                crop_adj_coors[1]:crop_adj_coors[1] + crop_right,
                :] = vol[crop_top:crop_bottom, crop_left:crop_right]
        else:
            if remap_labels:
                vol, remapping = fastremap.renumber(vol, in_place=in_place)
                vol += max_vox
                mv, mxv = fastremap.minmax(vol)
                max_vox += mxv + 1
            adj_coor = (sel_coor[:-1] - mins) * config.shape
            import ipdb;ipdb.set_trace()
            main[
                adj_coor[0]: adj_coor[0] + xoff,
                adj_coor[1]: adj_coor[1] + yoff,
                :] = vol  # rfo(vol)[0]
    elif prev is not None:  #  == 'bottom-up':
        # Do this for each unique sel_coor in the current plane
        # Extract "vol" from the prev
        vol = prev[
            adj_coor[0]: adj_coor[0] + xoff,
            adj_coor[1]: adj_coor[1] + yoff,
            :] = vol  # rfo(vol)[0]

        # Get midpoints for checking bottom-up merge
        mp_coor_mx_my_bz = (mx, my, lz)  # (sel_coor[:-1] - mins) * config.shape

        # Check bottom-up plane
        bu_plane = main[:, :, -1]

        # If there's content in a bottom-up plane we can merge
        bu_plane_sum = bu_plane.sum()

        # Grab some margin and then hunt for overlaps
        if bu_plane_sum:
            main_margin = main[:, :, -margin:]
            merge_margin = vol[:, :, :margin]
            remap = get_remapping(
                main_margin=main_margin,
                merge_margin=merge_margin)

            # Perform the remapping
            main = fastremap.remap_from_array_kv(remap.keys(), remap.value())
    else:
        raise RuntimeError('Something fucked up.')


path_extent = [9, 9, 3]
glob_debug = True
merge_debug = True
remap_labels = True
in_place = False
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
debug = -1  # 2
max_vox, count = 0, 0
slice_shape = np.concatenate((diffs[:-1], [384]))
# out_dir = '/media/data_cifs/connectomics/merge_data/'
out_dir = '/localscratch/merge/'
for zidx, z in tqdm(enumerate(unique_z), total=len(unique_z), desc="z-slice"):
    # Allocate tensor
    if zidx == 0:
        main = np.zeros(slice_shape, np.uint32)
    elif zidx > 24:
        out_dir = '/media/data/merge/'
    # else:
    #     main = merge

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
        import ipdb;ipdb.set_trace()
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
                    vol, remapping = fastremap.renumber(vol, in_place=in_place) 
                    vol += max_vox
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
                    vol, remapping = fastremap.renumber(vol, in_place=in_place)
                    vol += max_vox
                    mv, mxv = fastremap.minmax(vol)
                    max_vox += mxv + 1
                adj_coor = (sel_coor[:-1] - mins) * config.shape
                main[
                    adj_coor[0]: adj_coor[0] + xoff,
                    adj_coor[1]: adj_coor[1] + yoff,
                    :] = vol  # rfo(vol)[0]

        # Start merge logic:::
        if zidx > 0:
            # Perform horizontal merge if there's admixed main/merge
            for sel_coor in tqdm(z_sel_coors_merge, desc='Z: {}'.format(z)):
                main = process_merge(
                    main=main,
                    sel_coor=sel_coor,
                    mins=mins,
                    config=config,
                    plane_coors=np.copy(z_sel_coors_main),
                    path_extent=path_extent)

            for sel_coor in tqdm(z_sel_coors_main, desc='Z (mains): {}'.format(z)):
                # Perform bottom-up merge
                main = process_merge(
                    main=main,
                    sel_coor=sel_coor,
                    mins=mins,
                    config=config,
                    path_extent=path_extent,
                    prev=prev)

        np.save(os.path.join(out_dir, 'plane_z{}'.format(z)), main)

        # Save the current main for the next slice
        prev = np.copy(main)
    else:
        print('Skipping plane {}'.format(z))

