"""Clean segmentations and reasign dust to nearest neighbor segment."""
import re
import numpy as np
from skimage import measure, morphology
from tqdm import tqdm
from scipy import stats
from matplotlib import pyplot as plt
import nibabel as nib
from copy import deepcopy
from scipy import ndimage


def clean_segments(segments, connectivity=2, extent=1, background=0):
    """Run segment cleaning routines.
    1. Label segments to get connected components
    2. Measure region props and sort components by area
    """
    labeled_segments = measure.label(
        segments, connectivity=connectivity, background=background)
    props = np.asarray(measure.regionprops(labeled_segments, coordinates='rc'))
    # Sort in ascending order for greedy agglomeration
    areas = np.asarray([x.area for x in props])
    sort_areas = np.argsort(areas)
    props = props[sort_areas]
    areas = areas[sort_areas]
    # Gather cluster IDs
    ids = []
    for p in props:
        ids += [
            labeled_segments[p.coords[0][0], p.coords[0][1], p.coords[0][2]]]
    ids = np.asarray(ids)
    reassign = np.where(areas < threshold)[0]
    keep = np.where(areas >= threshold)[0]
    keep_ids = []
    for k in keep:
        coords = props[k].coords
        keep_ids += [
            segments[coords[0, 0], coords[0, 1], coords[0, 2]]]
    keep_ids = np.asarray(keep_ids)
    max_vals = (np.asarray(segments.shape) - 1).repeat(1)
    min_vals = np.asarray([0]).repeat(3)
    for pidx in tqdm(reassign, total=len(reassign)):
        coords = props[pidx].coords
        # Heuristic: Reassign to segment closest to NN
        z_min_idx = np.argmin(coords, axis=0)[2]
        z_max_idx = np.argmax(coords, axis=0)[2]
        z_search_seed_min = [
            coords[z_min_idx, 0],
            coords[z_min_idx, 1],
            coords[z_min_idx, 2] - extent]
        z_search_seed_max = [
            coords[z_max_idx, 0],
            coords[z_max_idx, 1],
            coords[z_max_idx, 2] + extent]
        y_min_idx = np.argmin(coords, axis=0)[1]
        y_max_idx = np.argmax(coords, axis=0)[1]
        y_search_seed_min = [
            coords[y_min_idx, 0],
            coords[y_min_idx, 1] - extent,
            coords[y_min_idx, 2]]
        y_search_seed_max = [
            coords[y_max_idx, 0],
            coords[y_max_idx, 1] + extent,
            coords[y_max_idx, 2]]
        x_min_idx = np.argmin(coords, axis=0)[0]
        x_max_idx = np.argmax(coords, axis=0)[0]
        x_search_seed_min = [
            coords[x_min_idx, 0] - extent,
            coords[x_min_idx, 1],
            coords[x_min_idx, 2]]
        x_search_seed_max = [
            coords[x_max_idx, 0] + extent,
            coords[x_max_idx, 1],
            coords[x_max_idx, 2]]
        x_search_seed_max = np.asarray(x_search_seed_max)
        y_search_seed_max = np.asarray(y_search_seed_max)
        z_search_seed_max = np.asarray(z_search_seed_max)
        x_search_seed_min = np.asarray(x_search_seed_min)
        y_search_seed_min = np.asarray(y_search_seed_min)
        z_search_seed_min = np.asarray(z_search_seed_min)
        x_search_seed_max = np.minimum(x_search_seed_max, max_vals)
        y_search_seed_max = np.minimum(y_search_seed_max, max_vals)
        z_search_seed_max = np.minimum(z_search_seed_max, max_vals)
        x_search_seed_max = np.maximum(x_search_seed_max, min_vals)
        y_search_seed_max = np.maximum(y_search_seed_max, min_vals)
        z_search_seed_max = np.maximum(z_search_seed_max, min_vals)
        x_search_seed_min = np.minimum(x_search_seed_min, max_vals)
        y_search_seed_min = np.minimum(y_search_seed_min, max_vals)
        z_search_seed_min = np.minimum(z_search_seed_min, max_vals)
        x_search_seed_min = np.maximum(x_search_seed_min, min_vals)
        y_search_seed_min = np.maximum(y_search_seed_min, min_vals)
        z_search_seed_min = np.maximum(z_search_seed_min, min_vals)
        # Get nearest neighbors in the x/y/z
        plus_x = labeled_segments[
            x_search_seed_max[0], x_search_seed_max[1], x_search_seed_max[2]]
        minus_x = labeled_segments[
            x_search_seed_min[0], x_search_seed_min[1], x_search_seed_min[2]]
        plus_y = labeled_segments[
            y_search_seed_max[0], y_search_seed_max[1], y_search_seed_max[2]]
        minus_y = labeled_segments[
            y_search_seed_min[0], y_search_seed_min[1], y_search_seed_min[2]]
        plus_z = labeled_segments[
            z_search_seed_max[0], z_search_seed_max[1], z_search_seed_max[2]]
        minus_z = labeled_segments[
            z_search_seed_min[0], z_search_seed_min[1], z_search_seed_min[2]]
        all_labels = [
            plus_x,
            minus_x,
            plus_y,
            minus_y,
            plus_z,
            minus_z,
        ]
        all_labels = np.asarray(all_labels)
        # Get NN counts
        area_plus_x = areas[ids == all_labels[0]]
        area_minus_x = areas[ids == all_labels[1]]
        area_plus_y = areas[ids == all_labels[2]]
        area_minus_y = areas[ids == all_labels[3]]
        area_plus_z = areas[ids == all_labels[4]]
        area_minus_z = areas[ids == all_labels[5]]
        all_areas = [
            area_plus_x,
            area_minus_x,
            area_plus_y,
            area_minus_y,
            area_plus_z,
            area_minus_z,
        ]
        # Argmax across the counts
        if len(np.concatenate(all_areas)):
            biggest_seg = np.argmax(all_areas)
            swap = all_labels[biggest_seg]
        else:
            swap = 0
        for coord in coords:
            labeled_segments[coord[0], coord[1], coord[2]] = swap
    return labeled_segments, reassign, keep_ids
# General settings
threshold = 1000  # 4096
med_filt = 3
split_threshold = threshold * 2  #  Used to be threshold * 4 for berson suff
erosion = 1
transpose = True
mode = 'reassign'  # 'remove' or 'reassign'
# volume = np.load('cube_sem_data_uint8.npy')
version = 41
segment_name = 'v2_seg_05.nii.gz'  # 'seg-0_0_2.npz'  # 'Sept19_curated_v%s.nii.gz' % version
label_name = 'v%s_labels.txt' % version
transpose = (2, 1, 0)
# Iterative grouping settings
iterations = 10
extent = 1
connectivity = 1
# Start script
if '.npy' in segment_name:
    segments = np.load(segment_name)  # In x/y/z
elif '.nii' in segment_name:
    segments = nib.load(segment_name).get_fdata().astype(np.uint16)  # In x/y/z
elif '.npz' in segment_name:
    segments = np.load(segment_name)['segmentation']
else:
    raise NotImplementedError
with open(label_name, 'rb') as f:
    cell_labels = np.asarray(f.readlines())
segments = segments.transpose(transpose)
header = cell_labels[:14]
cell_labels = cell_labels[14:]
if mode == 'remove':
    raise NotImplementedError
    labeled_segments = morphology.remove_small_objects(
        segments,
        min_size=threshold)
    raise NotImplementedError('Do not use remove small objects routine')
else:
    below_thresh = np.inf
    labeled_segments = np.copy(segments)
    stop = False
    # Continue running until there are no subthreshold segments
    while below_thresh > 0 and iterations > 0:  # or stop:
        labeled_segments, reassign, keep_ids = clean_segments(
            labeled_segments, extent=extent, connectivity=connectivity)
        new_thresh = len(reassign)
        print 'Iteration %s, %s below threshold' % (iterations, new_thresh)
        below_thresh = new_thresh
        iterations -= 1
    # Reassign new IDs the ID from the original volume that modally overlap
    unique_labels = np.unique(labeled_segments)
    fixed_segs = np.zeros_like(segments)
    for k in tqdm(
            unique_labels,
            desc='Reassigning labels',
            total=len(unique_labels)):
        mask = labeled_segments == k
        m = (mask).astype(float)
        seg_vec = segments * m
        seg_vec = seg_vec[seg_vec > 0]
        modal = stats.mode(seg_vec)[0][0]
        labeled_segments[mask] = modal
    # Iterate the IDs of discontinuous suprathreshold segments
    new_segs = np.unique(labeled_segments)
    counter = np.max(labeled_segments) + 1
    copy_labels = deepcopy(labeled_segments)
    for s in tqdm(new_segs, total=len(new_segs), desc='Fixing discontinuous'):
        selected_seg = measure.label(
            copy_labels == s, connectivity=connectivity, background=0)
        props = np.asarray(measure.regionprops(
            selected_seg, coordinates='rc'))
        if len(props) > 1:
            # Sort in ascending order
            areas = np.asarray([x.area for x in props])
            area_idx = np.argsort(areas)
            props = props[area_idx]
            props = props[:-1]  # Ignore biggest volume + bg
            for p in props:
                if p.area > split_threshold:  # Fix suprathreshold segments
                    coords = p.coords
                    for coor in coords:
                        labeled_segments[
                            coor[0], coor[1], coor[2]] = counter
                    counter += 1
    # Print out info on volumes
    old_segs = np.unique(segments)
    new_segs = np.unique(labeled_segments)
    print 'Original num labels %s' % len(old_segs)
    print 'New num labels %s' % len(new_segs)
# segments = segmentation.relabel_from_one(segments)[0]
# new_props = np.asarray(measure.regionprops(segments))
# Find correspondence between new/old labels
correspondence = np.in1d(old_segs, new_segs)
keep_labels = np.where(correspondence)[0]
labs = cell_labels[keep_labels]
# Build out rows for new labels
new_labels = new_segs[~np.in1d(new_segs, old_segs)]
template_label = cell_labels[-1]
swap_key = re.search(r'\d+', template_label).group()
new_labs = []
for lab in new_labels:
    new_labs += [deepcopy(template_label).replace(swap_key, str(lab))]
out_labs = np.concatenate([header, labs, new_labs])
# Write new labels
with open('cleaned_%s' % label_name, 'w') as f:
    for l in out_labs:
        f.write(l)
# Reassign segments to their human annotated IDs.
out_npy_name = 'filt_cleaned_predicted_segs_CURATED_v%s_max_%s.npy' % (
    version, threshold)
out_gipl_name = 'cleaned_predicted_segs_CURATED_v%s_max_%s.nii.gz' % (
    version, threshold)
out_filt_gipl_name = 'filt_cleaned_predicted_segs_CURATED_v%s_max_%s.nii.gz' % (
    version, threshold)
labeled_segments = labeled_segments.astype(np.uint16)
# Make sure background ID is correct then filter
bg = stats.mode(labeled_segments.ravel()[segments.ravel() == 0])[0][0]
labeled_segments[labeled_segments == bg] = 0
filt_labeled_segments = ndimage.median_filter(
    labeled_segments.astype(np.uint16), med_filt)
np.save(
    out_npy_name,
    filt_labeled_segments)
print len(np.unique(labeled_segments))
# Save the data as a nifti file
if transpose:
    img = nib.Nifti1Image(
        labeled_segments.transpose(2, 1, 0), np.eye(4))
    img_filt = nib.Nifti1Image(
        filt_labeled_segments.transpose(2, 1, 0), np.eye(4))
else:
    img = nib.Nifti1Image(
        labeled_segments, np.eye(4))
    img_filt = nib.Nifti1Image(filt_labeled_segments, np.eye(4))
nib.save(img, out_gipl_name)
nib.save(img_filt, out_filt_gipl_name)
# Save some plots
f = plt.figure()
slc = 40
plt.subplot(121)
plt.axis('off')
plt.title('original slice %s' % slc)
plt.imshow(segments[slc])
plt.subplot(122)
plt.axis('off')
plt.title('cleaned slice %s' % slc)
plt.imshow(labeled_segments[slc])
plt.savefig('im_%s.png' % slc)
plt.close(f)
f = plt.figure()
slc = 80
plt.subplot(121)
plt.axis('off')
plt.title('original slice %s' % slc)
plt.imshow(segments[slc])
plt.subplot(122)
plt.axis('off')
plt.title('cleaned slice %s' % slc)
plt.imshow(labeled_segments[slc])
plt.savefig('im_%s.png' % slc)
plt.close(f)
f = plt.figure()
slc = 120
plt.subplot(121)
plt.axis('off')
plt.title('original slice %s' % slc)
plt.imshow(segments[slc])
plt.subplot(122)
plt.axis('off')
plt.title('cleaned slice %s' % slc)
plt.imshow(labeled_segments[slc])
plt.savefig('im_%s.png' % slc)
plt.close(f)