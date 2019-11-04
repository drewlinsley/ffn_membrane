"""Clean segmentations and reasign dust to nearest neighbor segment."""
import re
import numpy as np
from skimage import measure, morphology
from tqdm import tqdm
from scipy import stats
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy import ndimage


def clean_segments(segments, connectivity=2, extent=1, background=0, threshold=1000):
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

