import sys
import os
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
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from utils.hybrid_utils import rdirs, pad_zeros


def save_tv(tv, sel_coor, config, path_extent):
    """Save tv as 128^3 volumes."""
    for z in range(path_extent[0]):
        for y in range(path_extent[1]):
            for x in range(path_extent[2]):
                out_path = config.merge_str % (
                    pad_zeros(sel_coor[0] + x, 4),
                    pad_zeros(sel_coor[1] + y, 4),
                    pad_zeros(sel_coor[2] + z, 4),
                    pad_zeros(sel_coor[0] + x, 4),
                    pad_zeros(sel_coor[1] + y, 4),
                    pad_zeros(sel_coor[2] + z, 4))
                rdirs(np.array(sel_coor) + np.array([x, y, z]), config.merge_str, verbose=False)
                v = tv[
                    z * config.shape[0]: z * config.shape[0] + config.shape[0],  # nopep8
                    y * config.shape[1]: y * config.shape[1] + config.shape[1],  # nopep8
                    x * config.shape[2]: x * config.shape[2] + config.shape[2]]  # nopep8
                np.save(out_path, v)


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


def print_solution(manager, routing, solution, z, report=True):
    """Prints solution on console."""
    # print('Objective: {} volumes'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Merging path for z={}:\n'.format(z)
    route_distance = 0
    outputs = [index]
    distances = [0]
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        outputs.append(index)
        cost = routing.GetArcCostForVehicle(previous_index, index, 0)
        route_distance += cost  # routing.GetArcCostForVehicle(previous_index, index, 0)
        distances.append(cost)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    if report:
        print(plan_output)
    plan_output += 'Route distance: {}miles\n'.format(route_distance)
    return outputs, distances


def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]


package_path = True
path_extent = [9, 9, 3]
config = Config()

# Get list of coordinates
og_coordinates = db.pull_membrane_coors()
og_coordinates = np.array([[r['x'], r['y'], r['z']] for r in og_coordinates if r['processed_segmentation']])

# Get list of merges
merges = db.pull_merge_membrane_coors()
merges = np.array([[r['x'], r['y'], r['z']] for r in merges if r['processed_segmentation']])

# Loop over coordinates
coordinates = np.concatenate((og_coordinates, np.zeros_like(og_coordinates)[:, 0][:, None]), 1)
merges = np.concatenate((merges, np.ones_like(merges)[:, 0][:, None]), 1)
coordinates = np.concatenate((coordinates, merges))
unique_z = np.unique(coordinates[..., -2])
# Loop through z-axis
max_vox, count = 0, 0
all_nodes = []
for zidx, z in tqdm(enumerate(unique_z), total=len(unique_z), desc="Z-slice"):
    if count > 0:
        # Find the closest z to merge upwards
        import ipdb;ipdb.set_trace()
        # z_merge = z_sel_coors[z_merge, :-1]

    z_mask = coordinates[..., -2] == z
    z_sel_coors = coordinates[z_mask]
    z_sel_coors = z_sel_coors[np.argsort(z_sel_coors[:, 1])]
    # z_sel_coors = z_sel_coors[np.argsort(z_sel_coors[:, 2])]


    adj_mat = distance.squareform(distance.pdist(z_sel_coors[..., :-1], 'cityblock'))
    # adj_mat[adj_mat > 9.] = 100.
    data = {'distance_matrix': adj_mat.astype(int).tolist(), 'num_vehicles': 1, 'depot': 0}
    manager = pywrapcp.RoutingIndexManager(len(adj_mat), data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        nodes, _ = print_solution(manager, routing, solution, z=z, report=False)
        all_nodes.append(nodes)
        ds = [0]
        node_pairs = []
        for idx in range(len(nodes[:-2])):
            seld = adj_mat[nodes[idx], nodes[idx + 1]]
            ds.append(seld)
            if seld < 9 and seld > 0:
                node_pairs.append((nodes[idx], nodes[idx + 1]))
    else:
        raise RuntimeError("TSP failed on slice: {}".format(z))

    if package_path:
        pass
    elif len(node_pairs):
        # Load and process here...
        for pidx, pair in enumerate(nodes[:-1]):
            sel_coor = z_sel_coors[pair, :-1]
            dist_check = adj_mat[nodes[pidx], nodes[pidx + 1]]
            dist_check = np.logical_and(seld < 9, seld > 0)
            vol = load_npz(sel_coor)
            if count == 0:
                tv = rfo(vol)[0]  # Relabel from 0
            elif not dist_check:
                tv = rfo(vol)[0] + max_vox # Relabel from 0 and iterate with running max
            else:
                # Merge here. This is difficult...
                # Get an x-neighbor, a y-neighbor, and a z-neighbor, merge these, let the TSP continue through the slice


                # Can be a z-merge!!
                # Offset and pad PV so that the two overlap
                import ipdb;ipdb.set_trace()
                diff = z_sel_coors[pair, :-1] - z_sel_coors[nodes[pidx - 1], :-1]

                # Merge tv with pv
                tv = drew_consensus(segs=tv, olds=pv)

            # Save per-volume of tv as numpy
            save_tv(tv=tv, sel_coor=sel_coor, config=config, path_extent=path_extent)

            # Iterate and continue
            count += 1
            max_vox = vol.max() + 1
            pv = tv
    else:
        for sel_coor in z_sel_coors:
            sel_coor = sel_coor[:-1]
            vol = load_npz(sel_coor)
            if count == 0:
                tv = rfo(vol)[0]  # Relabel from 0
            else:
                tv = rfo(vol)[0] + max_vox # Relabel from 0 and iterate with running max
            save_tv(tv=tv, sel_coor=sel_coor, config=config, path_extent=path_extent)
            count += 1
            pv = tv

    # --Finish iteration

if package_path:
    np.save('TSP_paths', all_nodes)
