# APPLY VOLUME MERGE ANNOTATION
#
# This script applies a webKnossos merger mode annotation
# to a given segmentation layer. The script will output a
# WKW layer.
#
# The --set_zero flag will relabel all non-annotated segments
# to 0.
#
# 1. Download the merger mode NML file.
# 2. Install Python 3 (if you don't have it)
# 3. Install the dependencies of this script:
#    pip install -U wkw wknml wkcuber
# 4. Run the script from the terminal:
#    python apply_merger_mode.py \
#      /path/to/input_wkw \
#      merger_mode.nml \
#      /path/to/output_wkw
# 5. The script will output a folder with a WKW layer
#
# License: MIT, scalable minds

import wkw
import wknml
import re
from argparse import ArgumentParser
from wkcuber.metadata import read_metadata_for_layer
import numpy as np
from glob import iglob
from os import path, makedirs
import os
import fastremap
from tqdm import tqdm
from numba import njit, jit, prange
from joblib import Parallel, delayed, parallel_backend


def skel_merges(z_start, origin, bbox, ds_in, set_zero, equiv_map_items, zoff=32):
    z_end = min(origin[2] + z_start + zoff, origin[2] + bbox[2])
    offset = (origin[0], origin[1], z_start)
    size = (bbox[0], bbox[1], z_end - z_start)

    # print("Processing cube offset={} size={}".format(offset, size))
    # print("Size: {}".format(size))
    cube_in = ds_in.read(offset, size)[0]

    # cube_out = np.zeros(size, dtype=np.uint32)
    # if not set_zero:
    #     cube_out = np.copy(cube_in)
    # else:
    #     cube_out = np.zeros(size, dtype=np.uint32)
    # # Resahpe cube for speed
    # cube_out_shape = cube_out.shape
    cube_in_shape = cube_in.shape
    # cube_out = cube_out.reshape(-1)
    # cube_in = cube_in.reshape(-1)
    # # equiv_map_items = [(k,v) for k,v in equiv_map.items()]
    cube_in = reassign(cube_in, None, equiv_map_items)
    # cube_out = cube_out.reshape(cube_out_shape)
    ds_out.write(offset, cube_in)
    print("Rewrote cube offset={} size={}".format(offset, size))


@njit(parallel=True, fastmath=True)
def par_reassign(cube_out, cube_id, equiv_map_items):
  el = len(equiv_map_items)
  for idx in prange(el):
      from_id, to_id = equiv_map_items[idx]
      cube_out[cube_in == from_id] = to_id
  return cube_out

def reassign(cube_out, cube_id, equiv_map_items):
  el = len(equiv_map_items)
  mappings = {}
  for idx in range(el):
      from_id, to_id = equiv_map_items[idx]
      mappings[from_id] = to_id
  cube_out = fastremap.remap(cube_out, mappings, preserve_missing_labels=True, in_place=True)
  return cube_out


# Prelude
parser = ArgumentParser(description="Apply webKnossos volume merge annotations")
parser.add_argument(
    "--set_zero",
    action="store_true",
    help="Set non-marked segments to zero.",
)
parser.add_argument("--input", help="Path to input WKW dataset")
parser.add_argument("--layer_name", "-l", help="Segmentation layer name", default="segmentation")
parser.add_argument("--nml", type=str, help="Path to NML file")
parser.add_argument("--output", help="Path to output WKW dataset")
args = parser.parse_args()

_, _, bbox, origin = read_metadata_for_layer(args.input, args.layer_name)

print("Merging merger mode annotations from {} and {}".format(args.input, args.nml))

# Collect equivalence classes from NML
with open(args.nml, "rb") as f:
  nml = wknml.parse_nml(f)

ds_in = wkw.Dataset.open(path.join(args.input, args.layer_name, "1"))
ds_out = wkw.Dataset.create(path.join(args.output, args.layer_name, "1"), wkw.Header(ds_in.header.voxel_type))
cube_size = ds_in.header.block_len * ds_in.header.file_len

equiv_classes = []
assigned_ids = set({})
quarantine_trees = {}
for treeid, tree in enumerate(nml.trees):
    try:
        ids = set(ds_in.read(node.position, (1,1,1))[0,0,0,0] for node in tree.nodes)
        if 0 in ids:
            ids.remove(0)

        # Check if any of the ids have already been assigned to a tree. If so, ignore and add these + the tree to a list.
        used = assigned_ids.intersection(ids)
        if len(used):

            # Filter used from ids -- this is the part you might have to kill
            ids.remove(used)

            # # Update dict with used
            # if treeid in quarantine_trees:
            #     quarantine_trees[treeid] = quarantine_trees[treeid].append(list(used))
            # else:
            #     quarantine_trees[treeid] = list(used)

        # equiv_classes.append(set(ds_in.read(node.position, (1,1,1))[0,0,0,0] for node in tree.nodes))
        if len(ids):
            equiv_classes.append(ids)

            # # Add the assigned ids to the record
            # assigned_ids.update(ids)

    except:
        print("Failed to read {}".format(treeid))

# Save the assigned_ids for curation
np.save("quarantine_trees", quarantine_trees)

# Start merging
equiv_map = {}
for klass in equiv_classes:
  base = next(iter(klass))
  # if base == 0:
  #     # Change base to some high value in 32-bit space
  #     base = 9999999
  if base != 0:
    for id in klass:
      equiv_map[id] = base

print("Found {} equivalence classes with {} nodes".format(len(equiv_classes), len(equiv_map)))
print(equiv_classes)

# Rewrite segmentation layer
_, _, bbox, origin = read_metadata_for_layer(args.input, args.layer_name)

makedirs(args.output, exist_ok=True)

zoff = 32
z_range = range(origin[2], origin[2] + bbox[2], zoff)
equiv_map_items = [(k,v) for k,v in equiv_map.items()]
with Parallel(n_jobs=24, backend='threading') as parallel:
    parallel(delayed(skel_merges)(z_start, origin, bbox, ds_in, args.set_zero, equiv_map_items, zoff=zoff) for z_start in z_range)
# Done
ds_in.close()
print("Rewrote segmentation as a segmentation layer to {}".format(args.output))
print("You may need to copy over additional layers (e.g. color layers) and compress the output segmentation")

