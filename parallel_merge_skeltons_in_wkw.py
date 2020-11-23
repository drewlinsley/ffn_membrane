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
for tree in nml.trees:
    try:
        equiv_classes.append(set(ds_in.read(node.position, (1,1,1))[0,0,0,0] for node in tree.nodes))
    except:
        import pdb;pdb.set_trace()

equiv_map = {}
for klass in equiv_classes:
  base = next(iter(klass))
  for id in klass:
    equiv_map[id] = base

print("Found {} equivalence classes with {} nodes".format(len(equiv_classes), len(equiv_map)))
print(equiv_classes)

# Rewrite segmentation layer
_, _, bbox, origin = read_metadata_for_layer(args.input, args.layer_name)

makedirs(args.output, exist_ok=True)
z_range = range(origin[2], origin[2] + bbox[2], 32)
with Parallel(n_jobs=32, backend='threading') as parallel:
# with Parallel(n_jobs=32, backend='loky') as parallel:
    for z_start in tqdm(z_range, total=z_range, desc="Z-slice main clock"):

    z_end = min(origin[2] + z_start + 32, origin[2] + bbox[2])
  offset = (origin[0], origin[1], z_start)
  size = (bbox[0], bbox[1], z_end - z_start)

  print("Processing cube offset={} size={}".format(offset, size))
  cube_in = ds_in.read(offset, size)[0]

  cube_out = np.zeros(size, dtype=np.uint32)
  if not args.set_zero:
      cube_out[:, :, :] = cube_in
  for from_id, to_id in equiv_map.items():
      cube_out[cube_in == from_id] = to_id
      # print("Applied {} -> {}".format(from_id, to_id))

  ds_out.write(offset, cube_out)
  print("Rewrote cube offset={} size={}".format(offset, size))

# Done
ds_in.close()
print("Rewrote segmentation as a segmentation layer to {}".format(args.output))
print("You may need to copy over additional layers (e.g. color layers) and compress the output segmentation")

