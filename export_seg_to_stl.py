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
from joblib import Parallel, delayed, parallel_backend
import fastremap
from stl import mesh


def skel_merges(z_start, origin, bbox, ds_in, set_zero, seg_ids, vol, zoff=32):
    z_end = min(origin[2] + z_start + zoff, origin[2] + bbox[2])
    offset = (origin[0], origin[1], z_start)
    size = (bbox[0], bbox[1], z_end - z_start)

    cube_in = ds_in.read(offset, size)[0]
    cube_in_shape = cube_in.shape
    cube_in = fastremap.mask_except(cube_in, seg_ids, in_place=True)
    vol[..., z_start: z_end] = cube_in
    del cube_in
    

# Prelude
parser = ArgumentParser(description="Apply webKnossos volume merge annotations")
parser.add_argument(
    "--set_zero",
    action="store_true",
    help="Set non-marked segments to zero.",
)
parser.add_argument("--input", help="Path to input WKW dataset")
parser.add_argument("--layer_name", "-l", help="Segmentation layer name", default="segmentation")
parser.add_argument("--seg_ids", type=str, help="Comma delimeted list of IDs to extract and convert to stl")
parser.add_argument("--output", help="Path to output WKW dataset")
args = parser.parse_args()

_, _, bbox, origin = read_metadata_for_layer(args.input, args.layer_name)

ds_in = wkw.Dataset.open(path.join(args.input, args.layer_name, "1"))
cube_size = ds_in.header.block_len * ds_in.header.file_len
seg_ids = args.seg_ids.split(",")
seg_ids = [int(x) for x in seg_ids]

# Rewrite segmentation layer
_, _, bbox, origin = read_metadata_for_layer(args.input, args.layer_name)

makedirs(args.output, exist_ok=True)

zoff = 32
z_range = range(origin[2], origin[2] + bbox[2], zoff)
vol = np.zeros(bbox, dtype=np.uint8)
# with Parallel(n_jobs=20, max_nbytes=None, require='sharedmem') as parallel:  # backend='threading',
with Parallel(n_jobs=1, max_nbytes=None) as parallel:  # backend='threading',
    parallel(delayed(skel_merges)(z_start, origin, bbox, ds_in, args.set_zero, vol=vol, seg_ids=seg_ids, zoff=zoff) for z_start in z_range)

# Done
ds_in.close()

# Temp save data as numpy
np.save("pre_stl", vol)
msh = mesh.Mesh(vol, remove_empty_areas=True)
msh.save("{}.stl".format(args.seg_ids))

