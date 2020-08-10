import os
import re
import wkw
import numpy as np
from glob2 import glob
from tqdm import tqdm
from joblib import Parallel, delayed


def process_files(f, dataset):
    """Function for processing each file."""
    # for f in tqdm(files, total=len(files), desc="Writing {}".format(out_dir)):
    offsets = f.split(os.path.sep)
    offset_x = offsets[-4]
    offset_y = offsets[-3]
    offset_z = offsets[-2]
    offset_x = int(re.split('0*([1-9][0-9]*|0)', offset_x)[1])
    offset_y = int(re.split('0*([1-9][0-9]*|0)', offset_y)[1])
    offset_z = int(re.split('0*([1-9][0-9]*|0)', offset_z)[1])
    offset_array = np.asarray((offset_x, offset_y, offset_z))
    adj_offsets = offset_array * 128
    if proc_type == "raw":
        data = np.fromfile(f, dtype=dtype).reshape(reshape)
        dataset.write(tuple(adj_offsets), data)
        del data
    else:
        raise NotImplementedError("Your processing type is not implemented yet.")


# Config
image_dir = "/media/data_cifs/connectomics/mag1"
out_dir = "/media/data_cifs/connectomics/cubed_mag1_manual/1"
match = "x*/**/**/*.raw"
dtype = np.uint8
proc_type = "raw"
reshape = (128, 128, 128)
store_globs = True
tmp_file = "tmp_globs_wkw.npy"

# Get files
if store_globs:
    if os.path.exists(tmp_file):
        files = np.load(tmp_file)
    else:
        files = glob(os.path.join(image_dir, match))
        np.save(tmp_file, files)
else:
    files = glob(os.path.join(image_dir, match))
if not len(files):
    raise RuntimeError("No files found at: {}".format(os.path.join(image_dir, match)))

# Create dataset
dataset = wkw.Dataset.open(
    out_dir,
    wkw.Header(dtype))

# Load and write files
with Parallel(n_jobs=-32, backend='threading') as parallel:
    parallel(delayed(process_files)(f, dataset) for f in tqdm(files, total=len(files), desc="Writing {}".format(out_dir)))

