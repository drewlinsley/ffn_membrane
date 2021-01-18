import os
from glob import glob
import numpy as np
import shutil


files = glob(os.path.join("fails", "*.npy"))
coords = []
for f in files:
    f = f.split(os.path.sep)[-1]
    f = f.split(".")[0]
    f = f.split("_")
    x, y, z = f[1], f[2], f[3]
    coords.append([int(x), int(y), int(z)])
coords = np.asarray(coords)
print(coords)
np.save("failed_coord_array", coords)
shutil.copy("failed_coord_array.npy", "/media/data_cifs/projects/prj_connectomics/ffn_membrane_v2/redo_coords.npy")

