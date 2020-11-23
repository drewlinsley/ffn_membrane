import numpy as np
from glob import glob


files = glob("fails/fail*")
coordinates = []
for f in files:
    d = np.load(f)
    coor = d["sel_coor"][:-1]
    coordinates.append(coor)
np.save("/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/ffn_membrane_v2/redo_coors", coordinates)
