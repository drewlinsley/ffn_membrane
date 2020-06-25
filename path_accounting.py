import os
import re
from tqdm import tqdm
import numpy as np
from utils.hybrid_utils import pad_zeros
from glob2 import glob


main_path = '/media/data_cifs/connectomics/mag1_segs'
merge_path = '/media/data_cifs/connectomics/mag1_merge_segs'
main_str = os.path.join(main_path, '/x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.nii')
merge_str = os.path.join(merge_path, '/x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.nii')
raw_paths = np.load('og_paths.npy')
extents = [9, 9, 3]

coords = []
for r in raw_paths:
    x = r.split(os.path.sep)[5]
    y = r.split(os.path.sep)[6]
    z = r.split(os.path.sep)[7]
    x = int(re.split('0*([1-9][0-9]*|0)', x)[1])
    y = int(re.split('0*([1-9][0-9]*|0)', y)[1])
    z = int(re.split('0*([1-9][0-9]*|0)', z)[1])
    coords.append([x, y, z])
coords = np.array(coords)

# Get the main coords connectivities (use the npzs from FFN)
if not os.path.exists("main_paths.npy"):
    main_paths = glob(os.path.join(main_path, "**", "**", "**", "0", "0", "*.npz")) 
    merge_paths = glob(os.path.join(merge_path, "**", "**", "**", "0", "0", "*.npz"))
    np.save("main_paths", main_paths)
    np.save("merge_paths", merge_paths)
else:
    main_paths = np.load("main_paths.npy")
    merge_paths = np.load("merge_paths.npy")


# Get the merge coords connectivities
"""



main_check, merge_check = np.full(len(coords), False, dtype=bool), np.full(len(coords), False, dtype=bool)
for idx, sel_coor in tqdm(enumerate(coords), total=len(coords)):
    if os.path.exists(main_str.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4), pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4))):
        main_check[idx] = True
    # else:
    #     main_check.append(False)
    if os.path.exists(merge_str.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4), pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4))):
        merge_check[idx] = True
    # else:
    #     merge_check.append(False)
main_check = np.array(main_check)
merge_check = np.array(merge_check)
np.savez('path_accounting', main_check=main_check, merge_check=merge_check) 
"""
