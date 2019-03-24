import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import regionprops as rgp


v1 = np.load('seg-0_0_0.npz')['segmentation']
v2 = np.load('ding_segmentations/x0015/y0015/z0017/v0/0/0/seg-0_0_0.npz')['segmentation']
bg = 0

# 1. Count size of all segments in v1 and v2
v1segs = rgp(v1)
v1segs = [x for x in v1segs if x.label != bg]
v2segs = rgp(v2)
v2segs = [x for x in v2segs if x.label != bg]

# 2. Sort segs by their areas
area_v1 = np.array([x.area for x in v1segs])
area_v2 = np.array([x.area for x in v2segs])
v1_idx = np.argsort(area_v1)[::-1]
v2_idx = np.argsort(area_v2)[::-1]
group_idx = np.zeros((len(v1_idx), len(v2_idx)))
combined_areas = np.concatenate((area_v1, area_v2))
combined_idx = np.argsort(combined_areas)




