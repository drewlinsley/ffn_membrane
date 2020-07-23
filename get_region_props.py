import sys
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure


def get_props(fn_a, out_a):
    segs = np.load(fn_a)  # np.load('/localscratch/plane_xplus9.npy')
    segs_props = measure.regionprops(segs, cache=False)
    segs_areas = np.array([x.area for x in segs_props])
    segs_ids = np.array([x.label for x in segs_props])
    np.savez(out_a, segs_areas=segs_areas, segs_ids=segs_ids)


if __name__ == '__main__':
    fn_a, out_a = sys.argv[1:]
    print('Region props for {}, {}'.format(fn_a, out_a))
    get_props(fn_a=fn_a, out_a=out_a)

