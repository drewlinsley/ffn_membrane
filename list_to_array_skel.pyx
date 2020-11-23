from numpy cimport ndarray as ar
import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef toarr(list xy):
    cdef int i, j, n=len(xy), h=len(xy[0]), w=len(xy[0][0]), d=len(xy[0][0][0])
    cdef ar[int, ndim=4] new = np.empty((n, h, w, d), dtype=np.int32)  # np.int32
    for i in xrange(n):
        new[i,:,:,:] = xy[i]
    return new

