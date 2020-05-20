from numpy cimport ndarray as ar
import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def toarr(xy):
    cdef int i, j, h=len(xy), w=len(xy[0]), 
    cdef ar[int,ndim=3] new = np.empty((h,w))
    for i in xrange(h):
        for j in xrange(w):
            new[i,j] = xy[i][j]
    return new

