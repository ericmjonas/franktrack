import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
def score(np.ndarray[np.float32_t, ndim=2] proposed, 
          np.ndarray[np.uint8_t, ndim=2] img):

    cdef int rmax = proposed.shape[0]
    cdef int cmax = proposed.shape[1]

    cdef float sd = 0.0
    cdef float p 
    for r in range(rmax):
        for c in range(cmax):
            p = img[r, c]
            sd += abs(proposed[r, c]*255 - p)

    return -sd


def score_slow(np.ndarray[np.float32_t, ndim=2] proposed, 
          np.ndarray[np.uint8_t, ndim=2] img):

    pi_pix = proposed*255

    delta = (pi_pix - img.astype(np.float32))
    s = - np.sum(np.abs(delta))

    return s

#@cython.boundscheck(False)
def frame_hist_add(np.ndarray[np.float32_t, ndim=3] hist, 
                   np.ndarray[np.uint8_t, ndim=2] img):

    cdef int rmax = hist.shape[0]
    cdef int cmax = hist.shape[1]
    cdef int p
    
    for r in range(rmax):
        for c in range(cmax):
            p = img[r, c]
            hist[r, c, p] += 1
