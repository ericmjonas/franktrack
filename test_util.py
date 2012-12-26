import numpy as np
from numpy.testing import * 
from nose.tools import *
import util2 as util

def test_geom_converter():
    gc = util.GeomConverter((320, 240), 
                            (160, 120), 
                            (-1.0, -1.0))
    x, y = gc.real_to_image(0, 0)
    assert_almost_equal(x, 120)
    assert_almost_equal(y, 160)
    x, y = gc.real_to_image(-1, -1)
    assert_almost_equal(x, 0)
    assert_almost_equal(y, 0)
    

    gc = util.GeomConverter((320, 240), 
                            (160, 120), 
                            (-1.0, -1.5))
    x, y = gc.real_to_image(0, 0)
    assert_almost_equal(x, 180)
    assert_almost_equal(y, 160)


def test_extract_region():
    
    R, C = 4, 6
    
    x = np.arange(R*C)
    x.shape = (R, C)
    
    for r in range(R):
        for c in range(C):
            s = util.extract_region_safe(x, r, c, 0, 17)
            assert_equal(s.shape,  (1, 1))
            assert_equal(s[0, 0], x[r, c])

    # upper left
    s = util.extract_region_safe(x, 0, 0, 1, 17)
    assert_equal(s.shape, (3, 3))
    assert_array_equal(s[0, :], 17)
    assert_array_equal(s[:, 0], 17)
    assert_array_equal(s[1:,1:], x[:2, :2])

    # upper right
    s = util.extract_region_safe(x, 0, 5, 1, 17)
    assert_equal(s.shape, (3, 3))
    assert_array_equal(s[0, :], 17)
    assert_array_equal(s[:, 2], 17)
    assert_array_equal(s[1:,:2], x[:2, 4:])


    # lower left
    s = util.extract_region_safe(x, 3, 0, 1, 17)
    assert_equal(s.shape, (3, 3))
    assert_array_equal(s[2, :], 17)
    assert_array_equal(s[:, 0], 17)
    assert_array_equal(s[:2,1:], x[2:, :2])

    # lower right
    s = util.extract_region_safe(x, 3, 5, 1, 17)
    assert_equal(s.shape, (3, 3))
    assert_array_equal(s[2, :], 17)
    assert_array_equal(s[:, 2], 17)
    assert_array_equal(s[:2,:2], x[2:, 4:])


    # now we move the window around for a variety of edge sizes, 
    # save the middle pixel, and reconstruct the matrix and compare

    for p in range(4):
        newx= np.zeros_like(x)
        for r in range(R):
            for c in range(C):
                s = util.extract_region_safe(x, r, c, p, 17)
                newx[r, c] = s[p, p]
        assert_array_equal(newx, x)
