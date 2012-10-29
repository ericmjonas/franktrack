from nose.tools import *
import util

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
