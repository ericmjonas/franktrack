from nose.tools import * 
import numpy as np
import likelihood
from matplotlib import pylab



def test_render_source_coords():
    W = 320
    H = 240
    eo = likelihood.EvaluateObj(H, W)
    eo.set_params(14, 5, 3)

    for x in range(0, W, 20):
        for y in range(0, H, 20):
            for phi in np.arange(0, 4*np.pi, 4):
                for theta in np.arange(0, np.pi, 4):
                    i1 = eo.render_source(x, y, phi, theta)

def test_render_at_all():
    W = 320
    H = 240
    eo = likelihood.EvaluateObj(H, W)
    eo.set_params(14, 5, 3)

    i1 = eo.render_source(100, 100, 0, np.pi/2)
    pylab.imshow(i1, interpolation='nearest', origin='lower', 
                 vmin=0, vmax=1.0, cmap=pylab.cm.gray)
    #pylab.show()

def test_interval():
    rr = likelihood.RenderRegion(100, 200)
    rr.add_x(0, 0)
    rr.add_y(0, 0)
    
    assert_equal(rr.get_x_bounded(), (0, 0))
    assert_equal(rr.get_y_bounded(), (0, 0))

    rr.add_x(4, 10)
    rr.add_y(8, 12)
    
    assert_equal(rr.get_x_bounded(), (4, 10))
    assert_equal(rr.get_y_bounded(), (8, 12))

    rr.add_x(10, 14)
    rr.add_y(12, 18)
    
    assert_equal(rr.get_x_bounded(), (4, 14))
    assert_equal(rr.get_y_bounded(), (8, 18))

    rr.add_x(-7, 14)
    rr.add_y(-5, 18)
    
    assert_equal(rr.get_x_bounded(), (0, 14))
    assert_equal(rr.get_y_bounded(), (0, 18))

    rr.add_x(-7, 150)
    rr.add_y(-5, 250)
    
    assert_equal(rr.get_x_bounded(), (0, 100))
    assert_equal(rr.get_y_bounded(), (0, 200))
