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
    pylab.show()
