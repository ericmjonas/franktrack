from nose.tools import * 
import numpy as np
import likelihood
from matplotlib import pylab
import util2 as util
import template


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

    i1 = eo.render_source(100, 100, np.pi /2., np.pi/2)
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

def test_likelihood_evaluator2():
    
    tr = template.TemplateRenderCircleBorder()
    tr.set_params(14, 6, 4)

    t1 = tr.render(0, np.pi/2)
    img = np.zeros((240, 320), dtype=np.uint8)

    env = util.Environmentz((1.5, 2.0), (240, 320))
    
    le2 = likelihood.LikelihoodEvaluator2(env, tr, similarity='normcc')

    img[(120-t1.shape[0]/2):(120+t1.shape[0]/2), 
        (160-t1.shape[1]/2):(160+t1.shape[1]/2)] += t1 *255
    pylab.subplot(1, 2, 1)
    pylab.imshow(img, interpolation='nearest', cmap=pylab.cm.gray)

    state = np.zeros(1, dtype=util.DTYPE_STATE)

    xvals = np.linspace(0, 2., 200)
    yvals = np.linspace(0, 1.5, 200)
    res = np.zeros((len(yvals), len(xvals)), dtype=np.float32)
    for yi, y in enumerate(yvals):
        for xi, x in enumerate(xvals):
            state[0]['x'] = x
            state[0]['y'] = y
            state[0]['theta'] = np.pi / 2. 
            res[yi, xi] =     le2.score_state(state, img)
    pylab.subplot(1, 2, 2)
    pylab.imshow(res)
    pylab.colorbar()
    pylab.show()
