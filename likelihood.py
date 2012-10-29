import numpy as np
import scipy.ndimage
from matplotlib import pylab
import util

def pos_to_int(p):
    return np.rint(p)

class EvaluateObj(object):
    def __init__(self, IMGHEIGHT, IMGWIDTH):
        self.IMGWIDTH = IMGWIDTH
        self.IMGHEIGHT = IMGHEIGHT

        self.img = np.zeros((IMGHEIGHT, IMGWIDTH))

    def set_params(self, length, front_size, back_size):
        
        self.length = length
        self.front_size = front_size
        self.back_size = back_size

    @profile    
    def render_source(self, x, y, phi, theta):
        front_pos, back_pos = util.compute_pos(self.length, 
                                               x, y, phi, theta)
        front_pos = pos_to_int(front_pos)
        back_pos = pos_to_int(back_pos)
        img = np.zeros((2, self.IMGHEIGHT, self.IMGWIDTH), 
                       dtype = np.float32)
        if front_pos[1] < self.IMGHEIGHT and front_pos[0] < self.IMGWIDTH:
            img[0, front_pos[1], front_pos[0]] = self.front_size**2 * 4
        if back_pos[1] < self.IMGHEIGHT and back_pos[0] < self.IMGWIDTH:
            img[1, back_pos[1], back_pos[0]] = self.back_size**2 * 4

        # render the back
        
        img[0] = scipy.ndimage.filters.gaussian_filter(img[0], 
                                                       self.front_size)
        img[1] = scipy.ndimage.filters.gaussian_filter(img[1], 
                                                       self.back_size)
        outimg = np.sum(img, axis=0)
        
        return outimg

class LikelihoodEvaluator(object):
    def __init__(self, env, evaluate_obj):
        self.env = env
        self.evaluate_obj = evaluate_obj
    @profile
    def score_state(self, state, img):
        x = state['x']
        y = state['y']
        theta = state['theta']
        phi = state['phi']
        x_pix, y_pix = self.env.gc.real_to_image(x, y)

        proposed_img = self.evaluate_obj.render_source(x_pix, y_pix,
                                                       phi, theta)
        pi_pix = proposed_img*255
        pi_pix.flat[pi_pix.flat > 255] = 255
        delta = (pi_pix.astype(float) - img.astype(float))
        s = - np.log(np.sum(np.abs(delta)))
        return s

        
if __name__ == "__main__":
    eo = EvaluateObj(320, 240)
    eo.set_params(12, 2, 1)
    x = 160
    y = 120
    phi = 0
    theta = np.pi/2

    front_pos, back_pos = eo.compute_pos(x, y, phi, theta)
    i = eo.render_source(x, y, phi, theta)
    pyplot.imshow(i, cmap=pyplot.cm.gray, interpolation='nearest', 
                  origin='lower')
    pyplot.plot([front_pos[0], back_pos[0]], 
                [front_pos[1], back_pos[1]], linewidth=2, c='r')
    pyplot.show()
