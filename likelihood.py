import numpy as np
import scipy.ndimage
from matplotlib import pylab
import util

def pos_to_int(p):
    return np.rint(p).astype(int)

class EvaluateObj(object):
    """
    This is slightly misnamed, it's really the rendering object
    to generate the model image

    """
    def __init__(self, IMGHEIGHT, IMGWIDTH):
        self.IMGWIDTH = IMGWIDTH
        self.IMGHEIGHT = IMGHEIGHT

        self.img = np.zeros((IMGHEIGHT, IMGWIDTH))

    def set_params(self, length, front_size, back_size):
        
        self.length = length
        self.front_size = front_size
        self.back_size = back_size

        # for render2, pre-render images
        pix_max = max(front_size, back_size) * 2
        self.pre_img = np.zeros((2, pix_max*4+1, pix_max*4 + 1), 
                                dtype=np.float32)
        self.pre_img[0, pix_max*2, pix_max*2] = self.front_size**2 * 6
        self.pre_img[1, pix_max*2, pix_max*2] = self.back_size**2 * 6

        # render each one
        self.pre_img[0] = scipy.ndimage.filters.gaussian_filter(self.pre_img[0], 
                                                       self.front_size)
        self.pre_img[1] = scipy.ndimage.filters.gaussian_filter(self.pre_img[1], 
                                                       self.back_size)
    def render_source(self, x, y, phi, theta):
        """
        Returns an image where max intensity is 
        1.0, min is 0.0, float32

        Faster version of original rendersource, comparing the two

        The render image is designed to support some minor
        out-of-frame rendering.

        """

        front_pos, back_pos = util.compute_pos(self.length, 
                                               x, y, phi, theta)
        front_pos = pos_to_int(front_pos)
        back_pos = pos_to_int(back_pos)
        # slightly larger 
        tile_size = self.pre_img[0].shape[0]
        tile_side = int(int((tile_size -1))/2)
        border = tile_size
        img = np.zeros((self.IMGHEIGHT + 2*border, self.IMGWIDTH + 2*border),
                       dtype = np.float32)
        front_pix_center_x = border + front_pos[0] 
        front_pix_center_y = border + front_pos[1]

        if img[front_pix_center_y - tile_side:front_pix_center_y+tile_side+1,
               front_pix_center_x - tile_side:front_pix_center_x+tile_side+1].shape == self.pre_img[0].shape:

            img[front_pix_center_y - tile_side:front_pix_center_y+tile_side+1,
                front_pix_center_x - tile_side:front_pix_center_x+tile_side+1] += self.pre_img[0]

        back_pix_center_x = border + back_pos[0] 
        back_pix_center_y = border + back_pos[1]
        if img[back_pix_center_y - tile_side:back_pix_center_y+tile_side+1,
            back_pix_center_x - tile_side:back_pix_center_x+tile_side+1].shape == self.pre_img[1].shape:

            img[back_pix_center_y - tile_side:back_pix_center_y+tile_side+1,
                back_pix_center_x - tile_side:back_pix_center_x+tile_side+1] += self.pre_img[1]


        subimg = img[border:-border, border:-border]
        subimg = np.minimum(subimg, 1.0) # flat[subimg.flat > 1.0] = 1.0 

        assert subimg.shape == (self.IMGHEIGHT, self.IMGWIDTH)
        return subimg


class LikelihoodEvaluator(object):
    def __init__(self, env, evaluate_obj):
        self.env = env
        self.evaluate_obj = evaluate_obj

    def score_state(self, state, img):
        x = state['x']
        y = state['y']


        theta = state['theta']
        phi = state['phi']
        x_pix, y_pix = self.env.gc.real_to_image(x, y)

        proposed_img = self.evaluate_obj.render_source(x_pix, y_pix,
                                                       phi, theta)
        pi_pix = proposed_img*255

        delta = (pi_pix - img.astype(np.float32))
        s = - 8 *  np.log(np.sum(np.abs(delta)))
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