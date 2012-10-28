import numpy as np
import scipy.ndimage
from matplotlib import pyplot


class EvaluateObj(object):
    def __init__(self, IMGWIDTH, IMGHEIGHT):
        self.IMGWIDTH = IMGWIDTH
        self.IMGHEIGHT = IMGHEIGHT

        self.img = np.zeros((IMGHEIGHT, IMGWIDTH))

    def set_params(self, length, front_size, back_size):
        
        self.length = length
        self.front_size = front_size
        self.back_size = back_size

        
    def render_source(self, x, y, phi, theta):

        front_pos, back_pos = compute_pos(self.length, 
                                          x, y, phi, theta)

        img = np.zeros((2, self.IMGHEIGHT, self.IMGWIDTH), 
                       dtype = np.float32)
        img[0, front_pos[1], front_pos[0]] = self.front_size**2
        img[1, back_pos[1], back_pos[0]] = self.back_size**2

        # render the back
        
        img[0] = scipy.ndimage.filters.gaussian_filter(img[0], 
                                                       self.front_size)
        img[1] = scipy.ndimage.filters.gaussian_filter(img[1], 
                                                       self.back_size)
        
        return np.sum(img, axis=0)


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
