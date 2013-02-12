import numpy as np
import scipy.ndimage

import skimage.measure
import skimage.feature
# import pyximport;

# pyximport.install(setup_args={'include_dirs': np.get_include()})

# import cutil
from matplotlib import pylab
import util2 as util
import template

def pos_to_int(p):
    return np.rint(p).astype(int)


class RenderRegion(object):
    def __init__(self, MAXX = None, MAXY=None):
        self.xregion = (0, 0)
        self.yregion = (0, 0)
        self.MAXX = MAXX
        self.MAXY = MAXY

    def add_x(self, xmin, xmax):
        self.xregion = self.add(self.xregion, xmin, xmax)

    def add_y(self, ymin, ymax):
        self.yregion = self.add(self.yregion, ymin, ymax)

    def get_x_bounded(self):
        return self.get_bounded(self.xregion, self.MAXX)

    def get_y_bounded(self):
        return self.get_bounded(self.yregion, self.MAXY)

    def get_bounded(self, region, maxv):
        if region == None:
            return None
        lower = np.max([region[0], 0])
        if maxv == None:
            upper = region[1]
        else:
            upper = np.min([region[1], maxv])
        return lower, upper

    def add(self, region, xmin, xmax):
        if xmin == xmax:
            return region
        if region[0] == region[1]:
            return (xmin, xmax)

        if xmax-xmin > 0:
            min_val = np.min([region[0], xmin, xmax])
            max_val = np.max([region[1], xmin, xmax])
            return (min_val, max_val)
        return region

    

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
        for (i, size) in [(0, self.front_size), 
                          (1, self.back_size)]:

            self.pre_img[i, pix_max*2, pix_max*2] = 1.0

            # render each one
            self.pre_img[i] = scipy.ndimage.filters.gaussian_filter(self.pre_img[i], size)
            # normalize 
            self.pre_img[i] = self.pre_img[i] / np.max(self.pre_img[i])

            # extend top 
            #self.pre_img[i][self.pre_img[i] > 0.5] = 1.0
            

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


class LikelihoodEvaluator2(object):
    def __init__(self, env, template_obj, similarity = 'dist', 
                 sim_params = None):
        self.env = env
        self.template_obj = template_obj
        self.similarity = similarity
        print "sim_params=", sim_params
        if sim_params == None:
            self.sim_params = {'power' : 2}
        else:
            self.sim_params = sim_params

    def score_state(self, state, img):
        return self.score_state_full(state, img)

    def score_state_full(self, state, img):
        assert len(img.shape)== 2
        x = state['x']
        y = state['y']

        theta = state['theta']
        phi = state['phi']
        x_pix, y_pix = self.env.gc.real_to_image(x, y)
        x_pix = int(x_pix)
        y_pix = int(y_pix)
        template_img = self.template_obj.render(phi, theta)
        template_pix = template_img*255
        img_region, template_region = template.template_select(img, template_pix, 
                                                               x_pix - template_pix.shape[1]/2, 
                                                               y_pix - template_pix.shape[0]/2)
        tr_size = template_region.count()
        if self.similarity == "dist":
            MINSCORE = -1e80
            if tr_size == 0:
                return MINSCORE
            delta = (template_region - img_region.astype(np.float32))
            s = - np.sum((delta)**self.sim_params['power']) 
            s = s / tr_size
        elif self.similarity == "normcc":
            """
            http://en.wikipedia.org/wiki/Cross-correlation#Normalized_cross-correlation
            """
            MINSCORE = 0
            if tr_size == 0:
                return MINSCORE
            template_mean = np.mean(template_region)
            img_mean = np.mean(img_region)
            si = (template_region - template_mean)*(img_region - img_mean)
            s = np.sum(si)
            template_std = np.std(template_region)
            img_std = np.std(img_region)
            if template_std <1e-7:
                return MINSCORE
            if img_std <1e-7:
                return MINSCORE

            s = s / (template_std * img_std)
            s = s / tr_size
            if 'scalar' in self.sim_params:
                s = s * self.sim_params['scalar']
            if not np.isfinite(s):
                return MINSCORE

        return s

class DiodeGeom(object):
    def __init__(self, length, front_radius, back_radius):
        self.length = length
        self.fr = front_radius 
        self.br = back_radius


class LikelihoodEvaluator3(object):
    def __init__(self, env, template_obj, params=None):

        self.env = env
        self.template_obj = template_obj
        self.img_cache = None
        if params == None:
            self.params = {'log' : False, 
                           'power' : 1, 
                           'normalize' : False}
        else:
            self.params = params

    def score_state(self, state, img):
        return self.score_state_full(state, img)

    def score_state_full(self, state, img):
        assert len(img.shape)== 2
        x = state['x']
        y = state['y']
        theta = state['theta']
        phi = state['phi']

        if self.img_cache == None or (self.img_cache != img).any():
            self.img_cache = img.copy()
            self.coordinates = skimage.feature.peak_local_max(img, min_distance=10, 
                                                              threshold_abs=200)
            
        
        # get the points 
        coordinates = self.coordinates
        

        x_pix, y_pix = self.env.gc.real_to_image(x, y)
        x_pix = int(x_pix)
        y_pix = int(y_pix)
        
        front_pos_pix, back_pos_pix = util.compute_pos(self.template_obj.length, 
                                                       x_pix, y_pix, phi, theta)
        front_deltas = np.abs((coordinates - np.array(front_pos_pix)[:2][::-1]))
        front_dists = np.sqrt(np.sum(front_deltas**2, axis=1))

        
        assert len(front_dists) == len(coordinates)



        back_deltas = np.abs((coordinates - np.array(back_pos_pix)[:2][::-1]))
        back_dists = np.sqrt(np.sum(back_deltas**2, axis=1))


        dist_thold = self.params.get('dist-thold', None)
        if dist_thold != None:
            front_deltas= front_deltas[front_deltas < dist_thold]
            back_deltas = back_deltas[back_deltas < dist_thold]
        
        closest_n = self.params.get('closest-n', None)
        if closest_n != None:
            front_deltas = np.sort(front_deltas)[::-1]
            back_deltas = np.sort(back_deltas)[::-1]
            front_deltas= front_deltas[:closest_n]
            back_deltas = back_deltas[:closest_n]
            
        power = self.params.get('power', 1)
        
        delta_sum = np.sum(front_dists**power) + np.sum(back_dists**power)

        if self.params.get('normalize', True):
            delta_sum = delta_sum / len(coordinates)

        if self.params.get('log', False):
            score = -np.log(delta_sum)
        elif self.params.get('exp', False):
            score = -np.exp(delta_sum)
        else:
            score = -delta_sum

        return score

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
