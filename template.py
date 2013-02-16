"""


"""

import numpy as np
import numpy.ma as ma
import util2 as util
import scipy.ndimage

def overlap(W1, W2, s):
    """
    return how much W2 overlaps W1 starting at position s
    (s, t] using python edge rules
    
    """
    # this is so gross but is a good first start
    x = np.arange(int(W1))
    y = np.arange(int(W2)) + int(s)
    z = np.intersect1d(x, y)
    if len(z) == 0:
        return (0, 0)
    
    return z[0], z[-1]+1

def template_select(image, template, temp_x, temp_y):
    """
    return a view of the original image and a view of the template
    that is their overlap
    
    note temp_x, temp_y = the location of the template in original-image coordinates
    temp_x, temp_y = (0, 0) : template upper-left pixel and image upper-left pixel are aligned

    temp_x, temp_y can be over ANY region, but note that this might result in 
    the returned images having zero pixels


    """
    IMG_R, IMG_C = image.shape[0], image.shape[1]
    T_R, T_C = template.shape[0], template.shape[1]
    # python's somewhat complex indexing rules, while normally our friend, 
    # make this more confusing 

    x1, x2 = overlap(IMG_C, T_C, temp_x)
    y1, y2 = overlap(IMG_R, T_R, temp_y)

    # select the base image region
    img_region = image[y1:y2, x1:x2]

    if temp_x >= 0:
        t_x1 = 0
    else:
        t_x1 = -temp_x

    t_x2 = t_x1 + (x2-x1)

    if temp_y >= 0:
        t_y1 = 0
    else:
        t_y1 = -temp_y

    t_y2 = t_y1 + (y2-y1)


    template_region = template[t_y1:t_y2, t_x1:t_x2]
    
    return img_region, template_region
    
class TemplateRenderGaussian(object):
    """
    Render target as two gaussian blobs. 
    Center of returned image is diode center
    There's actually no reason we can't just cache all of these
    (360 * 180 == not very many)
    """
    def __init__(self):
        pass

    def set_params(self, length, front_size, back_size):
        
        self.length = length
        self.front_size = front_size
        if back_size < 1:
            raise Exception("Too small of backsize")
        self.back_size = back_size
    
    def render(self, phi, theta):
        """
        Returns a template where max intensity is 
        1.0, min is 0.0, float32. The center of the 
        returned image is the center of the diode array

        """
        
        s = max(self.front_size, self.back_size)
        template = np.zeros((2, self.length + 4*s, 
                             self.length + 4*s))
        D, W, H = template.shape
        
        front_pos, back_pos = util.compute_pos(self.length, 
                                               W/2., H/2., phi, theta)
        
        def pos_to_int(p):
            return np.rint(p).astype(int)

        front_pos = pos_to_int(front_pos)
        back_pos = pos_to_int(back_pos)
            
        template[0, front_pos[1], front_pos[0]] = 1.0
        template[1, back_pos[1], back_pos[0]] = 1.0

        template[0] = scipy.ndimage.filters.gaussian_filter(template[0], 
                                                            self.front_size)
        template[0] = template[0] / np.max(template[0])
        template[0][template[0]<0.2] = 0.0


        template[1] = scipy.ndimage.filters.gaussian_filter(template[1], 
                                                            self.back_size)
        template[1] = template[1] / np.max(template[1])
        template[1][template[1]<0.2] = 0.0

        t = np.sum(template, axis=0)
        t = t / np.max(t)
        tm = np.ma.masked_less(t, 0.0001)
        return tm

        
class TemplateRenderCircleBorder(object):
    """
    """
    def __init__(self):
        pass

    def set_params(self, length, front_size, back_size):
        
        self.length = length
        self.front_size = front_size
        if back_size < 1:
            raise Exception("Too small of backsize")
        self.back_size = back_size
    
    def render(self, phi, theta):
        """
        Returns a template where max intensity is 
        1.0, min is 0.0, float32. The center of the 
        returned image is the center of the diode array

        """
        
        s = max(self.front_size, self.back_size)
        T_D = self.length + 4*s
        template = np.ma.zeros((2, T_D, 
                             T_D), dtype=np.float32)
        D, W, H = template.shape
        
        front_pos, back_pos = util.compute_pos(self.length, 
                                               W/2., H/2., phi, theta)
        
        def pos_to_int(p):
            return np.rint(p).astype(int)

        front_pos = pos_to_int(front_pos)
        back_pos = pos_to_int(back_pos)
        BORDER = 0.6
        for i, size, pos in [(0, self.front_size, front_pos), 
                             (1, self.back_size, back_pos)]:
            template[i] = util.render_hat_ma_fast(H, W, pos[1], pos[0], 
                                                  size, BORDER)

        t =  np.sum(template, axis=0)

        return t

        
        
class PointCloudCount(object):
    """
    A better model, which takes in a point cloud and then counts:
    1. number of points < R_front
    2. number of points < R_back
    3. number of points not in either set that are within the right dist

    """
    def __init__(self, front_xy, front_r, 
                     back_xy, back_r):
        self.front_xy = front_xy
        self.front_r = front_r
        self.back_xy = back_xy
        self.back_r = back_r


    def get_points(self, pointcloud):
        """
        returns front_point_index, back_point_index, between_index
        """
        
        center = np.mean([self.front_xy, self.back_xy], axis=0)

        def dist(xy1, xy2):
            return np.sqrt(np.sum((xy1 - xy2)**2.))

        r = dist(center, self.front_xy)
        front_point_index = []
        back_point_index = []
        between_index = []

        for pi, p in enumerate(pointcloud):
            # FIXME : first check if too far away, will require only
            # a single comparison
            if dist(self.front_xy, p) < self.front_r:
                front_point_index.append(pi)
            elif dist(self.back_xy, p) < self.back_r:
                back_point_index.append(pi)
            elif dist(center, p) < r:
                between_index.append(pi)

        return (np.array(front_point_index), 
                np.array(back_point_index), 
                np.array(between_index))
                
        
    
    
