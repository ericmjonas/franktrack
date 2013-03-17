import numpy as np
from scipy import ndimage
import skimage
import filters

import util2 as util
import model
from matplotlib import pylab

"""
Deterministic trackers: 

Point: given an image, output a deterministic estimate of the state variables

Var: given an image, output a deterministic estimate of the state variables as well as a variance / boundary assoicated with each

For each of these, it might be possible to return multiple points? 


    Return an estimate of mean/var for x, y, phi, theta. 
    
    env is the environment, to get pix/dist mapping
    eo_params are diode array params in pixels
    
    Can't return an estimate of velocity (would require multiple frames)


"""

def point_est_track(img, env, eo_params):
    """ 
    finds the filtered peaks, returns mean for x/y. 
    Only returns x,y, no variance or other vars
    """
    
    # min_distance is the dist between pix for peaks. Not sure if this
    # should be a min or a max for the thing

    # size_thold = the largest-sized region we should allow
    size_thold = 2*(eo_params[1] + eo_params[2]) * 1.2
    min_distance = eo_params[0]
    points_of_interest = filters.peak_region_filter(img, 
                                                    min_distance=min_distance, 
                                                    size_thold = size_thold)
    
    if len(points_of_interest) > 0:
        coord_means = env.gc.image_to_real(*np.mean(np.fliplr(points_of_interest), 
                                                    axis=0))
    else:
        coord_means = 0, 0
    return np.array((coord_means[0], 
                     coord_means[1], 
                     0, 0, 0, 0), dtype=model.DTYPE_LATENT_STATE)


def find_possible_front_diodes(img, eo_params, im_reg):
    DIODE_SEP = eo_params[0]
    FRONT_SIZE = float(eo_params[1])
    BACK_SIZE = float(eo_params[2])
    size_thold = (FRONT_SIZE+BACK_SIZE) * 2.5


    im_rf = (im_reg>0).astype(float)*255
    #im_rf = img.copy()
    im_rf[im_reg == 0] = 0
    im_f = ndimage.gaussian_filter(im_rf, FRONT_SIZE)
    coordinates = skimage.feature.peak_local_max(im_f, 
                                                 min_distance=FRONT_SIZE+BACK_SIZE, 
                                                 threshold_rel=0.7)

    # pylab.subplot(1, 3, 1)
    # pylab.imshow(img, interpolation='nearest', cmap=pylab.cm.gray)
    # pylab.subplot(1, 3, 2)
    # pylab.imshow(im_reg, interpolation='nearest')
    # pylab.subplot(1, 3, 3)
    # pylab.imshow(im_f, interpolation='nearest', cmap=pylab.cm.gray)
    # pylab.show()    

    return coordinates

def dist(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def find_possible_back_diodes(img, eo_params, candidate_front_diodes, 
                              im_reg):
    """
    For each candidate front diode returns the list of possible back diodes
    """
    
    DIODE_SEP = eo_params[0]
    FRONT_SIZE = float(eo_params[1])
    BACK_SIZE = float(eo_params[2])

    size_thold = (FRONT_SIZE+BACK_SIZE) * 2.5

    # this is fun, region properties must be increasing
    
    im_reg = filters.canonicalize_regions(im_reg)
    # now remove the possible front diode locations
    props = skimage.measure.regionprops(im_reg)
    out_coords = []
    for c in candidate_front_diodes:
        im_reg_c = im_reg.copy()
        for p in props:
            centroid = p['Centroid']
            d = dist(centroid, c)
            # right now this removes the regions that are too close; 
            # other ideas include erosion or just removing all pix that are too close
            # I don't care about the centroid, I care about whether or not it's the same thing
            if d <= FRONT_SIZE:
                im_reg_c[im_reg == p['Label']] = 0

        im_rf = (im_reg_c>0).astype(float)*255
    
        im_f = ndimage.gaussian_filter(im_rf, BACK_SIZE)
        
        coordinates = skimage.feature.peak_local_max(im_f, 
                                                     min_distance=(FRONT_SIZE+BACK_SIZE), 
                                                     threshold_rel=0.7)
        
        # ax = pylab.subplot(1, 3, 1)
        # pylab.imshow(img, interpolation='nearest', cmap=pylab.cm.gray)
        # ax = pylab.subplot(1, 3, 2)
        # ax.imshow(im_reg_c, interpolation='nearest')
        # circ = pylab.Circle((c[1], c[0]), radius=FRONT_SIZE, 
        #                     color='g')
        # ax.add_patch(circ)
        # ax.plot([c[1]], [c[0]], 'r.')

        # pylab.subplot(1, 3, 3)
        # pylab.imshow(im_f, interpolation='nearest', cmap=pylab.cm.gray)
        # pylab.show()    

        out_coords.append(coordinates)
    return out_coords
    
def filter_plausible_points(front, back_list, max_dist):
    ret = []
    for b in back_list:
        if dist(front, b) <= max_dist:
            ret.append(b)
        else:
            print "dist", dist(front, b), "is >= max_dist", max_dist

    return ret

def point_est_track2(img, env, eo_params):
    """ 
    1. get regions / filter for things that are interesting
    2. 

    Note that we return "candidate points", and if we don't know we return 0
    """
    
    # min_distance is the dist between pix for peaks. Not sure if this
    # should be a min or a max for the thing

    # size_thold = the largest-sized region we should allow
    DIODE_SEP = eo_params[0]
    FRONT_SIZE = float(eo_params[1])
    BACK_SIZE = float(eo_params[2])
    size_thold = (FRONT_SIZE+BACK_SIZE) * 2.5

    im_reg_coarse = filters.extract_region_filter(img, size_thold=size_thold, 
                                                  mark_min=100, mark_max=230)

    im_reg_fine = filters.extract_region_filter(img, size_thold=size_thold, 
                                                mark_min=220, mark_max=240)
    im_reg_fine[im_reg_coarse ==0] = 0
    # pylab.imshow(im_reg_fine)
    # pylab.show()

    min_distance = DIODE_SEP

    
    front_c = find_possible_front_diodes(img, eo_params, im_reg_fine)

    def none():
        return np.zeros(0, dtype=model.DTYPE_LATENT_STATE)


    if len(front_c) > 0:
        back_c = find_possible_back_diodes(img, eo_params, front_c, im_reg_fine)
        print "possible back diodes", back_c
        # just take the first one, because TEST
        plaus_back = filter_plausible_points(front_c[0], back_c[0], 
                                             eo_params[0]*2.0)
        print "plausible back diodes", plaus_back

        if len(plaus_back) > 0:
            a = np.vstack([front_c[0], plaus_back[0]])

            coord_means = env.gc.image_to_real(*np.mean(np.fliplr(a),
                                                        axis=0))
        else:
            return none()
    else:
        return none()

    return np.array([(coord_means[0], 
                     coord_means[1], 
                     0, 0, 0, 0)], dtype=model.DTYPE_LATENT_STATE)



