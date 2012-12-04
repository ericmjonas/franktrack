import numpy as np
import os
import skimage, skimage.measure

import organizedata

def centroid_frame(img, env, thold=255):
    """
    Find the centroid of all > threshold pixels for a single frame

    """
    thold_im = img > thold
    p = skimage.measure.regionprops(thold_im.astype(int))
    y_pix, x_pix = p[0]['Centroid']
    real_x, real_y = env.gc.image_to_real(x_pix, y_pix)
    
    # how to compute confidence? 
    conf = 1.0
    return real_x, real_y, conf

def dual_centroid_frame():
    """
    """
