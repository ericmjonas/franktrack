"""


"""

import numpy as np

def overlap(W1, W2, s):
    """
    return how much W2 overlaps W1 starting at position s
    (s, t] using python edge rules
    
    """
    # this is so gross but is a good first start
    x = np.arange(W1)
    y = np.arange(W2) + s
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
    

