import numpy as np
from scipy import ndimage
import skimage.measure
import skimage.feature
from matplotlib import pylab
import organizedata
from skimage.filter import sobel
from skimage import morphology 

def label_regions(im, mark_min=200, mark_max=230):
    elevation_map = sobel(im)

    markers = np.zeros_like(im)
    markers[im < mark_min] = 1
    markers[im > mark_max] = 2

    segmentation = morphology.watershed(elevation_map, markers)
    segmentation = ndimage.binary_fill_holes(segmentation - 1)

    labeled_regions, _ = ndimage.label(segmentation)
    
    return labeled_regions


def filter_regions(labeled_regions, size_thold = 300, max_width=30, max_height=30):
    """
    filter out the regions that exceed the criteria 
    """
    regions = labeled_regions.copy()
    
    region_cnt = np.max(labeled_regions) 
    for ri in range(1, region_cnt +1):
        region_where = regions== ri
        
        region_size = np.sum(region_where)
        if region_size > size_thold:
            regions[region_where] = 0
        
        x_axis = np.argwhere(np.sum(region_where, axis=0) > 0)
        width = np.max(x_axis) - np.min(x_axis)
        y_axis = np.argwhere(np.sum(region_where, axis=1) > 0)
        height = np.max(y_axis) - np.min(y_axis)

        if height > max_height or width > max_width:
            regions[region_where] = 0
    
    return regions

def points_in_mask(mask, coords):
    coords_ints = np.round(coords).astype(int)
    out_coords = []
    for coord in coords_ints:
        if mask[coord[0], coord[1]] > 0:
            out_coords.append(coord)
    return np.array(out_coords)

def peak_region_filter(img, region_threshold):

    coordinates = skimage.feature.peak_local_max(img, 
                                                 min_distance=30, 
                                                 threshold_rel=0.8)
    img_thold = img > region_threshold
    #pylab.subplot(2, 2, 1)
    #pylab.imshow(img, interpolation='nearest', cmap=pylab.cm.gray)
    #pylab.subplot(2, 2, 2)
    #pylab.imshow(img_thold, interpolation='nearest', cmap=pylab.cm.gray)

    #pylab.subplot(2, 2, 3)
    #pylab.imshow(img, interpolation='nearest', cmap=pylab.cm.gray)
    #pylab.plot([p[1] for p in coordinates], [p[0] for p in coordinates], 'r.')
    #pylab.subplot(2, 2, 4)
    #pylab.imshow(frame_regions)
    
    #pylab.plot([p[1] for p in fc], [p[0] for p in fc], 'r.')

    #pylab.show()


    frame_regions = label_regions(img)
    
    filtered_regions = filter_regions(frame_regions, 
                                      size_thold = 300, 
                                      max_width = 30,
                                      max_height=30)
    fc = points_in_mask(filtered_regions > 0, 
                        coordinates)

    
    return fc
    
