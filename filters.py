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
    print coords.dtype, coords.shape
    coords_ints = np.round(coords).astype(int)
    out_coords = []
    for coord in coords_ints:
        if mask[coord[0], coord[1]] > 0:
            out_coords.append(coord)
    return np.array(out_coords)
