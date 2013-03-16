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


def filter_regions(labeled_regions, max_width=30, max_height=30):
    """
    filter out the regions that exceed the criteria 
    """
    regions = labeled_regions.copy()
    
    region_cnt = np.max(labeled_regions) 
    for ri in range(1, region_cnt +1):
        region_where = regions== ri
        
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

def peak_region_filter(img, min_distance=30, threshold_rel=0.8, 
                       min_mark = 120, 
                       max_mark = 200, 
                       size_thold = 30):

    coordinates = skimage.feature.peak_local_max(img, 
                                                 min_distance=min_distance, 
                                                 threshold_rel=threshold_rel)
    
    frame_regions = label_regions(img, mark_min=min_mark, mark_max=max_mark)
    
    filtered_regions = filter_regions(frame_regions, 
                                      max_width = size_thold,
                                      max_height= size_thold)
    fc = points_in_mask(filtered_regions > 0, 
                        coordinates)
    
    return fc
    
def extract_region_filter(img, size_thold,
                          mark_min=200, mark_max=230):
    """
    For a given image, use watershed/thresholding to extract out the regions
    and then return the segmented, filtered regions
    
    """

    frame_regions = label_regions(img, mark_min=mark_min, 
                                  mark_max = mark_max)
    
    filtered_regions = filter_regions(frame_regions, 
                                      max_width = size_thold,
                                      max_height = size_thold)
    return filtered_regions
    


def canonicalize_regions(im_regions):
    """
    Take the region labels and turn them into a canonical set 
    of 0, 1, 2, etc. 
    """
    

    unique_regions = np.sort(np.unique(im_regions))
    region_n = len(unique_regions)
    missing = np.setdiff1d(np.arange(region_n), unique_regions)
    new_r = im_regions.copy()
    for m, u in zip(missing, unique_regions[-(len(missing)):]):
        new_r[new_r == u] =m
    return new_r

