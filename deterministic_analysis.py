import numpy as np
import glob
import scipy.stats
import pandas
import cPickle as pickle
import util2 as util
import organizedata
import measure
import time
from matplotlib import pylab
import os
import skimage.feature
import datasets

import filters


from ruffus import * 

T_DELTA = 1/30.

FL_DATA = "data/fl"

def clear_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

def params():
    # EPOCHS = ['bukowski_05.W1', 
    #           'bukowski_02.W1', 
    #           'bukowski_02.C', 
    #           'bukowski_04.W1', 
    #           'bukowski_04.W2',
    #           'bukowski_01.linear', 
    #           'bukowski_01.W1', 
    #           'bukowski_01.C', 
    #           'bukowski_05.linear', 
    #           'Cummings_03.w2', 
    #           'Cummings_03.linear', 
    #           'Dickinson_02.c', 
    #           'H206.1', 
    #           'H206.2'
    #           ]
    EPOCHS = datasets.CURRENT_EPOCHS
    
    np.random.seed(0)
    FRAMES = datasets.CURRENT_FRAMES

    # for epoch in EPOCHS:
    #     for frame_start, frame_end in FRAMES:
    for epoch, frame_start in datasets.bad():
        frame_end = frame_start + 500
        infile = [os.path.join(FL_DATA, epoch), 
                  os.path.join(FL_DATA, epoch, 'config.pickle'), 
                  os.path.join(FL_DATA, epoch, 'region.pickle'), 
                  os.path.join(FL_DATA, epoch, 'led.params.pickle'), 
                  ]

        outfile = 'deterministic.%s.%d-%d.pickle' % (epoch, frame_start, frame_end)
        yield (infile, outfile, epoch, 
               frame_start, frame_end)
           
 
       
@files(params)
def det_run((epoch_dir, epoch_config_filename, 
            region_filename, led_params_filename), outfile, 
           epoch, 
           frame_start, frame_end):
    np.random.seed(0)
    
    cf = pickle.load(open(epoch_config_filename, 'r'))
    region = pickle.load(open(region_filename, 'r'))
    
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])
    led_params = pickle.load(open(led_params_filename, 'r'))

    eo_params = measure.led_params_to_EO(cf, led_params)
    if frame_end > cf['end_f']:
        frame_end = cf['end_f']

    frame_pos = np.arange(frame_start, frame_end)
    # load frames
    frames = organizedata.get_frames(epoch_dir, frame_pos)

    FRAMEN = len(frames)
    THOLDS = [100, 200, 220, 240]

    tholds = {thold : np.zeros(FRAMEN, dtype=np.uint32) for thold in THOLDS}
    coordinates = []
    
    regions = np.zeros((FRAMEN, frames[0].shape[0], frames[0].shape[1]), 
                       dtype=np.uint8)

    for fi, frame in enumerate(frames):
        print fi, frame_pos[fi]
        abs_frame_index = frame_pos[fi]

        for thold in THOLDS:
            tholds[thold][fi] = np.sum(frame > thold)

        coordinates.append(skimage.feature.peak_local_max(frame, 
                                                          min_distance=30, 
                                                          threshold_abs=220))

        frame_regions = filters.label_regions(frame, mark_min=120, mark_max=220)
        regions[fi] = frame_regions
        
    pickle.dump({'frame_pos' : frame_pos, 
                 'tholds' : tholds, 
                 'coordinates' : coordinates, 
                 'regions' : regions, 
                 'eo_params' : eo_params}, 
                open(outfile, 'w'))


def params_rendered():
    for p in params():
         yield ((p[0][0], p[0][1], p[1]), (p[1] + ".png", 
                                           ), 
                p[2:])


@follows(det_run)
@files(params_rendered)
def det_plot((epoch_dir, epoch_config_filename, results), 
             (all_plot_filename,), other_params):
    
    
    data = np.load(results)
    frame_pos = data['frame_pos']
    FRAMEN = len(frame_pos)

    cf = pickle.load(open(epoch_config_filename))
    truth = np.load(os.path.join(epoch_dir, 'positions.npy'))
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])
    frames = organizedata.get_frames(epoch_dir, frame_pos)

    truth_interp, missing = measure.interpolate(truth)
    derived_truth = measure.compute_derived(truth_interp, 
                                            T_DELTA)


    f1 = pylab.figure(figsize=(32, 8))
    ROWN = 7
    FRAME_SUBSAMPLE = int(FRAMEN / 30)
    ax_frames = f1.add_subplot(ROWN, 1, 1)
    ax_truth = f1.add_subplot(ROWN, 1, 2)
    ax_coord_cnt = f1.add_subplot(ROWN, 1, 3)
    ax_coord_centroid = f1.add_subplot(ROWN, 1, 4)
    ax_coord_filt_mean = f1.add_subplot(ROWN, 1, 5)
    ax_points = f1.add_subplot(ROWN, 1, 6)

    a = np.hstack(frames[::FRAME_SUBSAMPLE])
    ax_frames.imshow(a, interpolation = 'nearest', 
                     vmin=0, vmax=255, cmap=pylab.cm.gray)
    clear_ticks(ax_frames)
    ax_truth.plot(frame_pos, truth_interp[frame_pos]['x'], label='x')
    ax_truth.plot(frame_pos, truth_interp[frame_pos]['y'], label='y')



    coordinates = data['coordinates']
    regions = data['regions']
    eo_params = data['eo_params']
    filtered_regions = np.zeros_like(data['regions'])
    filtered_coordinates = []
    region_props = []
    max_dim = (eo_params[1] + eo_params[2])*1.5
    for fi, f in enumerate(frame_pos):
        filtered_regions[fi] = filters.filter_regions(regions[fi], 
                                                  max_width = max_dim, 
                                                  max_height = max_dim)
        
        fc = filters.points_in_mask(filtered_regions[fi] > 0, 
                                                           coordinates[fi])
        filtered_coordinates.append(fc)
        region_props.append(skimage.measure.regionprops((filtered_regions[fi]>0).astype(int)))
        
    ### how many coordinate points
    ax_coord_cnt.plot(frame_pos, [len(x) for x in filtered_coordinates])
    ax_coord_cnt.plot(frame_pos, [len(x) for x in coordinates])

    ### Cooridnate centroid
    centroids = np.zeros((FRAMEN, 2))
    for rpi, rp in enumerate(region_props):
        if len(rp) > 0 :
            centroid_row, centroid_col = rp[0]['Centroid']
            centroids[rpi] = env.gc.image_to_real(centroid_col, centroid_row)
        
    ax_coord_centroid.plot(frame_pos, truth_interp[frame_pos]['x'], c='b')
    ax_coord_centroid.plot(frame_pos, truth_interp[frame_pos]['y'], c='r')
    ax_coord_centroid.scatter(frame_pos, centroids[:, 0], c='b', linewidth=0, s=4)
    ax_coord_centroid.scatter(frame_pos, centroids[:, 1], c='r', linewidth=0, s=4)
    ax_coord_centroid.set_xlim(np.min(frame_pos), np.max(frame_pos))
    
    for ax, coord in [ (ax_coord_filt_mean, filtered_coordinates)]:
        ### what is the center of mass of the coordinates? 
        coord_means = np.zeros((FRAMEN, 2))
        for i in range(FRAMEN):
            if len(coord[i]) > 0:
                coord_means[i] = env.gc.image_to_real(*np.mean(np.fliplr(coord[i]), 
                                                               axis=0))
        ax.plot(frame_pos, truth_interp[frame_pos]['x'], c='b')
        ax.plot(frame_pos, truth_interp[frame_pos]['y'], c='r')
        ax.scatter(frame_pos, coord_means[:, 0], c='b', linewidth=0, s=4)
        ax.scatter(frame_pos, coord_means[:, 1], c='r', linewidth=0, s=4)
        ax.set_xlim(np.min(frame_pos), np.max(frame_pos))
        
                                 
                                
    # plot the detected points
    a = np.hstack(filtered_regions[::FRAME_SUBSAMPLE])
    ax_points.imshow(a>0, interpolation='nearest', vmin=0, vmax=1, cmap=pylab.cm.gray)    
    for pi, points in enumerate(filtered_coordinates[::FRAME_SUBSAMPLE]):
        FRAME_W = frames[0].shape[1]
        ax_points.plot([p[1]+FRAME_W * pi for p in points], 
                       [p[0] for p in points], 'r.', ms=1.0)
    ax_points.set_xlim([0, a.shape[1]])
    ax_points.set_ylim([0, a.shape[0]])
    ax_points.set_xticks([])
    ax_points.set_yticks([])
    pylab.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    pylab.savefig(all_plot_filename, dpi=300)


pipeline_run([det_run, det_plot], multiprocess=3)
