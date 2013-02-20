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

import filters


from ruffus import * 

T_DELTA = 1/30.

FL_DATA = "data/fl"

def clear_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

def params():
    EPOCHS = ['bukowski_05.W1', 
              'bukowski_02.W1', 
              'bukowski_02.C', 
              'bukowski_04.W1', 
              'bukowski_04.W2',
              'bukowski_01.linear', 
              'bukowski_01.W1', 
              'bukowski_01.C', 
              'bukowski_05.linear', 
              'Cummings_03.w2', 
              'Cummings_03.linear', 
              ]
    #EPOCHS = [os.path.basename(f) for f in glob.glob("data/fl/*")]
    
    np.random.seed(0)
    FRAMES = [(0, 200)]

    for epoch in EPOCHS:
        for frame_start, frame_end in FRAMES:
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

    #eoparams = enlarge_sep(measure.led_params_to_EO(cf, led_params))

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
        abs_frame_index = frame_pos[fi]

        for thold in THOLDS:
            tholds[thold][fi] = np.sum(frame > thold)

        coordinates.append(skimage.feature.peak_local_max(frame, 
                                                          min_distance=10, 
                                                          threshold_abs=220))

        frame_regions = filters.label_regions(frame)
        regions[fi] = frame_regions
        
    pickle.dump({'frame_pos' : frame_pos, 
                 'tholds' : tholds, 
                 'coordinates' : coordinates, 
                 'regions' : regions}, 
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
    ax_coord_mean = f1.add_subplot(ROWN, 1, 4)
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
    filtered_regions = np.zeros_like(data['regions'])
    filtered_coordinates = []
    for fi, f in enumerate(frame_pos):
        filtered_regions[fi] = filters.filter_regions(regions[fi], 
                                                  size_thold = 3000, 
                                                  max_width = 40,
                                                  max_height=40)
        fc = filters.points_in_mask(filtered_regions[fi] > 0, 
                                                           coordinates[fi])
        filtered_coordinates.append(fc)
        
    ### how many coordinate points
    ax_coord_cnt.plot(frame_pos, [len(x) for x in filtered_coordinates])

    for ax, coord in [(ax_coord_mean, coordinates), 
                      (ax_coord_filt_mean, filtered_coordinates)]:

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
    ax_points.imshow(a, interpolation='nearest')# , cmap=pylab.cm.gray)    
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
