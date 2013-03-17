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
import dettrack

import filters
import model

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
    for epoch, frame_start in datasets.all():
        frame_end = frame_start + 10
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

    truth = np.load(os.path.join(epoch_dir, 'positions.npy'))
    truth_interp, missing = measure.interpolate(truth)
    derived_truth = measure.compute_derived(truth_interp, 
                                            T_DELTA)


    frame_pos = np.arange(frame_start, frame_end)
    # load frames
    frames = organizedata.get_frames(epoch_dir, frame_pos)

    FRAMEN = len(frames)

    coordinates = []
    
    regions = np.zeros((FRAMEN, frames[0].shape[0], frames[0].shape[1]), 
                       dtype=np.uint8)
    point_est_track_data = []

    for fi, frame in enumerate(frames):
        abs_frame_index = frame_pos[fi]

        coordinates.append(skimage.feature.peak_local_max(frame, 
                                                          min_distance=30, 
                                                          threshold_abs=220))

        frame_regions = filters.label_regions(frame, mark_min=120, mark_max=220)
        regions[fi] = frame_regions
        point_est_track_data.append(dettrack.point_est_track2(frame, env, eo_params))

    pickle.dump({'frame_pos' : frame_pos, 
                 'coordinates' : coordinates, 
                 'regions' : regions, 
                 'eo_params' : eo_params, 
                 'point_est_track' : point_est_track_data, 
                 'truth' : truth, 
                 'missing' : missing, 
                 'truth_interp' : truth_interp, 
                 'derived_truth' : derived_truth, 
                 'epoch' : epoch, 
                 'frame_start' : frame_start}, 
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
    FRAME_SUBSAMPLE = int(np.ceil(FRAMEN / 30.0))
    ax_frames = f1.add_subplot(ROWN, 1, 1)
    ax_truth = f1.add_subplot(ROWN, 1, 2)
    ax_coord_cnt = f1.add_subplot(ROWN, 1, 3)
    ax_phi_theta = f1.add_subplot(ROWN, 1, 4)
    ax_coord_filt_mean = f1.add_subplot(ROWN, 1, 5)
    ax_img_pos = f1.add_subplot(ROWN, 1, 6)

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
    point_est_track = data['point_est_track']
    region_props = []
    max_dim = (eo_params[1] + eo_params[2])*2*1.2
    for fi, f in enumerate(frame_pos):
        filtered_regions[fi] = filters.filter_regions(regions[fi], 
                                                  max_width = max_dim, 
                                                  max_height = max_dim)
        
        fc = filters.points_in_mask(filtered_regions[fi] > 0, 
                                                           coordinates[fi])
        region_props.append(skimage.measure.regionprops((filtered_regions[fi]>0).astype(int)))
        
    ### how many coordinate points
    #ax_coord_cnt.plot(frame_pos, [len(x) for x in filtered_coordinates])
    ax_coord_cnt.plot(frame_pos, [len(x) for x in coordinates])

    ### Cooridnate centroid
    centroids = np.zeros((FRAMEN, 2))
    for rpi, rp in enumerate(region_props):
        if len(rp) > 0 :
            centroid_row, centroid_col = rp[0]['Centroid']
            centroids[rpi] = env.gc.image_to_real(centroid_col, centroid_row)
    ax_phi_theta.plot(frame_pos, derived_truth['phi'][frame_pos] % (2*np.pi))
    # ax_coord_centroid.plot(frame_pos, truth_interp[frame_pos]['x'], c='b')
    # ax_coord_centroid.plot(frame_pos, truth_interp[frame_pos]['y'], c='r')
    # ax_coord_centroid.scatter(frame_pos, centroids[:, 0], c='b', linewidth=0, s=4)
    # ax_coord_centroid.scatter(frame_pos, centroids[:, 1], c='r', linewidth=0, s=4)
    # ax_coord_centroid.set_xlim(np.min(frame_pos), np.max(frame_pos))
    coord = point_est_track
    have_est = np.array([p.shape[0] > 0 for p in coord])
    frame_pos_pres = frame_pos[have_est]
    frame_pos_abs = frame_pos[np.logical_not(have_est)]

    coord_x = [coord[i][0]['x'] for i in np.argwhere(have_est)]
    coord_y = [coord[i][0]['y'] for i in np.argwhere(have_est)]
    coord_phi = np.array([coord[i][0]['phi'] for i in np.argwhere(have_est)])
    print coord_phi
    # plot x/y
    ax_coord_filt_mean.plot(frame_pos, truth_interp[frame_pos]['x'], c='b')
    ax_coord_filt_mean.plot(frame_pos, truth_interp[frame_pos]['y'], c='r')
    ax_coord_filt_mean.scatter(frame_pos_pres, coord_x, c='b', linewidth=0, s=4)
    ax_coord_filt_mean.scatter(frame_pos_pres, coord_y, c='r', linewidth=0, s=4)
    for f in frame_pos_abs:
        ax_coord_filt_mean.axvline(f, c='k', alpha=0.5)
    ax_coord_filt_mean.set_xlim(np.min(frame_pos), np.max(frame_pos))
    #plot phi
    ax_phi_theta.scatter(frame_pos_pres, coord_phi % (2*np.pi))
    ax_phi_theta.set_xlim(np.min(frame_pos), np.max(frame_pos))
                                
    # plot the detected points
    a = np.hstack(filtered_regions[::FRAME_SUBSAMPLE])
    FRAME_W = frames[0].shape[1]
    a_th = (a > 0 ).astype(np.uint8)*255
    a[:, ::FRAME_W] = 200

    ax_img_pos.imshow(a>0, interpolation='nearest', vmin=0, vmax=1, cmap=pylab.cm.gray)    
    for pi, point in enumerate(point_est_track[::FRAME_SUBSAMPLE]):
        # convert back to pix
        pix_x, pix_y = env.gc.real_to_image(point['x'], point['y'])

        ax_img_pos.plot([FRAME_W * pi + pix_x], 
                        [pix_y], 'r.', ms=1.0)

    ax_img_pos.set_xlim([0, a.shape[1]])
    ax_img_pos.set_ylim([0, a.shape[0]])
    ax_img_pos.set_xticks([])
    ax_img_pos.set_yticks([])
    pylab.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    pylab.savefig(all_plot_filename, dpi=300)


@merge(det_run, 'deterministic.results.pickle')
def aggregate_results(det_run, outfilename):
    summary = {}
    for infile in det_run:
        print "opening", infile
        d = pickle.load(open(infile))
        epoch = d['epoch']
        point_est_track = d['point_est_track']
        frame_pos = d['frame_pos']

        have_est = np.array([p.shape[0] > 0 for p in point_est_track])
        frame_pos_pres = frame_pos[have_est]
        frame_pos_abs = frame_pos[np.logical_not(have_est)]
        
        frame_idx_pres = np.argwhere(have_est).flatten()
        coord_x = np.array([point_est_track[i][0]['x'] for i in frame_idx_pres])
        true_x = d['truth_interp'][frame_pos_pres]['x']
        
        coord_y = np.array([point_est_track[i][0]['y'] for i in frame_idx_pres])
        true_y = d['truth_interp'][frame_pos_pres]['y']
        
        error_xy = np.sqrt((coord_x-true_x)**2 + (coord_y-true_y)**2)
        # print error_xy.shape, error_xy.dtype, np.mean(error_xy), np.max(error_xy)
        # pylab.subplot(3, 1, 1)
        # pylab.plot(frame_pos_pres, error_xy)
        # pylab.subplot(3, 1, 2)
        # pylab.plot(frame_pos_pres, coord_x)
        # pylab.plot(frame_pos_pres, true_x)
        # pylab.subplot(3, 1, 3)
        # pylab.plot(frame_pos_pres, coord_y)
        # pylab.plot(frame_pos_pres, true_y)
        
        # pylab.show()
        error_xy_mean = np.mean(error_xy)
        error_xy_median = np.median(error_xy)
        if len(error_xy) > 0:
            error_xy_max = np.max(error_xy)
        else:
            error_xy_max = 0.0
        
        summary["%s.%d" % (epoch,  frame_pos[0])] = {'frame_pos_pres' : len(frame_pos_pres), 
                                                     'frame_n' : len(frame_pos), 
                                                     'error_xy_mean' : error_xy_mean, 
                                                     'error_xy_median' : error_xy_median, 
                                                     'error_xy_max' : error_xy_max}
        
    pickle.dump(summary, open(outfilename, 'w'))

@files(aggregate_results, ('deterministic.fractions.png',
                           'deterministic.frac_vs_err.png'))
def plot_agg(infile, (fractions_filename, frac_vs_err_filename)):
    aggdata = pickle.load(open(infile, 'r'))
    
    ## fractions
    fracs = []
    per_xy_errors = []
    for epoch, data in aggdata.iteritems():
        frac = float(data['frame_pos_pres']) / data['frame_n']
        fracs.append(frac)
        per_xy_errors.append(data['error_xy_mean'])
        print epoch, data['error_xy_mean'], frac
    fracs = np.array(fracs)
    per_xy_errors = np.array(per_xy_errors)

    fracs_sorted = np.sort(fracs)[::-1]
    pylab.figure()
    pylab.plot(fracs_sorted)
    pylab.savefig(fractions_filename)

    fig = pylab.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(fracs, per_xy_errors)
    ax.set_yscale('log')
    pylab.savefig(frac_vs_err_filename)

pipeline_run([det_run, #det_plot, 
              aggregate_results, 
              plot_agg], multiprocess=6)
