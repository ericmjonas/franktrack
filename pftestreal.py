import numpy as np
import glob
import scipy.stats
import pandas
import cPickle as pickle
import likelihood
import util2 as util
import model
import template
import time
from matplotlib import pylab
import cloud
import plotparticles
import os
import organizedata
import videotools
import measure
import video
from ssm import particlefilter as pf
import ssm
import proposals
import filters
import skimage.feature
import datasets

from ruffus import * 

T_DELTA = 1/30.

LIKELIHOOD_CONFIGS = [
    # ('le1', {'power' : 1, 'log' : False, 'normalize' : False}),
    # ('le2', {'power' : 2, 'log' : False, 'normalize' : False}),
    # ('le3', {'power' : 10, 'log' : False, 'normalize' : False}),
    # ('le4', {'power' : 1, 'log' : False, 'normalize' : True}),
    # ('le5', {'power' : 2, 'log' : False, 'normalize' : True}),
    # ('le6', {'power' : 10, 'log' : False, 'normalize' : True}),
    # ('le7', {'power' : 1, 'log' : True, 'normalize' : False}),
    # ('le1', {'power' : 1.0, 'log' : False,
    #          'exp' : False, 'normalize' : True, 
    #          'dist-thold' : 30}),
    # ('le8', {'power' : 0.5, 'log' : True,
    #          'exp' : False, 'normalize' : True, 
    #          'dist-thold' : None, 
    #          'closest-n' : 5}),
    ('le9', {'power' : 2.0, 'log' : True,
             'exp' : False, 'normalize' : True, 
             'dist-thold' : None, 
             'closest-n' : 50, 
             'pix-threshold' : 0}),
    # ('le10', {'power' : 0.5, 'log' : False, 
    #          'exp' : True, 'normalize' : True, 
    #          'dist-thold' : None, 
    #          'closest-n' : 5}),
    # ('le9', {'power' : 2, 'log' : True, 'normalize' : True}),
    # ('le10', {'power' : 2, 'log' : True, 'normalize' : False}),
    # ('le11', {'power' : 10, 'log' : True, 'normalize' : True}),
    ]


FL_DATA = "data/fl"

TemplateObj = template.TemplateRenderCircleBorder

def enlarge_sep(eo_params, amount=1.0, front_amount = 1.0, back_amount=1.0):
    
    b = (eo_params[0]*amount, eo_params[1]*front_amount, eo_params[2]*back_amount)
    return b

def params():
    PARTICLEN = 100
    np.random.seed(0)
    posnoise = 0.01
    velnoise = 0.05
    
    for epoch, frame_start in [('Cummings_07.w2', 1000)]:
                               #("I106.4", 0)]: 

        frame_end = frame_start + 100
        for pix_threshold in [230]:
            for config_name, config_params in LIKELIHOOD_CONFIGS:

                infile = [os.path.join(FL_DATA, epoch), 
                          os.path.join(FL_DATA, epoch, 'config.pickle'), 
                          os.path.join(FL_DATA, epoch, 'region.pickle'), 
                          os.path.join(FL_DATA, epoch, 'led.params.pickle'), 
                          ]

                outfile = 'particles.%s.%s.%f.%f.%d.%d.%d-%d.npz' % (epoch, config_name, posnoise, 
                                                                     velnoise, pix_threshold, 
                                                                     PARTICLEN, frame_start, frame_end)

                yield (infile, outfile, epoch, 
                       (config_name,  config_params), 
                       posnoise, velnoise, pix_threshold, 
                       PARTICLEN, frame_start, frame_end)
           
class CombinedLE(object):
    def __init__(self, LEs, weights):
        self.LEs = LEs
        self.weights = weights

    def score_state(self, state, img):
        score = 0.0
        for le, weight in zip(self.LEs, self.weights):
            score += weight * le.score_state(state, img)
        return score
        
@files(params)
def pf_run((epoch_dir, epoch_config_filename, 
            region_filename, led_params_filename), outfile, 
           epoch, (config_name, config_params), 
           posnoise, 
           velnoise, pix_threshold, PARTICLEN, 
           frame_start, frame_end):
    np.random.seed(0)
    
    cf = pickle.load(open(epoch_config_filename, 'r'))
    region = pickle.load(open(region_filename, 'r'))
    
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])
    led_params = pickle.load(open(led_params_filename, 'r'))

    eoparams = enlarge_sep(measure.led_params_to_EO(cf, led_params))
    #print "EO PARAMS ARE", eoparams
    tr = TemplateObj(0.8)
    tr.set_params(*eoparams)
    
    le1 = likelihood.LikelihoodEvaluator2(env, tr, similarity='dist', 
                                         likeli_params = {'power' : 1.0, 
                                                          'pix-threshold' : pix_threshold})
    
    #le2 = likelihood.LikelihoodEvaluator3(env, tr, params=config_params)
    


    model_inst = model.CustomModel(env, le1, 
                                   POS_NOISE_STD=posnoise,
                                   VELOCITY_NOISE_STD=velnoise)
    frame_pos = np.arange(frame_start, frame_end)
    # load frames
    frames = organizedata.get_frames(epoch_dir, frame_pos)

    y = frames

    prop2 = proposals.HigherIsotropic()
    def img_to_points(img):
        return filters.peak_region_filter(img, min_distance=10)
        
    prop3 = proposals.HigherIsotropicAndData(env, img_to_points)
    mpk = ssm.proposal.MixtureProposalKernel([prop2, prop3], 
                                             [0.5, 0.5])

    unnormed_weights, particles, ancestors = pf.arbitrary_prop(y, model_inst, 
                                                               mpk,
                                                               PARTICLEN)
    
    np.savez_compressed(outfile, 
                        frame_pos = frame_pos, 
                        unnormed_weights=unnormed_weights, 
                        particles=particles, 
                        ancestors=ancestors)

def params_rendered():
    for p in params():
         yield ((p[0][0], p[0][1], p[1]), (p[1] + ".png", 
                                           p[1] + ".xy.pdf", 
                                           p[1] + ".examples.png", 
                                           p[1] + ".stats.pickle"), 
                                           p[2:])


@follows(pf_run)
@files(params_rendered)
def pf_plot((epoch_dir, epoch_config_filename, particles_file), 
            (all_plot_filename, just_xy_filename, examples_filename, 
             results_pickle), other_params):
    
    
    a = np.load(particles_file)
    frame_pos = a['frame_pos']
    weights = ssm.util.norm_weights(a['unnormed_weights'])
    particles = a['particles']
    N = len(particles)

    led_params_filename = os.path.join(epoch_dir, "led.params.pickle")

    cf = pickle.load(open(epoch_config_filename))
    truth = np.load(os.path.join(epoch_dir, 'positions.npy'))
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])
    led_params = pickle.load(open(led_params_filename, 'r'))

    eoparams = enlarge_sep(measure.led_params_to_EO(cf, led_params))

    tr = TemplateObj()
    tr.set_params(*eoparams)
    truth_interp, missing = measure.interpolate(truth)


    STATEVARS = ['x', 'y', 'xdot', 'ydot', 'phi', 'theta']
    # convert types
    vals = dict([(x, []) for x in STATEVARS])
    for p in particles:
        for v in STATEVARS:
            vals[v].append([s[v] for s in p])
    for v in STATEVARS:
        if v == 'phi':
            vals[v] = np.array(vals[v]) % (2*np.pi)
        else:
            vals[v] = np.array(vals[v])

    vals_dict = {}

    # build up the dictionary of true values
    derived_truth = measure.compute_derived(truth_interp, 
                                            T_DELTA)
    truth_interp_dict = {'x' : truth_interp['x'], 
                         'y' : truth_interp['y'], 
                         'xdot' : derived_truth['xdot'], 
                         'ydot' : derived_truth['ydot'], 
                         'phi' : derived_truth['phi'] % (2*np.pi), 
                         'theta' : np.zeros(len(truth_interp['x']))}
    results = {}
    
    pylab.figure(figsize=(8, 10))
    for vi, v in enumerate(STATEVARS):
        v_bar = np.average(vals[v], axis=1, weights=weights)
        vals_dict[v] = v_bar
        v_truth_interp = truth_interp_dict[v][frame_pos]
        x = frame_pos
        cred = np.zeros((len(x), 2), dtype=np.float)
        for ci, (p, w) in enumerate(zip(vals[v], weights)):
            cred[ci] = util.credible_interval(p, w)

        ax = pylab.subplot(len(STATEVARS) + 1,1, 1+vi)

        # plot the estimate and cred interval
        ax.plot(x, v_bar, color='b')
        ax.fill_between(x, cred[:, 0],
                           cred[:, 1], facecolor='b', 
                           alpha=0.4)
        # plot truth
        ax.plot(x, v_truth_interp, 
                linewidth=1, c='k')
        
        ax.set_ylim(np.min(v_truth_interp)-0.1, 
                    np.max(v_truth_interp) + 0.1)

        results[v] = {'pfmean' : v_bar, 
                      'pfcred' : cred, 
                      'truth' : v_truth_interp}
        
        ax.grid(1)
    # this should probably be a separate chunk of the pipeline, 
    # but whatever
    pickle.dump({'params' : other_params, 
                 'variables' : results}, 
                open(results_pickle, 'w'))

    pylab.subplot(len(STATEVARS) + 1, 1, len(STATEVARS)+1)
    # now plot the # of particles consuming 95% of the prob mass
    real_particle_num = []
    for w in weights:
        w = w / np.sum(w) # make sure they're normalized
        w = np.sort(w)[::-1] # sort, reverse order
        wcs = np.cumsum(w)
        wcsi = np.searchsorted(wcs, 0.95)
        real_particle_num.append(wcsi)

    pylab.plot(frame_pos, real_particle_num)

    pylab.savefig(all_plot_filename, dpi=400)

    f2 = pylab.figure(figsize=(16, 8))


    for vi, v in enumerate(['x', 'y']):
        v_bar = np.average(vals[v], axis=1, weights=weights)
        truth_interp = truth_interp_dict[v][frame_pos]
        x = frame_pos
        cred = np.zeros((len(x), 2), dtype=np.float)
        for ci, (p, w) in enumerate(zip(vals[v], weights)):
            cred[ci] = util.credible_interval(p, w)
        ax = pylab.subplot(2, 1, vi+1)

        ax.plot(x, v_bar, color='b')
        ax.fill_between(x, cred[:, 0],
                           cred[:, 1], facecolor='b', 
                           alpha=0.4)
        ax.plot(x, truth_interp, 
                linewidth=1,  c='k')
        for i in np.argwhere(np.isnan(truth[v][frame_pos])):
            ax.axvline(i, c='r', linewidth=0.1, alpha=0.5)

        ax.grid(1)
        ax.set_xlim((np.min(x), np.max(x)))
        ax.set_ylim((np.min(truth_interp), 
                     np.max(truth_interp)))
        ax.set_xlabel("time (frames)")
        ax.set_ylabel("position (m)")
    f2.suptitle(just_xy_filename)
    pylab.savefig(just_xy_filename, dpi=300)
                         
    # find the errors per point
    
    PLOT_ERRORS = 12

    errors = np.zeros(len(vals), dtype=np.float32)
    deltas = np.sqrt((vals_dict['x'] - truth['x'][frame_pos])**2  + \
                    (vals_dict['y'] - truth['y'][frame_pos])**2)    
    deltas[np.isnan(deltas)] = -1 # only to find the Nans and make sorting work

    # find index of errors in decreasing order
    error_i = np.argsort(deltas)[::-1]
    errs = error_i[:PLOT_ERRORS]
    framepos_errs = frame_pos[errs]
    error_frames = organizedata.get_frames(epoch_dir, 
                                           framepos_errs)

    f = pylab.figure()
    WINDOW_PIX = 30
    for plot_error_i, error_frame_i in enumerate(errs):
        abs_frame = framepos_errs[plot_error_i]
        ax = pylab.subplot(3, 4, plot_error_i + 1)
        ax.imshow(error_frames[plot_error_i], 
                  interpolation='nearest', cmap=pylab.cm.gray)
        true_x = truth['x'][abs_frame]
        true_y = truth['y'][abs_frame]
        true_x_pix, true_y_pix = env.gc.real_to_image(true_x, true_y)

        est_x  = vals_dict['x'][error_frame_i]
        est_y = vals_dict['y'][error_frame_i]
        est_phi = vals_dict['phi'][error_frame_i]
        est_theta = vals_dict['theta'][error_frame_i]

        est_x_pix, est_y_pix = env.gc.real_to_image(est_x, est_y)
        

        rendered_img = tr.render(est_phi, est_theta)
                                 

            # now compute position of diodes
        front_pos, back_pos = util.compute_pos(tr.length, est_x_pix, 
                                               est_y_pix, 
                                               est_phi, est_theta)

        cir = pylab.Circle(front_pos, radius=eoparams[1],  
                           ec='g', fill=False,
                           linewidth=2)
        ax.add_patch(cir)
        cir = pylab.Circle(back_pos, radius=eoparams[2],  
                           ec='r', fill=False, 
                           linewidth=2)
        ax.add_patch(cir)

        # true
        ax.axhline(true_y_pix, c='b')
        ax.axvline(true_x_pix, c='b')
        # extimated
        ax.axhline(est_y_pix, c='y')
        ax.axvline(est_x_pix, c='y')

        # LED points in ground truth
        for field_name, color in [('led_front', 'g'), ('led_back', 'r')]:
            lpx, lpy = env.gc.real_to_image(*truth[field_name][abs_frame])
            ax.scatter([lpx], [lpy], c=color)


        ax.set_xlim((true_x_pix - WINDOW_PIX, true_x_pix + WINDOW_PIX))
        ax.set_ylim((true_y_pix - WINDOW_PIX, true_y_pix + WINDOW_PIX))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(framepos_errs[plot_error_i])

    pylab.savefig(examples_filename, dpi=300)

def params_render_vid():
    for p in params():
         yield ((p[0][0], p[0][1], p[1]), (p[1] + ".avi",), p[6])

@follows(pf_run)
@files(params_render_vid)
def pf_render_vid((epoch_dir, epoch_config_filename, particles_file), 
            (vid_filename,), pix_threshold):
    
    a = np.load(particles_file)
    frame_pos = a['frame_pos']
    weights = ssm.util.norm_weights(a['unnormed_weights'])
    particles = a['particles']
    N = len(particles)

    led_params_filename = os.path.join(epoch_dir, "led.params.pickle")

    cf = pickle.load(open(epoch_config_filename))
    truth = np.load(os.path.join(epoch_dir, 'positions.npy'))
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])
    led_params = pickle.load(open(led_params_filename, 'r'))

    eoparams = enlarge_sep(measure.led_params_to_EO(cf, led_params))

    tr = TemplateObj()
    tr.set_params(*eoparams)

    STATEVARS = ['x', 'y', 'xdot', 'ydot', 'phi', 'theta']
    # convert types
    vals = dict([(x, []) for x in STATEVARS])
    for p in particles:
        for v in STATEVARS:
            vals[v].append([s[v] for s in p])
    for v in STATEVARS:
        vals[v] = np.array(vals[v])


    vals_dict = {}

    for vi, v in enumerate(STATEVARS):
        v_bar = np.average(vals[v], axis=1, weights=weights)
        vals_dict[v] = v_bar

    frames = organizedata.get_frames(epoch_dir, 
                                     frame_pos)
    truth, missing = measure.interpolate(truth)

    derived_truth = measure.compute_derived(truth, 
                                            T_DELTA)

    WINDOW_PIX = 40
    f = pylab.figure()
    ax_est = pylab.subplot(2,3, 1)
    ax_particles = pylab.subplot(2, 3, 2)
    ax_particles_global = pylab.subplot(2, 3, 3)
    ax_rawvid = pylab.subplot(2, 3, 4)
    ax_filt = pylab.subplot(2, 3, 5)
    ax_scale_bars = pylab.subplot(2,3, 6)

    plot_temp_filenames = []
    for fi in range(N):
        abs_frame = frame_pos[fi] # absolute frame position
        ax_est.clear()
        ax_particles.clear()
        ax_rawvid.clear()
        ax_filt.clear()
        ax_particles_global.clear()
        ax_scale_bars.clear()

        true_x = truth['x'][abs_frame]
        true_y = truth['y'][abs_frame]
        true_phi = derived_truth['phi'][abs_frame]
        true_x_pix, true_y_pix = env.gc.real_to_image(true_x, true_y)

        est_x  = vals_dict['x'][fi]
        est_y = vals_dict['y'][fi]
        est_phi = vals_dict['phi'][fi]
        est_theta = vals_dict['theta'][fi]

        est_x_pix, est_y_pix = env.gc.real_to_image(est_x, est_y)
        

            # now compute position of diodes
        front_pos, back_pos = util.compute_pos(tr.length, est_x_pix, 
                                               est_y_pix, 
                                               est_phi, est_theta)




        ax_est.imshow(frames[fi], 
                  interpolation='nearest', cmap=pylab.cm.gray)
        ax_scale_bars.imshow(frames[fi].copy(), 
                             interpolation='nearest', cmap=pylab.cm.gray)
        
        ax_rawvid.imshow(frames[fi].copy(), 
                         interpolation='nearest', cmap=pylab.cm.gray)


        # filtered image
        coordinates = filters.peak_region_filter(frames[fi])
        r_width = (tr.front_size + tr.back_size )*1.5
        regions = filters.extract_region_filter(frames[fi], r_width)
        ax_filt.imshow(regions,
                       interpolation='nearest') # , cmap=pylab.cm.gray)

        #ax_filt.plot([p[1] for p in coordinates], [p[0] for p in coordinates], 'r.')
        ax_filt.set_title('ax_filt')
        # plot the mean
        if len(coordinates) > 0:
            ax_filt.axhline(coordinates.mean(axis=0)[0])
            ax_filt.axvline(coordinates.mean(axis=0)[1])

        frames[fi][frames[fi] < pix_threshold] = 0
        ax_particles.imshow(frames[fi], 
                            interpolation='nearest', cmap=pylab.cm.gray)

        ax_particles_global.imshow(frames[fi].copy(), 
                                   interpolation='nearest', cmap=pylab.cm.gray)
        # plot the circles
        cir = pylab.Circle(front_pos[:2], radius=eoparams[1],  
                           ec='g', fill=False,
                           linewidth=2)

        ax_est.add_patch(cir)
        cir_back = pylab.Circle(back_pos[:2], radius=eoparams[2],  
                           ec='r', fill=False, 
                           linewidth=2)
        ax_est.add_patch(cir_back)

        # true
        ax_est.axhline(true_y_pix, c='b')
        ax_est.axvline(true_x_pix, c='b')
        # extimated
        ax_est.axhline(est_y_pix, c='y')
        ax_est.axvline(est_x_pix, c='y')

        # true
        ax_rawvid.axhline(true_y_pix, c='b')
        ax_rawvid.axvline(true_x_pix, c='b')
        # extimated
        ax_rawvid.axhline(est_y_pix, c='y')
        ax_rawvid.axvline(est_x_pix, c='y')
        
        # LED points in ground truth
        for field_name, color in [('led_front', 'g'), ('led_back', 'r')]:
            lpx, lpy = env.gc.real_to_image(*truth[field_name][abs_frame])
            ax_est.scatter([lpx], [lpy], c=color)

        # now all particles
        particle_pix_pts = np.zeros((len(particles[fi]), 2, 3), dtype=np.float)
        for pi in range(len(particles[fi])):
            
            lpx, lpy = env.gc.real_to_image(particles[fi, pi]['x'], 
                                            particles[fi, pi]['y'])
            front_pos, back_pos = util.compute_pos(tr.length, lpx, lpy, 
                                                   particles[fi, pi]['phi'], 
                                                   particles[fi, pi]['theta'])

            particle_pix_pts[pi, 0] = front_pos
            particle_pix_pts[pi, 1] = back_pos
        ax_particles.scatter(particle_pix_pts[:, 0, 0], 
                             particle_pix_pts[:, 0, 1], 
                             linewidth=0, 
                             alpha = 1.0, c='g', s=1)
        ax_particles.scatter(particle_pix_pts[:, 1, 0], 
                             particle_pix_pts[:, 1, 1], 
                             linewidth=0, 
                             alpha = 1.0, c='r', s=1)

        ax_particles_global.scatter(particle_pix_pts[:, 0, 0], 
                             particle_pix_pts[:, 0, 1], 
                             linewidth=0, 
                             alpha = 1.0, c='g', s=2)
        ax_particles_global.scatter(particle_pix_pts[:, 1, 0], 
                             particle_pix_pts[:, 1, 1], 
                             linewidth=0, 
                             alpha = 1.0, c='r', s=2)

        for ax in [ax_est, ax_particles, ax_scale_bars]:
            ax.set_xlim((true_x_pix - WINDOW_PIX, true_x_pix + WINDOW_PIX))
            ax.set_ylim((true_y_pix - WINDOW_PIX, true_y_pix + WINDOW_PIX))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(frame_pos[fi])

        bar_offset_x = 5 + true_x_pix - WINDOW_PIX
        bar_offset_y = true_y_pix - WINDOW_PIX
        ax_scale_bars.plot([bar_offset_x, 
                            eoparams[0] + bar_offset_x], 
                           [bar_offset_y + 5, bar_offset_y + 5], 
                           c='b', linewidth=5)
        ax_scale_bars.plot([bar_offset_x, 
                            eoparams[1]*2 + bar_offset_x], 
                           [bar_offset_y + 10, bar_offset_y + 10], 
                           c='g', linewidth=3)
        ax_scale_bars.plot([bar_offset_x, 
                            eoparams[2]*2 + bar_offset_x], 
                           [bar_offset_y + 15, bar_offset_y + 15], 
                           c='r', linewidth=3)

        ax_rawvid.set_xticks([])
        ax_rawvid.set_yticks([])
        ax_filt.set_xticks([])
        ax_filt.set_yticks([])

        ax_particles_global.set_xticks([])
        ax_particles_global.set_yticks([])
        plot_filename = "%s.%08d.png" % (vid_filename, fi)
        f.savefig(plot_filename, dpi=200)
        plot_temp_filenames.append(plot_filename)
        
    video.frames_to_mpng("%s.*.png" % vid_filename, vid_filename, fps = float(len(plot_temp_filenames))/10.0)
    # delete extras
    for f in plot_temp_filenames:
        os.remove(f)

@merge(pf_plot, 'particles.summary.pickle')
def results_summarize(infiles, summary_file):
    df_rows = []
    for infile in infiles:
        stats_filename = infile[-1]
        data = pickle.load(open(stats_filename, 'r'))
        for var_name, var_data in data['variables'].iteritems():
            dp = data['params']
            d = {'epoch' : dp[0], 
                 'likelihood_name' : dp[1][0], 
                 'likelihood_power' : dp[1][1]['power'], 
                 'likelihood_log' : dp[1][1]['log'], 
                 'likelihood_normalize' : dp[1][1]['normalize'], 
                 'posnoise' : dp[2], 
                 'velnoise' : dp[3], 
                 'pix_threshold' : dp[4], 
                 'particlen' : dp[5], 
                 'frame_start' : dp[6], 
                 'frame_end' : dp[7], 
                 'variable' : var_name, 
                 'pfmean' : var_data['pfmean'], 
                 'pfcred' : var_data['pfcred'], 
                 'truth' : var_data['truth']}
            df_rows.append(d)
    df = pandas.DataFrame(df_rows)
    pickle.dump(df, open(summary_file, 'w'))

pipeline_run([pf_run, pf_plot, pf_render_vid, 
              results_summarize], multiprocess=4)
