import numpy as np
import scipy.stats
import cPickle as pickle
from matplotlib import pyplot
import likelihood
import util2 as util
import model
import time
from matplotlib import pylab
import cloud
import plotparticles
import os
import organizedata
import videotools
import measure
import video

from ruffus import * 
import pf

SIMILARITIES = [('dist2', 'dist', {'power' : 2}), 
                ('dist4', 'dist', {'power' : 4})]

FL_DATA = "data/fl"
def params():
    PARTICLEN = 1000
    FRAMEN = 500
    EPOCHS = ['bukowski_04.W1', 'bukowski_04.W2']

    for epoch in EPOCHS:
        for posnoise in [0.01 ]:
            for velnoise in [0.01]:
                for pix_threshold in [200]:
                    for sim_name, sim_type, sim_params in SIMILARITIES:

                        infile = [os.path.join(FL_DATA, epoch), 
                                  os.path.join(FL_DATA, epoch, 'config.pickle'), 
                                  os.path.join(FL_DATA, epoch, 'region.pickle'), 
                                  os.path.join(FL_DATA, epoch, 'led.params.pickle'), 
                                  ]

                        outfile = 'particles.%s.%s.%f.%f.%d.%d.%d.npz' % (epoch, sim_name, posnoise, 
                                                                       velnoise, pix_threshold, 
                                                                       PARTICLEN, FRAMEN)

                        yield (infile, outfile, epoch, (sim_name, sim_type, sim_params), 
                               posnoise, velnoise, pix_threshold, 
                               PARTICLEN, FRAMEN)
           

@files(params)
def pf_run((epoch_dir, epoch_config_filename, 
            region_filename, led_params_filename), outfile, 
           epoch, (sim_name, sim_type, sim_params), 
           posnoise, 
           velnoise, pix_threshold, PARTICLEN, 
           FRAMEN):
    np.random.seed(0)
    
    cf = pickle.load(open(epoch_config_filename, 'r'))
    region = pickle.load(open(region_filename, 'r'))
    
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])
    led_params = pickle.load(open(led_params_filename, 'r'))

    eoparams = measure.led_params_to_EO(cf, led_params)

    eo = likelihood.EvaluateObj(*cf['frame_dim_pix'])
    eo.set_params(*eoparams)
    
    le = likelihood.LikelihoodEvaluator(env, eo, similarity=sim_type, 
                                        sim_params = sim_params)

    model_inst = model.CustomModel(env, le, 
                                   POS_NOISE_STD=posnoise,
                                   VELOCITY_NOISE_STD=velnoise)
    # load frames
    frames = organizedata.get_frames(epoch_dir, np.arange(FRAMEN))
    for fi, f in enumerate(frames):
        frames[fi][frames[fi] < pix_threshold] = 0
        pix_ul = env.gc.real_to_image(region['x_pos_min'], 
                                   region['y_pos_min'])
        frames[fi][:pix_ul[1], :] = 0
        frames[fi][:, :pix_ul[0]] = 0

        pix_lr = env.gc.real_to_image(region['x_pos_max'], 
                                   region['y_pos_max'])
        frames[fi][pix_lr[1]:, :] = 0
        frames[fi][:, pix_lr[0]:] = 0

        

    y = frames
    videotools.dump_grey_movie('test.avi', y)

    weights, particles = pf.particle_filter(y, model_inst, 
                                            len(y), PARTICLEN)
    np.savez_compressed(outfile, 
                        weights=weights, particles=particles)

def params_rendered():
    for p in params():
         yield ((p[0][0], p[0][1], p[1]), (p[1] + ".png", 
                                           p[1] + ".xy.pdf", 
                                           p[1] + ".examples.png"))

@follows(pf_run)
@files(params_rendered)
def pf_plot((epoch_dir, epoch_config_filename, particles_file), 
            (all_plot_filename, just_xy_filename, examples_filename)):
    
    T_DELTA = 1/30.
    
    a = np.load(particles_file)
    FRAMES = 100000000
    weights = a['weights'][:FRAMES]
    particles = a['particles'][:FRAMES]
    N = len(particles)

    led_params_filename = os.path.join(epoch_dir, "led.params.pickle")

    cf = pickle.load(open(epoch_config_filename))
    truth = np.load(os.path.join(epoch_dir, 'positions.npy'))
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])
    led_params = pickle.load(open(led_params_filename, 'r'))

    eoparams = measure.led_params_to_EO(cf, led_params)

    eo = likelihood.EvaluateObj(*cf['frame_dim_pix'])
    eo.set_params(*eoparams)
    truth_interp, missing = measure.interpolate(truth)


    STATEVARS = ['x', 'y', 'xdot', 'ydot', 'phi', 'theta']
    # convert types
    vals = dict([(x, []) for x in STATEVARS])
    for p in particles:
        for v in STATEVARS:
            vals[v].append([s[v] for s in p])
    for v in STATEVARS:
        vals[v] = np.array(vals[v])

    vals_dict = {}

    pylab.figure(figsize=(8, 10))
    for vi, v in enumerate(STATEVARS):
        v_bar = np.average(vals[v], axis=1, weights=weights)
        vals_dict[v] = v_bar

        x = np.arange(0, len(v_bar))
        cred = np.zeros((len(x), 2), dtype=np.float)
        for ci, (p, w) in enumerate(zip(vals[v], weights)):
            cred[ci] = util.credible_interval(p, w)

        ax = pylab.subplot(len(STATEVARS) + 1,1, 1+vi)

        ax.plot(x, v_bar, color='b')
        ax.fill_between(x, cred[:, 0],
                           cred[:, 1], facecolor='b', 
                           alpha=0.4)
        if v in ['x', 'y']:
            ax.scatter(np.arange(N), truth[v][:N], 
                          linewidth=0, s=1, c='k')
        if v in ['xdot', 'ydot']:
            truedelta = truth_interp[v[0]][1:(N+1)] - truth_interp[v[0]][:N]
            ax.plot(np.arange(N), truedelta, 
                          linewidth=1, c='k')
        ax.grid(1)
        ax.set_xlim((0, N))

    pylab.subplot(len(STATEVARS) + 1, 1, len(STATEVARS)+1)
    # now plot the # of particles consuming 95% of the prob mass
    real_particle_num = []
    for w in weights:
        w = w / np.sum(w) # make sure they're normalized
        w = np.sort(w)[::-1] # sort, reverse order
        wcs = np.cumsum(w)
        wcsi = np.searchsorted(wcs, 0.95)
        real_particle_num.append(wcsi)

    pylab.plot(real_particle_num)

    pylab.savefig(all_plot_filename, dpi=400)

    f2 = pylab.figure(figsize=(16, 8))


    for vi, v in enumerate(['x', 'y']):
        v_bar = np.average(vals[v], axis=1, weights=weights)
        x = np.arange(0, len(v_bar))
        cred = np.zeros((len(x), 2), dtype=np.float)
        for ci, (p, w) in enumerate(zip(vals[v], weights)):
            cred[ci] = util.credible_interval(p, w)
        ax = pylab.subplot(2, 1, vi+1)

        ax.plot(x, v_bar, color='b')
        ax.fill_between(x, cred[:, 0],
                           cred[:, 1], facecolor='b', 
                           alpha=0.4)
        ax.scatter(np.arange(N), truth[v][:N], 
                   linewidth=0, s=2, c='k')
        for i in np.argwhere(np.isnan(truth[v][:N])):
            ax.axvline(i, c='r', linewidth=0.1, alpha=0.5)

        ax.grid(1)
        ax.set_xlim((0, N))
        ax.set_xlabel("time (frames)")
        ax.set_ylabel("position (m)")
    f2.suptitle(just_xy_filename)
    pylab.savefig(just_xy_filename, dpi=300)
                         
    # find the errors per point
    
    PLOT_ERRORS = 12

    errors = np.zeros(len(vals), dtype=np.float32)
    deltas = np.sqrt((vals_dict['x'] - truth['x'][:N])**2  + \
                    (vals_dict['y'] - truth['y'][:N])**2)    
    deltas[np.isnan(deltas)] = -1 # only to find the Nans and make sorting work

    # find index of errors in decreasing order
    error_i = np.argsort(deltas)[::-1]
    errs = error_i[:PLOT_ERRORS]
    error_frames = organizedata.get_frames(epoch_dir, 
                                           errs)

    f = pylab.figure()
    WINDOW_PIX = 30
    for plot_error_i, error_frame_i in enumerate(errs):
        ax = pylab.subplot(3, 4, plot_error_i + 1)
        ax.imshow(error_frames[plot_error_i], 
                  interpolation='nearest', cmap=pylab.cm.gray)
        true_x = truth['x'][errs[plot_error_i]]
        true_y = truth['y'][errs[plot_error_i]]
        true_x_pix, true_y_pix = env.gc.real_to_image(true_x, true_y)

        est_x  = vals_dict['x'][error_frame_i]
        est_y = vals_dict['y'][error_frame_i]
        est_phi = vals_dict['phi'][error_frame_i]
        est_theta = vals_dict['theta'][error_frame_i]

        est_x_pix, est_y_pix = env.gc.real_to_image(est_x, est_y)
        

        rendered_img = eo.render_source(est_x_pix, est_y_pix, 
                                        est_phi, est_theta)

            # now compute position of diodes
        front_pos, back_pos = util.compute_pos(eo.length, est_x_pix, 
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
            lpx, lpy = env.gc.real_to_image(*truth[field_name][error_frame_i])
            ax.scatter([lpx], [lpy], c=color)


        ax.set_xlim((true_x_pix - WINDOW_PIX, true_x_pix + WINDOW_PIX))
        ax.set_ylim((true_y_pix - WINDOW_PIX, true_y_pix + WINDOW_PIX))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(error_frame_i)

    pylab.savefig(examples_filename, dpi=300)

def params_render_vid():
    for p in params():
         yield ((p[0][0], p[0][1], p[1]), (p[1] + ".avi",))

@follows(pf_run)
@files(params_render_vid)
def pf_render_vid((epoch_dir, epoch_config_filename, particles_file), 
            (vid_filename,)):
    
    T_DELTA = 1/30.
    
    a = np.load(particles_file)
    FRAMES = 100000000
    weights = a['weights'][:FRAMES]
    particles = a['particles'][:FRAMES]
    N = len(particles)

    led_params_filename = os.path.join(epoch_dir, "led.params.pickle")

    cf = pickle.load(open(epoch_config_filename))
    truth = np.load(os.path.join(epoch_dir, 'positions.npy'))
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])
    led_params = pickle.load(open(led_params_filename, 'r'))

    eoparams = measure.led_params_to_EO(cf, led_params)

    eo = likelihood.EvaluateObj(*cf['frame_dim_pix'])
    eo.set_params(*eoparams)

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
                                     np.arange(N))
    truth, missing = measure.interpolate(truth)

    WINDOW_PIX = 60
    f = pylab.figure()
    ax_est = pylab.subplot(1,2, 1)
    ax_particles = pylab.subplot(1, 2, 2)
    plot_temp_filenames = []
    for fi in range(N):
        ax_est.clear()
        ax_particles.clear()
        true_x = truth['x'][fi]
        true_y = truth['y'][fi]
        true_x_pix, true_y_pix = env.gc.real_to_image(true_x, true_y)

        est_x  = vals_dict['x'][fi]
        est_y = vals_dict['y'][fi]
        est_phi = vals_dict['phi'][fi]
        est_theta = vals_dict['theta'][fi]

        est_x_pix, est_y_pix = env.gc.real_to_image(est_x, est_y)
        

            # now compute position of diodes
        front_pos, back_pos = util.compute_pos(eo.length, est_x_pix, 
                                               est_y_pix, 
                                               est_phi, est_theta)

        cir = pylab.Circle(front_pos, radius=eoparams[1],  
                           ec='g', fill=False,
                           linewidth=2)

        ax_est.imshow(frames[fi], 
                  interpolation='nearest', cmap=pylab.cm.gray)
        ax_particles.imshow(frames[fi], 
                            interpolation='nearest', cmap=pylab.cm.gray)

        ax_est.add_patch(cir)
        cir = pylab.Circle(back_pos, radius=eoparams[2],  
                           ec='r', fill=False, 
                           linewidth=2)
        ax_est.add_patch(cir)

        # true
        ax_est.axhline(true_y_pix, c='b')
        ax_est.axvline(true_x_pix, c='b')
        # extimated
        ax_est.axhline(est_y_pix, c='y')
        ax_est.axvline(est_x_pix, c='y')

        # LED points in ground truth
        for field_name, color in [('led_front', 'g'), ('led_back', 'r')]:
            lpx, lpy = env.gc.real_to_image(*truth[field_name][fi])
            ax_est.scatter([lpx], [lpy], c=color)

        # now all particles
        particle_pix_pts = np.zeros((len(particles[fi]), 2, 3), dtype=np.float)
        for pi in range(len(particles[fi])):
            
            lpx, lpy = env.gc.real_to_image(particles[fi, pi]['x'], 
                                            particles[fi, pi]['y'])
            front_pos, back_pos = util.compute_pos(eo.length, lpx, lpy, 
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

        for ax in [ax_est, ax_particles]:
            ax.set_xlim((true_x_pix - WINDOW_PIX, true_x_pix + WINDOW_PIX))
            ax.set_ylim((true_y_pix - WINDOW_PIX, true_y_pix + WINDOW_PIX))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(fi)
        plot_filename = "%s.%08d.png" % (vid_filename, fi)
        f.savefig(plot_filename)
        plot_temp_filenames.append(plot_filename)

    video.frames_to_mpng("%s.*.png" % vid_filename, vid_filename)
    # delete extras
    for f in plot_temp_filenames:
        os.remove(f)

pipeline_run([pf_run, pf_plot, pf_render_vid], multiprocess=4)
