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

from ruffus import * 
import pf

PIX_THRESHOLD = 200

FL_DATA = "data/fl"
def params():
    PARTICLEN = 1000
    FRAMEN = 5000
    EPOCHS = ['bukowski_04.W2']
    epoch = EPOCHS[0]
    for posnoise in [0.01]:
        for velnoise in [0.001, 0.01]:
            
            infile = [os.path.join(FL_DATA, epoch), 
                      os.path.join(FL_DATA, epoch, 'config.pickle'), 
                      os.path.join(FL_DATA, epoch, 'region.pickle'), 
                      os.path.join(FL_DATA, epoch, 'led.params.pickle'), 
                      ]

            outfile = 'particles.%s.%f.%f.%d.%d.npz' % (epoch, posnoise, 
                                                           velnoise, 
                                                           PARTICLEN, FRAMEN)


            yield (infile, outfile, epoch, posnoise, velnoise, PARTICLEN, FRAMEN)
           

@files(params)
def pf_run((epoch_dir, epoch_config_filename, 
            region_filename, led_params_filename), outfile, 
           epoch, posnoise, 
           velnoise, PARTICLEN, 
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
    
    le = likelihood.LikelihoodEvaluator(env, eo)

    model_inst = model.LinearModel(env, le, 
                                   POS_NOISE_STD=posnoise,
                                   VELOCITY_NOISE_STD=velnoise)
    # load frames
    frames = organizedata.get_frames(epoch_dir, np.arange(FRAMEN))
    for fi, f in enumerate(frames):
        frames[fi][frames[fi] < PIX_THRESHOLD] = 0
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
                                           p[1] + ".xy.png"))

@follows(pf_run)
@files(params_rendered)
def pf_plot((epoch_dir, epoch_config_filename, particles_file), 
            (all_plot_filename, just_xy_filename)):
    
    T_DELTA = 1/30.
    
    a = np.load(particles_file)
    FRAMES = 100000000
    weights = a['weights'][:FRAMES]
    particles = a['particles'][:FRAMES]
    N = len(particles)

    cf = pickle.load(open(epoch_config_filename))
    truth = np.load(os.path.join(epoch_dir, 'positions.npy'))
    
    STATEVARS = ['x', 'y', 'xdot', 'ydot', 'phi', 'theta']
    # convert types
    vals = dict([(x, []) for x in STATEVARS])
    for p in particles:
        for v in STATEVARS:
            vals[v].append([s[v] for s in p])
    for v in STATEVARS:
        vals[v] = np.array(vals[v])

    pylab.figure(figsize=(8, 10))
    for vi, v in enumerate(STATEVARS):
        v_bar = np.average(vals[v], axis=1, weights=weights)
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
                   linewidth=0, s=1, c='k')
        for i in np.argwhere(np.isnan(truth[v][:N])):
            ax.axvline(i, c='r')

        ax.grid(1)
        ax.set_xlim((0, N))
        
    pylab.savefig(just_xy_filename, dpi=300)
                         
                       
    
pipeline_run([pf_run, pf_plot], multiprocess=6)
