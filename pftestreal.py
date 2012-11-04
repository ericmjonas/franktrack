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

from ruffus import * 
import pf

FL_DATA = "data/fl"
def params():
    PARTICLEN = 1000
    FRAMEN = 100
    EPOCHS = ['bukowski_04.W1']
    epoch = EPOCHS[0]
    posnoise = 0.005
    velnoise = 0.05
            
    infile = [os.path.join(FL_DATA, epoch), 
              os.path.join(FL_DATA, epoch, 'config.pickle'), 
              os.path.join(FL_DATA, epoch, 'frameagg.npz'), 
              ]

    outfile = 'particles.%s.%f.%f.%d.%d.npz' % (epoch, posnoise, 
                                                   velnoise, 
                                                   PARTICLEN, FRAMEN)
                                                   
                
    yield (infile, outfile, epoch, posnoise, velnoise, PARTICLEN, FRAMEN)
           

@files(params)
def pf_run((epoch_dir, epoch_config_filename, 
            frame_agg_filename), outfile, epoch, posnoise, 
           velnoise, PARTICLEN, 
           FRAMEN):
    np.random.seed(0)
    
    cf = pickle.load(open(epoch_config_filename))
    frameagg = np.load(frame_agg_filename)
    
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])

    eo = likelihood.EvaluateObj(*cf['frame_dim_pix'])
    eo.set_params(10, 4, 2)
    
    le = likelihood.LikelihoodEvaluator(env, eo)

    model_inst = model.LinearModel(env, le, 
                                   POS_NOISE_STD=posnoise,
                                   VELOCITY_NOISE_STD=velnoise)
    # load frames
    frames = organizedata.get_frames(epoch_dir, np.arange(FRAMEN))
    frame_mean = frameagg['mean']
    for fi, f in enumerate(frames):
        frames_normed = f.astype(float) - frame_mean
        frames_trunc = np.maximum(frames_normed, 0)
        assert np.min(frames_trunc) >= 0
        assert np.max(frames_trunc) < 256
        frames[fi] = frames_trunc
    y = frames
    videotools.dump_grey_movie('test.avi', y)

    weights, particles = pf.particle_filter(y, model_inst, 
                                            len(y), PARTICLEN)
    np.savez_compressed(outfile, 
                        weights=weights, particles=particles)

def params_rendered():
    for p in params():
         yield ((p[0][0], p[0][1], p[1]), p[1] + ".pdf")

@follows(pf_run)
@files(params_rendered)
def pf_plot((epoch_dir, epoch_config_filename, particles_file), 
            plot_filename):
    
    T_DELTA = 1/30.
    
    a = np.load(particles_file)
    weights = a['weights']
    particles = a['particles']
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

    pylab.figure()
    for vi, v in enumerate(STATEVARS):
        v_bar = np.average(vals[v], axis=1, weights=weights)
        x = np.arange(0, len(v_bar))
        cred = np.zeros((len(x), 2), dtype=np.float)
        for ci, (p, w) in enumerate(zip(vals[v], weights)):
            cred[ci] = util.credible_interval(p, w)

        pylab.subplot(len(STATEVARS) + 1,1, 1+vi)

        pylab.plot(x, v_bar, color='b')
        pylab.fill_between(x, cred[:, 0],
                           cred[:, 1], facecolor='b', 
                           alpha=0.4)
        if v in ['x', 'y']:
            pylab.scatter(np.arange(N), truth[v][:N], 
                          linewidth=0, s=2)

        
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

    pylab.savefig(plot_filename)


pipeline_run([pf_run, pf_plot])
