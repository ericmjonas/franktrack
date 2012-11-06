"""
Code to brute-force compare the likelihood model
"""

import numpy as np
import scipy.stats
import scipy.ndimage
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
import glob

from ruffus import * 
import pf

FL_DATA = "data/fl"
def params():
    EPOCHS = ['bukowski_04.W1', 'bukowski_04.W2', 
              'bukowski_04.C', 'bukowski_04.linear']
    FRAMES = np.arange(100)*50
    
    for epoch in EPOCHS:
        for frame in FRAMES:
            infile = [os.path.join(FL_DATA, epoch), 
                      os.path.join(FL_DATA, epoch, 'config.pickle'), 
                      os.path.join(FL_DATA, epoch, 'frameagg.npz'), 
                      ]

            outfile = '%s.likelihoodscores.%05d.npz' % (epoch, frame)
            
            yield (infile, outfile, epoch, frame)
           
def frame_params():
    EPOCHS = [os.path.split(f)[1] for f in glob.glob(FL_DATA + "/*")]
    FRAMES = np.arange(100)*50
    
    for epoch in EPOCHS:
        for frame_pos in FRAMES:
            for f in range(3):
                frame = frame_pos + f
                infile = [os.path.join(FL_DATA, epoch), 
                          os.path.join(FL_DATA, epoch, 'config.pickle'), 
                          os.path.join(FL_DATA, epoch, 'frameagg.npz'), 
                          ]

                outfile = '%s.likelihoodscores.%05d.pdf' % (epoch, frame)

                yield (infile, outfile, epoch, frame)

@files(params)
def score_frame((epoch_dir, epoch_config_filename, 
            frame_agg_filename), outfile, epoch, frame):

    np.random.seed(0)
    
    cf = pickle.load(open(epoch_config_filename))
    frameagg = np.load(frame_agg_filename)
    
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])

    eo = likelihood.EvaluateObj(*cf['frame_dim_pix'])
    eo.set_params(10, 4, 2)
    
    le = likelihood.LikelihoodEvaluator(env, eo)

    frames = organizedata.get_frames(epoch_dir, np.array([frame]))

    # create the state vector
    x_range = np.linspace(0, cf['field_dim_m'][1], 1)
    y_range = np.linspace(0, cf['field_dim_m'][0], 1)
    phi_range = np.linspace(0, 2*np.pi, 1)
    theta_range = np.array([np.pi/2.])

    state = np.zeros(1, dtype=util.DTYPE_STATE)
    
    # now evaluate
    scores = np.zeros((len(y_range), len(x_range), 
                       len(phi_range), len(theta_range)), 
                      dtype=np.float32)

    frame_mean = frameagg['mean']
    # for fi, f in enumerate(frames):
    #     frames_normed = f.astype(float) - frame_mean
    #     frames_trunc = np.maximum(frames_normed, 0)
    #     assert np.min(frames_trunc) >= 0
    #     assert np.max(frames_trunc) < 256
    #     frames[fi] = frames_trunc

    for yi, y in enumerate(y_range):
        print yi, len(y_range)
        for xi, x in enumerate(x_range):
            for phii, phi in enumerate(phi_range):
                for thetai, theta in enumerate(theta_range):
                    state['x'][0] = x
                    state['y'][0] = y
                    state['phi'][0] = phi
                    state['theta'][0] = theta

                    score = le.score_state(state, frames[0])
                    scores[yi, xi, phii, thetai] = score

    # videotools.dump_grey_movie('test.avi', y)

    # weights, particles = pf.particle_filter(y, model_inst, 
    #                                         len(y), PARTICLEN)
    np.savez_compressed(outfile, 
                        x_range = x_range, y_range=y_range, 
                        phi_range = phi_range, theta_range = theta_range,
                        scores=scores, frame=frames[0], 
                        frame_mean = frame_mean)


@transform(score_frame, suffix(".npz"), [".png", ".hist.png", ".frame.png"])
def plot(infile, (outfile, outfile_hist, frame_file)):
    data = np.load(infile)
    scores = data['scores']
    phi_range = data['phi_range']
    x = int(np.ceil(np.sqrt(len(phi_range))))

    max_scores_i = np.argmax(scores)
    (ms_y, ms_x, ms_phi, ms_theta) = np.unravel_index(max_scores_i, 
                                                      scores.shape)
    pylab.figure()
    pylab.hist(scores.flat, bins=255)
    pylab.savefig(outfile_hist, dpi=300)
    # use global min, max
    minscore = np.min(scores)
    maxscore = np.max(scores)
    print "MAX SCORE =", maxscore
    pylab.figure()
    for pi, i in enumerate(phi_range):
        pylab.subplot(x, x, pi +1)
        img = scores[:, :, pi, 0]
        i_max = np.argmax(img)
        y_max, x_max = np.unravel_index(i_max, img.shape)
        pylab.imshow(img, interpolation='nearest', 
                     vmin=minscore, vmax=maxscore) 
        if pi == ms_phi:
            c = 'g'
        else:
            c = 'r'
        pylab.axhline(y_max, c=c)
        pylab.axvline(x_max, c=c)
    pylab.savefig(outfile, dpi=300)
    pylab.figure()
    frame = data['frame']
    pylab.subplot(1, 2, 1)
    pylab.imshow(frame, interpolation='nearest')
    maxpoint = np.argmax(frame)
    y, x = np.unravel_index(maxpoint, frame.shape)
    pylab.axhline(y)
    pylab.axvline(x)
    pylab.subplot(1, 2, 2)
    pylab.plot(frame[y, :])
    pylab.savefig(frame_file, dpi=300)



@files(frame_params)
def examine_frame((epoch_dir, epoch_config_filename, 
            frame_agg_filename), outfile, epoch, frame):

    np.random.seed(0)
    
    cf = pickle.load(open(epoch_config_filename))
    frameagg = np.load(frame_agg_filename)
    
    frames = organizedata.get_frames(epoch_dir, np.array([frame]))
    frame = frames[0]
    frame_mean = frameagg['mean']
    frame_std = np.sqrt(frameagg['variance'])
    # for fi, f in enumerate(frames):
    #     frames_normed = f.astype(float) - frame_mean
    #     frames_trunc = np.maximum(frames_normed, 0)
    #     assert np.min(frames_trunc) >= 0
    #     assert np.max(frames_trunc) < 256
    #     frames[fi] = frames_trunc

    maxpoint = np.argmax(frame)
    max_y, max_x = np.unravel_index(maxpoint, frame.shape)

    pylab.figure()
    pylab.subplot(3, 2, 1)
    pylab.imshow(frame, interpolation='nearest')
    pylab.axhline(max_y, c='r', alpha=0.3)
    pylab.axvline(max_x, c='g', alpha=0.3)

    pylab.subplot(3, 2, 3)
    pylab.plot(frame[max_y, :], 'r')
    pylab.plot(frame[:, max_x], 'g')

    # attempt to select all pix that are outside their std dev

    pyplot.subplot(3, 2, 5)
    pylab.imshow(frame, interpolation='nearest', cmap=pylab.cm.gray)

    for thold_i, thold in enumerate([240, 250, 255]):
        #outlier_mask = frame > (frame_mean + 2*frame_std)
        high_mask = frame >= thold

        tgt_mask = high_mask # np.logical_and(high_mask, outlier_mask)
        frame2 = frame.copy()

        S = 6
        a = scipy.ndimage.binary_dilation(tgt_mask, 
                                          structure=np.ones((S, S)))
        frame2[np.logical_not(a)] = 0

        pyplot.subplot(3, 2, 2*thold_i + 2)
        pylab.imshow(frame2, interpolation='nearest', cmap=pylab.cm.gray)

    pylab.savefig(outfile, dpi=400)

# Def params_rendered():
#     for p in params():
#          yield ((p[0][0], p[0][1], p[1]), p[1] + ".pdf")

# @follows(pf_run)
# @files(params_rendered)
# def pf_plot((epoch_dir, epoch_config_filename, particles_file), 
#             plot_filename):
    
#     T_DELTA = 1/30.
    
#     a = np.load(particles_file)
#     weights = a['weights']
#     particles = a['particles']
#     N = len(particles)

#     cf = pickle.load(open(epoch_config_filename))
#     truth = np.load(os.path.join(epoch_dir, 'positions.npy'))
    
#     STATEVARS = ['x', 'y', 'xdot', 'ydot', 'phi', 'theta']
#     # convert types
#     vals = dict([(x, []) for x in STATEVARS])
#     for p in particles:
#         for v in STATEVARS:
#             vals[v].append([s[v] for s in p])
#     for v in STATEVARS:
#         vals[v] = np.array(vals[v])

#     pylab.figure()
#     for vi, v in enumerate(STATEVARS):
#         v_bar = np.average(vals[v], axis=1, weights=weights)
#         x = np.arange(0, len(v_bar))
#         cred = np.zeros((len(x), 2), dtype=np.float)
#         for ci, (p, w) in enumerate(zip(vals[v], weights)):
#             cred[ci] = util.credible_interval(p, w)

#         pylab.subplot(len(STATEVARS) + 1,1, 1+vi)

#         pylab.plot(x, v_bar, color='b')
#         pylab.fill_between(x, cred[:, 0],
#                            cred[:, 1], facecolor='b', 
#                            alpha=0.4)
#         if v in ['x', 'y']:
#             pylab.scatter(np.arange(N), truth[v][:N], 
#                           linewidth=0, s=2)

        
#     pylab.subplot(len(STATEVARS) + 1, 1, len(STATEVARS)+1)
#     # now plot the # of particles consuming 95% of the prob mass
#     real_particle_num = []
#     for w in weights:
#         w = w / np.sum(w) # make sure they're normalized
#         w = np.sort(w)[::-1] # sort, reverse order
#         wcs = np.cumsum(w)
#         wcsi = np.searchsorted(wcs, 0.95)
#         real_particle_num.append(wcsi)

#     pylab.plot(real_particle_num)

#     pylab.savefig(plot_filename)


pipeline_run([examine_frame], multiprocess=4)
