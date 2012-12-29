"""
Code to brute-force compare the likelihood model
"""

import numpy as np
import scipy.stats
import scipy.ndimage
import cPickle as pickle
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
import measure

from ruffus import * 
import pf

PIX_THRESHOLD = 220
FL_DATA = "data/fl"

#cloud.start_simulator()

def params():
    EPOCHS = ['bukowski_04.W2']# , 'bukowski_04.W2', 
    #'bukowski_04.C', 'bukowski_04.linear']
    FRAMES = np.arange(10)*100
    
    for epoch in EPOCHS:
        for frame in FRAMES:
            infiles = [os.path.join(FL_DATA, epoch), 
                      os.path.join(FL_DATA, epoch, 'config.pickle'), 
                      os.path.join(FL_DATA, epoch, 'framehist.npz'), 
                      ]
            basename = '%s.likelihoodscores.%05d' % (epoch, frame)
            outfiles = [basename + ".wait.pickle", 
                        basename + ".wait.npz"]
            
            yield (infiles, outfiles, epoch, frame)
           
# def frame_params():
#     EPOCHS = [os.path.split(f)[1] for f in glob.glob(FL_DATA + "/*")]
#     FRAMES = np.arange(10)*50
    
#     for epoch in EPOCHS:
#         for frame_pos in FRAMES:
#             for f in range(3):
#                 frame = frame_pos + f
#                 infile = [os.path.join(FL_DATA, epoch), 
#                           os.path.join(FL_DATA, epoch, 'config.pickle'), 
#                           os.path.join(FL_DATA, epoch, 'region.pickle'), 
#                           os.path.join(FL_DATA, epoch, 'framehist.npz'), 
#                           ]

#                 outfile = '%s.likelihoodscores.%05d.pdf' % (epoch, frame)

#                 yield (infile, outfile, epoch, frame)

@files(params)
def score_frame_queue((dataset_dir, dataset_config_filename, 
            frame_hist_filename), (outfile_wait, 
                                   outfile_npz), dataset_name, frame):

    np.random.seed(0)
    
    dataset_dir = os.path.join(FL_DATA, dataset_name)

    cf = pickle.load(open(dataset_config_filename))
    led_params = pickle.load(open(os.path.join(dataset_dir, "led.params.pickle")))

    EO = measure.led_params_to_EO(cf, led_params)

    x_range = np.linspace(0, cf['field_dim_m'][1], 150)
    y_range = np.linspace(0, cf['field_dim_m'][0], 150)
    phi_range = np.linspace(0, 2*np.pi, 20)
    degrees_from_vertical = 30
    radian_range = degrees_from_vertical/180. * np.pi
    theta_range = np.linspace(np.pi/2.-radian_range, 
                              np.pi/2. + radian_range, 6)

    sv = create_state_vect(y_range, x_range, phi_range, theta_range)

    # now the input args
    chunk_size = 40000
    chunks = int(np.ceil(len(sv) / float(chunk_size)))

    args = []
    for i in range(chunks):
        args += [  (i*chunk_size, (i+1)*chunk_size)]

    CN = chunks
    results = []
    print "MAPPING TO THE CLOUD" 
    jids = cloud.map(picloud_score_frame, [dataset_name]*CN,
                     [x_range]*CN, [y_range]*CN, 
                     [phi_range]*CN, [theta_range]*CN, 
                     args, [frame]*CN,  [EO]*CN, 
                     _type='f2', _vol="my-vol", _env="base/precise")

    np.savez_compressed(outfile_npz, 
                        x_range = x_range, y_range=y_range, 
                        phi_range = phi_range, theta_range = theta_range)
    pickle.dump({'frame' : frame, 
                 'dataset_name' : dataset_name, 
                 'dataset_dir' : dataset_dir, 
                 'jids' : jids}, open(outfile_wait, 'w'))


@transform(score_frame_queue, regex(r"(.+).wait.(.+)$"), [r"\1.pickle", r"\1.npz"])
def score_frame_wait((infile_wait, infile_npz), (outfile_pickle, outfile_npz)):
    dnpz = np.load(infile_npz)
    p = pickle.load(open(infile_wait))
    
    jids = p['jids']

    results = cloud.result(jids)
    scores = np.concatenate(results)
    np.savez_compressed(outfile_npz, scores=scores, **dnpz)
    pickle.dump(p, open(outfile_pickle, 'w'))
    # save the results
    
#     data = np.load(infile)
#     scores = data['scores']
#     phi_range = data['phi_range']
#     x = int(np.ceil(np.sqrt(len(phi_range))))


@transform(score_frame_wait, suffix(".pickle"), [".png", ".hist.png"])
def plot_likelihood((infile_pickle, infile_npz),
                    (outfile, outfile_hist)):
    data = np.load(infile_npz)
    data_p = pickle.load(open(infile_pickle))
    scores = data['scores']

    sv = create_state_vect(data['y_range'], data['x_range'], 
                           data['phi_range'], data['theta_range'])

    scores = scores[:len(sv)]

    pylab.figure()
    pylab.hist(scores.flat, bins=255)
    pylab.savefig(outfile_hist, dpi=300)

    TOP_R, TOP_C = 3, 4
    TOP_N = TOP_R * TOP_C

    score_idx_sorted = np.argsort(scores)[::-1]
    
    #get the frame
    frames = organizedata.get_frames(data_p['dataset_dir'], 
                                     np.array([data_p['frame']]))

    # config file
    cf = pickle.load(open(os.path.join(data_p['dataset_dir'], 
                                       'config.pickle')))
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])

    img = frames[0]
    f = pylab.figure()
    for r in range(TOP_N):
        s_i = score_idx_sorted[r]
        score = scores[s_i]
        ax =f.add_subplot(TOP_R, TOP_C, r+1)
        ax.imshow(img, interpolation='nearest', cmap=pylab.cm.gray)
        x_pix, y_pix = env.gc.real_to_image(sv[s_i]['x'], sv[s_i]['y'])
        ax.axhline(y_pix, linewidth=1, c='b', alpha=0.5)
        ax.axvline(x_pix, linewidth=1, c='b', alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    f.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.1, wspace=.1)
    f.savefig(outfile, dpi=300)

@transform(score_frame_wait, suffix(".pickle"), [".zoom.png"])
def plot_likelihood_zoom((infile_pickle, infile_npz),
                         (zoom_outfile, )):
    """
    zoom in on the region of interest
    plot front and back diodes
    """
    data = np.load(infile_npz)
    data_p = pickle.load(open(infile_pickle))
    scores = data['scores']

    sv = create_state_vect(data['y_range'], data['x_range'], 
                           data['phi_range'], data['theta_range'])

    scores = scores[:len(sv)]

    TOP_R, TOP_C = 3, 4
    TOP_N = TOP_R * TOP_C

    score_idx_sorted = np.argsort(scores)[::-1]
    
    #get the frame
    frames = organizedata.get_frames(data_p['dataset_dir'], 
                                     np.array([data_p['frame']]))

    # config file
    cf = pickle.load(open(os.path.join(data_p['dataset_dir'], 
                                       'config.pickle')))
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])
    eo = likelihood.EvaluateObj(*cf['frame_dim_pix'])
    led_params = pickle.load(open(os.path.join(data_p['dataset_dir'], 
                                               "led.params.pickle")))

    EO_PARAMS = measure.led_params_to_EO(cf, led_params)

    eo.set_params(*EO_PARAMS)

    img = frames[0]
    img_thold = img.copy()
    img_thold[img < PIX_THRESHOLD] = 0
    f = pylab.figure()
    X_MARGIN = 30
    Y_MARGIN = 20
    for row in range(TOP_R):
        for col in range(TOP_C):
            r = row * TOP_C + col
            s_i = score_idx_sorted[r]
            score = scores[s_i]
            ax = pylab.subplot2grid((TOP_R *2, TOP_C), (row*2, col))
            ax.imshow(img, interpolation='nearest', cmap=pylab.cm.gray)
            ax_thold = pylab.subplot2grid((TOP_R*2, TOP_C), 
                                      (row*2+1, col))
            ax_thold.imshow(img_thold, interpolation='nearest', 
                            cmap=pylab.cm.gray)
            x = sv[s_i]['x']
            y = sv[s_i]['y']
            phi = sv[s_i]['phi']
            theta = sv[s_i]['theta']

            x_pix, y_pix = env.gc.real_to_image(x, y)

            # now compute position of diodes
            front_pos, back_pos = util.compute_pos(eo.length, x_pix, y_pix, 
                                                   phi, theta)

            cir = pylab.Circle(front_pos, radius=EO_PARAMS[1],  ec='g', fill=False,
                               linewidth=2)
            ax.add_patch(cir)
            cir = pylab.Circle(back_pos, radius=EO_PARAMS[2],  ec='r', fill=False, 
                               linewidth=2)
            ax.add_patch(cir)
            for a in [ax, ax_thold]:
                a.set_xticks([])
                a.set_yticks([])
                a.set_xlim(x_pix - X_MARGIN, x_pix+X_MARGIN)
                a.set_ylim(y_pix - Y_MARGIN, y_pix+Y_MARGIN)
    f.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.1, wspace=.1)
    f.savefig(zoom_outfile, dpi=300)


# @files(frame_params)
# def examine_frame((epoch_dir, epoch_config_filename, epoch_region_filename,
#             frame_hist_filename), outfile, epoch, frame):
#     np.random.seed(0)
    
#     cf = pickle.load(open(epoch_config_filename))
#     framehist = np.load(frame_hist_filename)
#     region = pickle.load(open(epoch_region_filename))

#     frames = organizedata.get_frames(epoch_dir, np.array([frame]))
#     frame = frames[0]
#     frame_hist = framehist['hist']
#     frame_mean = np.mean(frame_hist, axis=2)

#     env = util.Environmentz(cf['field_dim_m'], 
#                             cf['frame_dim_pix'])
    
#     region_lower_corner = env.gc.real_to_image(region['x_pos_min'], 
#                                              region['yXb_pos_min'])
#     region_upper_corner = env.gc.real_to_image(region['x_pos_max'], 
#                                              region['y_pos_max'])

#     maxpoint = np.argmax(frame)
#     max_y, max_x = np.unravel_index(maxpoint, frame.shape)

#     pylab.figure()
#     pylab.subplot(3, 2, 1)
#     pylab.imshow(frame, interpolation='nearest')
#     pylab.axhline(max_y, c='r', alpha=0.3)
#     pylab.axvline(max_x, c='g', alpha=0.3)

#     pylab.subplot(3, 2, 3)
#     pylab.plot(frame[max_y, :], 'r')
#     pylab.plot(frame[:, max_x], 'g')

#     # attempt to select all pix that are outside their std dev
#     pylab.subplot(3, 2, 5)
#     pylab.imshow(frame, interpolation='nearest', cmap=pylab.cm.gray)
    
#     pylab.gca().add_patch(pylab.Rectangle(region_lower_corner,
#                               region_upper_corner[0] - region_lower_corner[0], 
#                               region_upper_corner[1] - region_lower_corner[1], fill=False, linewidth=2, edgecolor='r'))
#     for thold_i, thold in enumerate([250, 255]):
#         #outlier_mask = frame > (frame_mean + 2*frame_std)
#         high_mask = frame >= thold

#         tgt_mask = high_mask # np.logical_and(high_mask, outlier_mask)
#         frame2 = frame.copy()

#         S = 6
#         a = scipy.ndimage.binary_dilation(tgt_mask, 
#                                           structure=np.ones((S, S)))
#         frame2[np.logical_not(a)] = 0

#         pylab.subplot(3, 2, 2*thold_i + 2)
#         pylab.imshow(frame2, interpolation='nearest', cmap=pylab.cm.gray)

#     # # now look at entropy of spots
#     # ent = np.zeros((frame_hist.shape[0], frame_hist.shape[1]))
#     # for r in range(frame_hist.shape[0]):
#     #     for c in range(frame_hist.shape[1]):
#     #         v = frame_hist[r, c]
#     #         not_zeros = v > 0
#     #         p = v / np.sum(v)
            
#     #         ent[r, c] = -np.sum(p[not_zeros] * np.log2(p[not_zeros]))
#     # pylab.subplot(3, 2, 6)
#     # pylab.imshow(ent)
#     pylab.savefig(outfile, dpi=400)

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

def create_state_vect(y_range, x_range, phi_range, theta_range):
    N = len(y_range) * len(x_range) * len(phi_range) * len(theta_range)

    state = np.zeros(N, dtype=util.DTYPE_STATE)
    
    i = 0
    for yi, y in enumerate(y_range):
        for xi, x in enumerate(x_range):
            for phii, phi in enumerate(phi_range):
                for thetai, theta in enumerate(theta_range):
                    state['x'][i] = x
                    state['y'][i] = y
                    state['phi'][i] = phi
                    state['theta'][i] = theta
                    i += 1
    return state

def picloud_score_frame(dataset_name, x_range, y_range, phi_range, theta_range,
                        state_idx, frame, EO_PARAMS):
    """
    pi-cloud runner, every instance builds up full state, but
    we only evaluate the states in [state_idx_to_eval[0], state_idx_to_eval[1])
    and return scores
    """
    print "DATSET_NAME=", dataset_name
    dataset_dir = os.path.join(FL_DATA, dataset_name)
    dataset_config_filename = os.path.join(dataset_dir, "config.pickle")
    dataset_region_filename = os.path.join(dataset_dir, "region.pickle")
    frame_hist_filename = os.path.join(dataset_dir, "framehist.npz")
    
    np.random.seed(0)
    
    cf = pickle.load(open(dataset_config_filename))
    region = pickle.load(open(dataset_region_filename))

    framehist = np.load(frame_hist_filename)
    
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])

    eo = likelihood.EvaluateObj(*cf['frame_dim_pix'])
    eo.set_params(*EO_PARAMS)
    
    le = likelihood.LikelihoodEvaluator(env, eo)

    frames = organizedata.get_frames(dataset_dir, np.array([frame]))
    frame = frames[0]
    frame[frame < PIX_THRESHOLD] = 0
    # create the state vector

    state = create_state_vect(y_range, x_range, phi_range, theta_range)
    
    SCORE_N = state_idx[1] - state_idx[0]
    scores = np.zeros(SCORE_N, dtype=np.float32)
    for i, state_i in enumerate(state[state_idx[0]:state_idx[1]]):
        x = state_i['x']
        y = state_i['y']
        if region['x_pos_min'] <= x <= region['x_pos_max'] and \
                region['y_pos_min'] <= y <= region['y_pos_max']:
        
            score = le.score_state(state_i, frames)
            scores[i] = score
        else:
            scores[i] = -1e100
    return scores

if __name__ == "__main__":
    pipeline_run([score_frame_wait, plot_likelihood, 
                  plot_likelihood_zoom], multiprocess=4)
