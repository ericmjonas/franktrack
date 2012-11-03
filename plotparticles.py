import numpy as np
import cPickle as pickle
from matplotlib import pylab
import util2 as util

def plot_particles(PARTICLE_FILENAME, TRUTH_FILENAME, OUT_FILENAME):
    T_DELTA = 1/30.

    a = np.load(PARTICLE_FILENAME)
    truth = pickle.load(open(TRUTH_FILENAME, 'r'))
    weights = a['weights']
    particles = a['particles']
    truth_state = truth['state'][:len(particles)]

    STATEVARS = ['x', 'y', 'xdot', 'ydot', 'phi', 'theta']
    vals = dict([(x, []) for x in STATEVARS])
    for p in particles:
        for v in STATEVARS:
            vals[v].append([s[v] for s in p])
    for v in STATEVARS:
        vals[v] = np.array(vals[v])

    pylab.figure()
    for vi, v in enumerate(STATEVARS):


        if 'dot' in v:
            v_truth = np.diff(truth_state[v[0]])/T_DELTA
        else:
            v_truth = truth_state[v]

        v_bar = np.average(vals[v], axis=1, weights=weights)
        x = np.arange(0, len(v_bar))
        cred = np.zeros((len(x), 2), dtype=np.float)
        for ci, (p, w) in enumerate(zip(vals[v], weights)):
            cred[ci] = util.credible_interval(p, w)

        pylab.subplot(len(STATEVARS) + 1,1, 1+vi)
        pylab.plot(v_truth, color='g', linestyle='-', 
                   linewidth=1)

        pylab.plot(x, v_bar, color='b')
        pylab.fill_between(x, cred[:, 0],
                           cred[:, 1], facecolor='b', 
                           alpha=0.4)


        pylab.ylim([np.min(v_truth), 
                    np.max(v_truth)])
        
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

    pylab.savefig(OUT_FILENAME)


