import numpy as np
import cPickle as pickle
from matplotlib import pylab

NOISE = 200
T_DELTA = 1/30.
a = np.load('test.%03d.npz' % NOISE)
truth = pickle.load(open('simulate.%03d.pickle' % NOISE, 'r'))
weights = a['weights']
particles = a['particles']
truth_state = truth['state'][:len(particles)]

print particles
STATEVARS = ['x', 'y', 'xdot', 'ydot', 'phi', 'theta']
vals = dict([(x, []) for x in STATEVARS])
for p in particles:
    for v in STATEVARS:
        vals[v].append([s[v] for s in p])
for v in STATEVARS:
    vals[v] = np.array(vals[v])

for vi, v in enumerate(STATEVARS):
    v_bar = np.mean(vals[v], axis=1)
    x = np.arange(0, len(v_bar))

    v_std = np.std(vals[v], axis=1)
    pylab.subplot(len(STATEVARS) + 1,1, 1+vi)
    pylab.plot(x, v_bar, color='b')
    pylab.fill_between(x, v_bar - v_std, 
                       v_bar + v_std, facecolor='b', 
                       alpha=0.4)
    if 'dot' in v:
        pylab.plot(np.diff(truth_state[v[0]])/T_DELTA, color='g')
    else:
        pylab.plot(truth_state[v], color='g')
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
 
pylab.show()


