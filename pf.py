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

def resample_multinomial(particles, weights):
    """
    Numpy-based multinomial sampling
    """

    N = len(particles)

    r = np.random.multinomial(N, weights)

    new_particles = []

    for i, v in enumerate(r):

        for vi in range(v):
            new_particles.append(particles[i])

    return new_particles

def normalize(x):
    return util.scores_to_prob(x)

#@profile
def particle_filter(y, model, N, PARTICLE_N = 100):
    #particles = np.zeros((N, PARTICLE_N))
    particles = []
    weights = np.zeros((N, PARTICLE_N))

    #resampled_weights = np.zeros(N)

    def weight_calc(y, x_n):
        """
        Simple weighting by the likelihood
        """
        if y == None:
            return 0.0
        else:
            return model.score_obs(y, x_n)
    
    # N=1 case:
    # sample initial particle set
    particles.append([model.sample_latent_from_prior() for _ in range(PARTICLE_N)])

    # compute weights
    for i in range(PARTICLE_N):
        weights[0, i] = weight_calc(y[0], particles[0][i])

    # normalize
    weights[0] = normalize(weights[0])

    # resample
    resampled_particles = resample_multinomial(particles[0], weights[0])
    for n in range(1, N):
        t1 = time.time()
        proposed_particles = []
        for i in range(PARTICLE_N):
            p = model.sample_next_latent(resampled_particles[i], n)
            proposed_particles.append(p)
            weights[n, i] = weight_calc(y[n], p)
        weights[n] = normalize(weights[n])
        particles.append(proposed_particles)
        # if n > 10:
        #     STATEVARS = ['x', 'y', 'xdot', 'ydot', 'phi', 'theta']
        #     for vi, v in enumerate(STATEVARS):
        #         pylab.subplot(len(STATEVARS) + 1,1, 1+vi)
        #         pylab.hist([q[v] for q in proposed_particles], bins=20)
        #     pylab.show()

        resampled_particles = resample_multinomial(proposed_particles, weights[n])
        t2 = time.time()
        print "n=", n, "%3.1f secs" % (t2-t1), (t2-t1)*(N-n), "remaining"
    return weights, particles
