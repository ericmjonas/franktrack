import numpy as np
import scipy.stats
import cPickle as pickle
from matplotlib import pyplot
import likelihood
import util
import model
import time

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
    particles[0] = resample_multinomial(particles[0], weights[0])
    print particles[0]
    for n in range(1, N):
        t1 = time.time()
        new_particles = []
        for i in range(PARTICLE_N):
            p = model.sample_next_latent(particles[-1][i], n)
            new_particles.append(p)
            weights[n, i] = weight_calc(y[n], p)
        weights[n] = normalize(weights[n])

        particles.append(resample_multinomial(new_particles, weights[n]))
        t2 = time.time()
        print "n=", n, "%3.1f secs" % (t2-t1), (t2-t1)*(N-n), "remaining"
    return weights, particles

np.random.seed(0)

d = pickle.load(open('simulate.pickle'))

env = util.Environment((1.5, 2), (240, 320))

eo = likelihood.EvaluateObj(240, 320)
eo.set_params(10, 4, 2)
le = likelihood.LikelihoodEvaluator(env, eo)

model_inst = model.LinearModel(env, le)

PARTICLEN = 1000
FRAMEN = len(d['video'])
y = d['video'][:FRAMEN]

weights, particles = particle_filter(y, model_inst, FRAMEN, PARTICLEN)

np.savez_compressed('test.npz', weights=weights, particles=particles)
