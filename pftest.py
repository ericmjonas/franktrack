import numpy as np
import cPickle as pickle
from matplotlib import pylab

import likelihood
import util
import model

np.random.seed(0)

d = pickle.load(open('simulate.pickle'))

env = util.Environment((1.5, 2), (240, 320))

eo = likelihood.EvaluateObj(240, 320)
eo.set_params(10, 4, 2)
le = likelihood.LikelihoodEvaluator(env, eo)

model_inst = model.LinearModel(env, le)

PARTICLEN = 4000

for frameno in range(1, 300, 30):
    print "frame", frameno
    prior_states = model_inst.sample_latent_from_prior(PARTICLEN)

    img = d['video'][frameno]

    scores = np.zeros(PARTICLEN)

    for si, state in enumerate(prior_states):
        scores[si] = model_inst.score_obs(img, state)

    score_sort_idx = np.argsort(scores)

    scores_sorted = scores[score_sort_idx]
    states_sorted = prior_states[score_sort_idx]

    f = pylab.figure()
    ax1 = f.add_subplot(1, 1, 1)
    ax1.imshow(img, interpolation='nearest', origin='lower', 
                 cmap = pylab.cm.gray)

    for i in range(1, 10):
        best_state = states_sorted[-i]

        pix_x, pix_y = env.gc.real_to_image(best_state['x'], best_state['y'])
        pylab.axhline(pix_y)
        pylab.axvline(pix_x)

    f.savefig('good_particles.%04d.png' % frameno)
