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

from ruffus import * 
import pf

def params():
    PARTICLEN = 2000
    FRAMEN = 10000
    NOISE = 255
    for posnoise in [0.005]:
        for velnoise in [0.05, 0.08]:
            for log in [True, False]:
                infile = 'data/synth/circle.%03d.pickle'  % NOISE
                outfile = 'particles.%f.%f.%d.%d.%d.%d.npz' % (posnoise, velnoise, 
                                                            PARTICLEN, FRAMEN, NOISE, log)
                
                yield (infile, outfile, posnoise, velnoise, PARTICLEN, FRAMEN, 
                   NOISE, log)

@files(params)
def pf_run(infile, outfile, posnoise, velnoise, PARTICLEN, 
                    FRAMEN, NOISE, log):
    np.random.seed(0)
    print "Loading data..."
    d = pickle.load(open(infile))
    print "done!" 
    env = util.Environmentz((1.5, 2), (240, 320))

    eo = likelihood.EvaluateObj(240, 320)
    eo.set_params(10, 4, 2)
    le = likelihood.LikelihoodEvaluator(env, eo, log)

    model_inst = model.LinearModel(env, le, 
                                   POS_NOISE_STD=posnoise,
                                   VELOCITY_NOISE_STD=velnoise)

    y = d['video'][:FRAMEN]

    weights, particles = pf.particle_filter(y, model_inst, 
                                            len(y), PARTICLEN)
    np.savez_compressed(outfile, 
                        weights=weights, particles=particles)

def params_rendered():
    for p in params():
        yield ([p[0], p[1]], p[1] + ".pdf")

@follows(pf_run)
@files(params_rendered)
def pf_plot((orig_data, particles), plot_file):
    plotparticles.plot_particles(particles, orig_data, plot_file)


pipeline_run([pf_run, pf_plot], multiprocess=4)
