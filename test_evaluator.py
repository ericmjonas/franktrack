import numpy as np
from nose.tools import * 
import cPickle as pickle
from matplotlib import pylab

import likelihood
import util
import model

def test_le():
    d = pickle.load(open('simulate.pickle'))
    
    env = util.Environment((1.5, 2), (240, 320))
    
    eo = likelihood.EvaluateObj(240, 320)
    eo.set_params(10, 4, 2)
    le = likelihood.LikelihoodEvaluator(env, eo)

    img = d['video'][0]
    s = d['state'][0]
    print d['state'][0]
    print d['state'][1]
    
    print le.score_state(s, img)
    s['x'] += 0.01
    print le.score_state(s, img)
