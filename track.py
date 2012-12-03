import numpy as np
import cPickle as pickle
import os
import measure
from ruffus import * 

DATA_DIR = "data/fl"
def ddir(x):
    return os.path.join(DATA_DIR, x)

REPORT_DIR = "results"
def rdir(x):
    return os.path.join(REPORT_DIR, x)

DTYPE_POS_CONF = [('x', np.float32), 
                  ('y', np.float32), 
                  ('confidence', np.float32)]

def truth(basedir):
    """
    Returns the ground truth, interpolated and everything, for a given
    data directory
    """
    positions_file = os.path.join(basedir, "positions.npy")
    positions = np.load(positions_file)

    cf = pickle.load(open(os.path.join(basedir, "config.pickle")))

    invalid_sep = measure.detect_invalid_sep(positions)
    positions_cleaned = positions.copy()
    positions_cleaned[invalid_sep] = ((np.nan, np.nan), 
                                      (np.nan, np.nan), np.nan, np.nan)
    
    positions_interp, missing = measure.interpolate(positions)
    pci, cleaned_missing = measure.interpolate(positions_cleaned)

    # extract out the fields that matter
    d = np.zeros(len(pci), dtype=DTYPE_POS_CONF)
    d['x'] = pci['x']
    d['y'] = pci['y']
    d['confidence'] = 1.0

    return d

def current_method(basedir):
    positions_file = os.path.join(basedir, "positions.npy")
    positions = np.load(positions_file)

    conf = np.ones(len(positions))
    conf[np.isnan(positions['x'])] = 0.0
    # since this requires manual input, these empty spots will be surrounded
    # by valid positions
    conf_exp = conf.copy()
    zeros = np.argwhere(conf < 0.9)[1:-1] # round off edge cases
    conf_exp[zeros+1] = 0.0
    conf_exp[zeros-1] = 0.0

    d = np.zeros(len(positions), dtype=DTYPE_POS_CONF)
    # now we do a trick

    d['x'] = positions['x']
    d['y'] = positions['y']
    d['x'][conf_exp == 0.0] = 0.0
    d['y'][conf_exp == 0.0] = 0.0

    d['confidence'] = conf_exp
    assert np.isnan(d['x']).all() == False
    return d

@transform(ddir("*/positions.npy"), 
           regex(r".+/(.+)/positions.npy$"), 
           [os.path.join(REPORT_DIR, 
                         r"\1", "truth.npy")], 
           r"\1"
           )
def get_truth(positions_file, (output_file, ), basedir):
    truth_data = truth(ddir(basedir))
    os.makedirs(rdir(basedir))
    np.save(output_file, truth_data)

@transform(ddir("*/positions.npy"), 
           regex(r".+/(.+)/positions.npy$"), 
           [os.path.join(REPORT_DIR, 
                         r"\1", "algo", "current.npy")], 
           r"\1"
           )
def get_algo_current(positions_file, (output_file, ), basedir):
    algo_data = current_method(ddir(basedir))
    try:
        os.makedirs(rdir(os.path.join(basedir, "algo")))
    except OSError:
        pass

    np.save(output_file, algo_data)

if __name__ == "__main__":
    pipeline_run([get_truth, get_algo_current])
