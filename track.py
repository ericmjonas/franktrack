import numpy as np
import cPickle as pickle
import os
import util2 as util
from util2 import ddir, rdir
import measure
import methods
import organizedata
from ruffus import * 


DTYPE_POS_CONF = [('x', np.float32), 
                  ('y', np.float32), 
                  ('confidence', np.float32)]

FRAMES_TO_ANALYZE = 15000 # Analyze this many frames in each epoch

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



def per_frame(basedir, func, config):
    config_file = os.path.join(basedir, "config.pickle")
    cf = pickle.load(open(config_file))
    env = util.Environmentz(cf['field_dim_m'], cf['frame_dim_pix'])
    FRAMEN = cf['end_f'] - cf['start_f'] + 1
    

    d = np.zeros(FRAMES_TO_ANALYZE, dtype=DTYPE_POS_CONF)
    FRAMES_AT_A_TIME = 10
    frames = np.arange(FRAMES_TO_ANALYZE)
    for frame_subset in util.chunk(frames, FRAMES_AT_A_TIME):
        fs = organizedata.get_frames(basedir, frame_subset)
        for fi, frame_no in enumerate(frame_subset):
            real_x, real_y, conf = func(fs[fi], env, **config)
            d[frame_no]['x'] = real_x
            d[frame_no]['y'] = real_y
            d[frame_no]['confidence'] = conf
            
    return d


@transform(ddir("*/positions.npy"), 
           regex(r".+/(.+)/positions.npy$"), 
           [os.path.join(REPORT_DIR, 
                         r"\1", "truth.npy")], 
           r"\1"
           )
def get_truth(positions_file, (output_file, ), basedir):
    truth_data = truth(ddir(basedir))
    try:
        os.makedirs(rdir(basedir))
    except OSError:
        pass
    np.save(output_file, truth_data)

def algodir(basedir):
    try:
        os.makedirs(rdir(os.path.join(basedir, "algo")))
    except OSError:
        pass
    
@transform(ddir("*/positions.npy"), 
           regex(r".+/(.+)/positions.npy$"), 
           [os.path.join(REPORT_DIR, 
                         r"\1", "algo", "current.npy")], 
           r"\1"
           )
def get_algo_current(positions_file, (output_file, ), basedir):
    algo_data = current_method(ddir(basedir))
    algodir(basedir)
    algo_data = current_method(ddir(basedir))
                          
    np.save(output_file, algo_data)

@transform(ddir("*/positions.npy"), 
           regex(r".+/(.+)/positions.npy$"), 
           [os.path.join(REPORT_DIR, 
                         r"\1", "algo", "centroid.npy")], 
           r"\1"
           )
def get_algo_centroid(positions_file, (output_file, ), basedir):
    print "basedir=", basedir, "ddir(basedir)=", ddir(basedir)
    algodir(ddir(basedir))
    algo_data = per_frame(ddir(basedir), 
                          methods.centroid_frame, {'thold' : 240})
    
    np.save(output_file, algo_data)

if __name__ == "__main__":
    pipeline_run([get_truth, get_algo_current, get_algo_centroid], 
                 multiprocess=4)
