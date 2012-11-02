import numpy as np
import os
import glob
from ruffus import * 
import subprocess
import frankdata as fl
import tarfile
import cPickle as pickle
from matplotlib import pylab


"""
Measure various properties about the frank lab data, synthetic data, 
etc. 
"""

DATA_DIR = "data/fl"

FIGS_DIR = "figs"


def interpolate(positions):
    """
    Take in a positions vector and perform linear interpolation on it
    """

    t = np.arange(0, len(positions))
    missing_bool = np.isnan(positions['led_front'][:, 0])
    missing = np.argwhere(missing_bool)[:, 0]
    present = np.argwhere(np.logical_not(missing_bool))[:, 0]
    
    new_val = np.copy(positions)
    for l in ['led_front', 'led_back']:
        for i in range(2):
            interp_vals = np.interp(t[missing], t[present], 
                                      positions[l][:, i][present])
            new_val[l][missing, i] = interp_vals

    for l in ['x', 'y']:
            interp_vals = np.interp(t[missing], t[present], 
                                      positions[l][present])
            new_val[l][missing] = interp_vals.astype(np.float32)
    return new_val, missing

def compute_theta(front, back):
    """
    Unwrap does NOT handle nan well, so be careful and use the interpolated
    diode positions
    
    """
    d = front - back
    return np.unwrap(np.arctan2(d[:, 0], d[:, 1]))

    
def compute_derived(positions_interp):
    """
    Return velocity, theta, angular velocity
    
    """
    
    unwrapped_theta = compute_theta(positions_interp['led_front'], 
                                    positions_interp['led_back'])
    
    omega = np.diff(unwrapped_theta)

    xdot = np.diff(positions_interp['x'])
    ydot = np.diff(positions_interp['y'])
    
    return {'xdot' : xdot, 'ydot' : ydot, 
            'thetadot' : omega, 
            'theta' : unwrapped_theta}


@merge("%s/*/positions.npy" % DATA_DIR, ["velocity.pdf", "thetadot.pdf"])
def agg_stats(input_files, outfile):
    derived = []
    missings = []
    DELTA_T = 1/30.
    for f in input_files:
        positions = np.load(f)

        positions_interp, missing = interpolate(positions)
        missings.append((len(positions), len(missing)))
        pos_derived = compute_derived(positions_interp)
        derived.append(pos_derived)
    concats = {}
    for var in ['xdot', 'ydot', 'thetadot', 'theta']:
        c = np.concatenate([d[var] for d in derived])
        concats[var] = c

    # compute velocity vector magnitude
    vdot = np.sqrt(concats['xdot']**2 + concats['ydot']**2)
    f = pylab.figure()
    v_sorted = np.sort(vdot)
    v_99 = v_sorted[:len(v_sorted)*0.99]
    pylab.hist(v_99/DELTA_T*100, bins=10, 
               normed=True)
    pylab.xlabel('cm/sec')
    pylab.title('magnitude of velocity')
    pylab.savefig(outfile[0])
    
    f = pylab.figure()
    pylab.subplot(2, 1, 1)
    td_sorted = np.sort(np.abs(concats['thetadot']))
    td_99 = td_sorted[:len(td_sorted)*0.99]
    pylab.hist(td_99/DELTA_T, bins=50, 
               normed=True)
    pylab.xlabel('rad/sec')
    pylab.title('magnitude of angular velocity')
    pylab.subplot(2,1, 2)
    td_20 = td_sorted[0:len(td_sorted)*0.40]
    pylab.hist(td_20/DELTA_T, bins=50, 
               normed=True)
    pylab.xlabel('rad/sec')
    pylab.title('mag of angular velocity, lowest 30%')
    pylab.savefig(outfile[1])


pipeline_run([agg_stats])
    
    
    
    
    
