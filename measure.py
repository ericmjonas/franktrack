import numpy as np
import os
import glob
from ruffus import * 
import util2 as util
import subprocess
import frankdata as fl
import tarfile
import cPickle as pickle
from matplotlib import pylab
import organizedata

"""
Measure various properties about the frank lab data, synthetic data, 
etc. 
"""

DATA_DIR = "data/fl"

FIGS_DIR = "figs"

def compute_xy_from_leds(positions):
    t = np.arange(0, len(positions))
    missing_bool = np.isnan(positions['led_front'][:, 0])
    present = np.argwhere(np.logical_not(missing_bool))[:, 0]
    
    xyvals = np.zeros(len(t), dtype=[('x', np.float32), 
                                     ('y', np.float32)])
    
    xyvals['x'] = np.mean([positions['led_front'][:, 0], 
                           positions['led_back'][:, 0]])

    xyvals['y'] = np.mean([positions['led_front'][:, 1], 
                           positions['led_back'][:, 1]])

    return xyvals

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

def compute_phi(front, back):
    """
    Unwrap does NOT handle nan well, so be careful and use the interpolated
    diode positions
    
    """
    d = front - back
    return np.unwrap(np.arctan2(d[:, 0], d[:, 1]))

    
def compute_derived(positions_interp):
    """
    Return velocity, phi, angular velocity
    
    """
    
    unwrapped_phi = compute_phi(positions_interp['led_front'], 
                                    positions_interp['led_back'])
    
    omega = np.diff(unwrapped_phi)

    xdot = np.diff(positions_interp['x'])
    ydot = np.diff(positions_interp['y'])
    
    return {'xdot' : xdot, 'ydot' : ydot, 
            'phidot' : omega, 
            'phi' : unwrapped_phi}

def compute_led_sep(positions):
    delta_x = positions['led_front'][:, 0] - positions['led_back'][:, 0]
    delta_y = positions['led_front'][:, 1] - positions['led_back'][:, 1]
    delta = np.sqrt(delta_x**2 + delta_y**2)
    return delta

def detect_invalid_sep(positions, thold_std= 4):
    """
    Take in the raw positions file and return where the LED sep is too high
    
    """

    sep = compute_led_sep(positions)
    sep_std = np.std(sep[np.isnan(sep) == False])
    sep_mean = np.mean(sep[np.isnan(sep) == False])
    return np.argwhere(sep > (sep_std*thold_std + sep_mean))[:, 0]


REPORT_DIR = "figs"

@transform(os.path.join(DATA_DIR, "*/positions.npy"), 
           regex(r".+/(.+)/positions.npy$"), 
           [os.path.join(REPORT_DIR, 
                         r"\1.ledsep.png"), 
            os.path.join(REPORT_DIR, 
                         r"\1.velocity.png")]
            )
def sanity_check(positions_file, (led_sep, velocity_file)):
    positions = np.load(positions_file)
    basedir = os.path.dirname(positions_file)
    cf = pickle.load(open(os.path.join(basedir, "config.pickle")))

    ### distributions of LED separation
    ### remove the outliers
    good_idx = np.argwhere(np.logical_not(np.isnan(positions['led_front'][:, 0])))[:, 0]
    
    delta = compute_led_sep(positions[good_idx]) * 100
    invalid_sep = detect_invalid_sep(positions)
    positions[invalid_sep] = ((np.nan, np.nan), 
                              (np.nan, np.nan), np.nan, np.nan)

    ## Plot the results
    pylab.figure()
    pylab.subplot(4, 1, 1)
    pylab.plot(good_idx, delta)
    pylab.scatter(invalid_sep, np.zeros(len(invalid_sep)), 
                  c='k')
    pylab.ylabel('cm')
    pylab.xlim([0, len(delta)])

    pylab.subplot(4, 1, 2)

    pylab.hist(delta, bins=50)
    pylab.ylabel('cm')


    positions_interp, missing = interpolate(positions)
    pos_derived = compute_derived(positions_interp)
    led_front_delta = np.diff(positions_interp['led_front'], axis=0)
    led_front_vel = np.sqrt(np.sum(led_front_delta**2, axis=1))

    led_back_delta = np.diff(positions_interp['led_back'], axis=0)
    led_back_vel = np.sqrt(np.sum(led_back_delta**2, axis=1))
    
    pos_delta_x = np.diff(positions_interp['x'])
    pos_delta_y = np.diff(positions_interp['y'])
    pos_vel = np.sqrt(pos_delta_x**2 + pos_delta_y**2)
    
    pylab.subplot(4, 1, 3)
    pylab.plot(led_front_vel[good_idx[:-1]], c='g')
    pylab.plot(led_back_vel[good_idx[:-1]], c='r')
    pylab.plot(pos_vel[good_idx[:-1]], c= 'b')
    pylab.xlim([0, len(delta)])
    pylab.savefig(led_sep, dpi=300)


    # now the real velocity outliers: For the top N velocity outliers,
    # plot the image before, during, and after the peak
    
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])

    
    def rc(x, y):
        # convenience real-to-immg
        return env.gc.real_to_image(x, y)
    top_vel_idx = np.argsort(pos_vel)[-1]
    tgt_frame_idx = np.array([top_vel_idx -2, 
                              top_vel_idx - 1,
                              top_vel_idx, 
                              top_vel_idx + 1, 
                              top_vel_idx + 2])
    f = organizedata.get_frames(basedir, tgt_frame_idx)
    pylab.figure()
    for i, fi in enumerate(tgt_frame_idx):
        pylab.subplot(2, len(tgt_frame_idx), 
                      i+1)
        pylab.imshow(f[i], interpolation='nearest', cmap=pylab.cm.gray)
        for l, c in [('led_front', 'g'), ('led_back', 'r')]:
            img_x, img_y = rc(positions_interp[l][fi, 0],
                              positions_interp[l][fi, 1])
            pylab.scatter(img_x, img_y, c=c, s=1, linewidth=0)
    for i, fi in enumerate(tgt_frame_idx[:-1]):
        pylab.subplot(2, len(tgt_frame_idx), 
                      len(tgt_frame_idx) + i+1)
        pylab.imshow(f[i+1] -f[i], interpolation='nearest')
    pylab.savefig(velocity_file, dpi=1000)
    
    
@merge("%s/*/positions.npy" % DATA_DIR, ["velocity.pdf", "phidot.pdf"])
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
    for var in ['xdot', 'ydot', 'phidot', 'phi']:
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
    td_sorted = np.sort(np.abs(concats['phidot']))
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


@transform(os.path.join(DATA_DIR, "*/positions.npy"), 
           regex(r".+/(.+)/positions.npy$"), 
           [os.path.join(REPORT_DIR, 
                         r"\1.hardtrackingregion.png")]
            )
def difficult_regions(positions_file, (hard_tracking_region, )):
    positions = np.load(positions_file)
    basedir = os.path.dirname(positions_file)
    cf = pickle.load(open(os.path.join(basedir, "config.pickle")))

    invalid_sep = detect_invalid_sep(positions)
    positions_cleaned = positions.copy()
    positions_cleaned[invalid_sep] = ((np.nan, np.nan), 
                                      (np.nan, np.nan), np.nan, np.nan)
    
    positions_interp, missing = interpolate(positions)
    pci, cleaned_missing = interpolate(positions_cleaned)
    
    # find the regions of NA
    
    f = pylab.figure()
    pylab.subplot(2, 1, 1)
    pylab.plot(pci['x'], pci['y'], c='k', alpha=0.3)
    pylab.scatter(pci['x'][missing], 
                  pci['y'][missing], 
                  linewidth='0', s=3, c='b', 
                  label='omitted pos')
    pylab.scatter(pci['x'][invalid_sep], 
                  pci['y'][invalid_sep],
                  linewidth='0', s=6, c='r', 
                  label='velocity error')
    pylab.xlim((np.min(pci['x']), np.max(pci['x'])))
    pylab.ylim((np.min(pci['y']), np.max(pci['y'])))
    pylab.xlabel('x pos')
    pylab.ylabel('y pos')
    pylab.title(hard_tracking_region)

    pylab.subplot(2, 1, 2)
    pylab.axhline(0)
    pylab.scatter(missing, np.ones_like(missing)*-0.2, c='b', 
                  marker='|')
    pylab.scatter(invalid_sep, np.ones_like(invalid_sep)*-0.2, c='r', 
                  marker='|')
    
    # chunk and figure out what fraction of latest had missing points
    x = np.zeros(len(pci))
    x[cleaned_missing] = 1.0
    res = np.convolve(x, np.ones(100)/100)
    pylab.plot(res)

    pylab.xlim(0, len(positions))
    pylab.xlabel('time')
    pylab.ylabel('fraction error')
    pylab.savefig(hard_tracking_region, dpi=300)

    
@transform(os.path.join(DATA_DIR, "*/framehist.npz"), 
           regex(r".+/(.+)/framehist.npz$"), 
           [os.path.join(REPORT_DIR, 
                         r"\1.entropy.png")]
            )
def entropy_vs_pos(positions_file, (pix_entropypix,)):
    positions = np.load(positions_file)
    basedir = os.path.dirname(positions_file)
    cf = pickle.load(open(os.path.join(basedir, "config.pickle")))

    invalid_sep = detect_invalid_sep(positions)
    positions_cleaned = positions.copy()
    positions_cleaned[invalid_sep] = ((np.nan, np.nan), 
                                      (np.nan, np.nan), np.nan, np.nan)
    
    positions_interp, missing = interpolate(positions)
    pci, cleaned_missing = interpolate(positions_cleaned)
    
    # find the regions of NA
    
    f = pylab.figure()
    pylab.subplot(2, 1, 1)
    pylab.plot(pci['x'], pci['y'], c='k', alpha=0.3)
    pylab.scatter(pci['x'][missing], 
                  pci['y'][missing], 
                  linewidth='0', s=3, c='b', 
                  label='omitted pos')
    pylab.scatter(pci['x'][invalid_sep], 
                  pci['y'][invalid_sep],
                  linewidth='0', s=6, c='r', 
                  label='velocity error')
    pylab.xlim((np.min(pci['x']), np.max(pci['x'])))
    pylab.ylim((np.min(pci['y']), np.max(pci['y'])))
    pylab.xlabel('x pos')
    pylab.ylabel('y pos')
    pylab.title(hard_tracking_region)

    pylab.subplot(2, 1, 2)
    pylab.axhline(0)
    pylab.scatter(missing, np.ones_like(missing)*-0.2, c='b', 
                  marker='|')
    pylab.scatter(invalid_sep, np.ones_like(invalid_sep)*-0.2, c='r', 
                  marker='|')
    
    # chunk and figure out what fraction of latest had missing points
    x = np.zeros(len(pci))
    x[cleaned_missing] = 1.0
    res = np.convolve(x, np.ones(100)/100)
    pylab.plot(res)

    pylab.xlim(0, len(positions))
    pylab.xlabel('time')
    pylab.ylabel('fraction error')
    pylab.savefig(hard_tracking_region, dpi=300)

if __name__ == "__main__":
    pipeline_run([agg_stats, sanity_check, difficult_regions], 
                 multiprocess=4)
    
    
    
    
    
