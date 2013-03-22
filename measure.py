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
import scipy.signal

"""
Measure various properties about the frank lab data, synthetic data, 
etc. 
"""

DATA_DIR = "data/fl"

REPORT_DIR = "figs"

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
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))

    
def compute_derived(positions_interp, DELTA_T = 1.0, ledsep = None):
    """
    Return velocity, phi, angular velocity
    
    Specify the inter-frame time

    ledsep is the maximal distance in meters between the LEDs, 
    and will return an "always positive direction" theta 
    (that is, theta will range from 0 to pi/2)
    
    """
    
    unwrapped_phi = compute_phi(positions_interp['led_front'], 
                                positions_interp['led_back'])
    
    omega = np.diff(unwrapped_phi) / DELTA_T

    xdot = np.diff(positions_interp['x']) / DELTA_T
    ydot = np.diff(positions_interp['y']) / DELTA_T

    vals =  {'xdot' : xdot, 'ydot' : ydot, 
             'phidot' : omega, 
             'phi' : unwrapped_phi}
    
    if ledsep != None:
        d1 = (positions_interp['led_front'] - positions_interp['led_back'])
        delta = np.sqrt(np.sum(d1**2, axis=1))
        if (delta > ledsep).any():
            print "WARNING, delta > ledsep"
        delta[delta>ledsep] = ledsep
        ratio = delta / ledsep
        vals['theta'] = np.arcsin(ratio)
        print vals['theta'].shape
    return vals

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


@transform(os.path.join(DATA_DIR, "*/positions.npy"), 
           regex(r".+/(.+)/positions.npy$"), 
           [
            os.path.join(DATA_DIR, r"\1", "led.params.pickle")]
            )
def measure_diode_params(positions_file, (led_params,)):
    positions = np.load(positions_file)
    basedir = os.path.dirname(positions_file)
    cf = pickle.load(open(os.path.join(basedir, "config.pickle")))
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])


    invalid_sep = detect_invalid_sep(positions)
    positions_cleaned = positions.copy()
    positions_cleaned[invalid_sep] = ((np.nan, np.nan), 
                                      (np.nan, np.nan), np.nan, np.nan)
    
    # just take the sanitized data, don't bother interpolating. 
    # ignore all other positions
    valid_frames = np.argwhere(np.isfinite(positions_cleaned['x']))[:-3, 0]
    # the -3 above is to deal with some strange offset issues we have where len(positions) != total-number-of-frames


    PIX_REGION = 24 # num pixels on either side
    ledimgs = np.zeros((len(valid_frames), 2, PIX_REGION*2+1, 
                        PIX_REGION*2+1), dtype = np.uint8)
    
    framepos = 0
    for frame_chunk in util.chunk(valid_frames, 100):
        frames = organizedata.get_frames(basedir, frame_chunk)
        
        for frame_idx, frame in zip(frame_chunk, frames):
            for led, field in [(0, 'led_front'), 
                               (1, 'led_back')]:
                real_pos = positions_cleaned[field][frame_idx]
                x, y = env.gc.real_to_image(real_pos[0], real_pos[1])

                ledimgs[framepos, led, :, :] = util.extract_region_safe(frame, int(y), int(x), PIX_REGION, 0)
                    
            framepos +=1
    ledimgs_mean = np.mean(ledimgs.astype(np.float32), 
                      axis=0)
    subsamp = 60
    
    led_params_dict = {'ledimgs_mean' : ledimgs_mean, 
                       'dist' : compute_led_sep(positions[valid_frames]), 
                       'subsampled_frames' : ledimgs[::subsamp]}
    pickle.dump(led_params_dict, open(led_params, 'w'))

def led_measure(img_row):
    
    import scipy.stats
    midval = (np.max(img_row) - np.min(img_row)) / 2.0 + np.min(img_row)
    aw = np.argwhere(img_row > midval)[:]
    width = aw[-1] - aw[1]

    N = len(img_row)
    mu = N / 2
    sigma = width/2.0

    x = np.linspace(0, N, 100)
    a = scipy.stats.norm.pdf(x, mu, sigma)
    a = a * np.max(img_row)/np.max(a)
    
    
    return sigma, a, x

def led_params_to_EO(cf, led_params):
    """ Returns "EO" tuple of (led_dist_in_pix, front_led_radius, back_led_radius"""
    
    env = util.Environmentz(cf['field_dim_m'], 
                            cf['frame_dim_pix'])
    
    ledimgs_mean = led_params['ledimgs_mean']


    S = ledimgs_mean[0].shape[0]

    # in both cases taking the horizontal one
    sigma_front_h, a, a_x = led_measure(ledimgs_mean[0, S/2+1, :])
    sigma_back_h, a, a_x = led_measure(ledimgs_mean[1, S/2+1, :])

    sigma_front_v, a, a_x = led_measure(ledimgs_mean[0, :, S/2+1])
    sigma_back_v, a, a_x = led_measure(ledimgs_mean[1, :, S/2+1])

    w_front = (sigma_front_h + sigma_front_v)/2.
    w_back = (sigma_back_h + sigma_back_v)/2.

    dist_in_m = np.mean(led_params['dist'])
    dist_in_pix = int(dist_in_m * env.gc.pix_per_meter[0])
    return (dist_in_pix, np.ceil(w_front), np.ceil(w_back))

@follows(measure_diode_params)    
@transform(os.path.join(DATA_DIR, "*/led.params.pickle"), 
           regex(r".+/(.+)/led.params.pickle$"), 
           [os.path.join(REPORT_DIR, 
                         r"\1.led.params.png"), 
            os.path.join(REPORT_DIR, 
                         r"\1.led.examples.png"), 
            ]
            )
def plot_diode_params(led_params_pickle, (params_png, examples_png)):
    led_params = pickle.load(open(led_params_pickle, 'r'))

    ledimgs_mean = led_params['ledimgs_mean']
    subsampled_frames = led_params['subsampled_frames']
    S = ledimgs_mean[0].shape[0]
    pylab.figure()
    for i in range(2):
        ax = pylab.subplot2grid((3, 2), (i, 0))
        ax.imshow(ledimgs_mean[i], interpolation='nearest', 
                  cmap=pylab.cm.gray, vmin = 0, vmax=255)
        ax.axhline(S/2+1, c='b')
        ax.axvline(S/2+1, c='r')
        ax.grid(1)

        axl = pylab.subplot2grid((3, 2), (i, 1))
        axl.plot(ledimgs_mean[i, S/2+1, :], c='b')
        sigma, a, a_x = led_measure(ledimgs_mean[i, S/2+1, :])
        axl.plot(a_x, a, c='b', linestyle='--')
        axl.plot(ledimgs_mean[i, :, S/2+1], c='r')
        sigma, a, a_x = led_measure(ledimgs_mean[i, :, S/2+1])
        axl.plot(a_x, a, c='r', linestyle='--')
        axl.grid(1)

        # now the histogram
    axh = pylab.subplot2grid((3, 2), (2, 1), colspan=2)
    axh.hist(led_params['dist'], bins=20)
        
    pylab.savefig(params_png, dpi=300)

    pylab.figure()
    for i in range(2):
        ax = pylab.subplot2grid((4, 8), (i*2, 0), colspan=2, rowspan=2)
        ax.imshow(ledimgs_mean[i], interpolation='nearest', 
                  cmap=pylab.cm.gray, vmin = 0, vmax=255)
        ax.axhline(S/2+1, c='b')
        ax.axvline(S/2+1, c='r')
        ax.grid(1)
        ax.set_xticks([])
        ax.set_yticks([])

        # now the N examples
        fpos = 0
        for c in range(2, 8):
            for r in range(i*2, i*2 + 2):
                ax = pylab.subplot2grid((4, 8), (r, c))
                print subsampled_frames.shape
                ax.imshow(subsampled_frames[fpos, i, :, :], 
                          interpolation='nearest', 
                          cmap=pylab.cm.gray, vmin=0, vmax=255)
                ax.grid(1)
                ax.set_xticks([])
                ax.set_yticks([])
                fpos = min(fpos + 1, subsampled_frames.shape[0]-1)


    pylab.savefig(examples_png, dpi=300)



@transform(os.path.join(DATA_DIR, "*/positions.npy"), 
           regex(r".+/(.+)/positions.npy$"), 
           [os.path.join(REPORT_DIR, 
                         r"\1.motionstatistics.xy.png")]
            )
def motion_statistics(positions_file, (stats, )):
    """
    For each epoch, look at the distribution of the various 
    state parameters

    """
    positions = np.load(positions_file)
    basedir = os.path.dirname(positions_file)
    cf = pickle.load(open(os.path.join(basedir, "config.pickle")))

    invalid_sep = detect_invalid_sep(positions)
    positions_cleaned = positions.copy()
    positions_cleaned[invalid_sep] = ((np.nan, np.nan), 
                                      (np.nan, np.nan), np.nan, np.nan)
    
    positions_interp, missing = interpolate(positions)
    pci, cleaned_missing = interpolate(positions_cleaned)
    pos_derived = compute_derived(pci)

    vars = {'x' : pci['x'], 
            'y' : pci['y'], 
            'phi_unwrapped' : pos_derived['phi'], 
            'phi' : pos_derived['phi'] % (2*np.pi), 
            'theta' : np.zeros(10)} # fixme

    f = pylab.figure()
    N = 0
    M = 5000
    for vi, var in enumerate(['x', 'y','phi_unwrapped']):
        d = vars[var][N:M]
        vd = np.diff(d)
        SMOOTH_N = 8
        vd_smoothed = scipy.signal.filtfilt(np.ones(SMOOTH_N)/SMOOTH_N, 
                                            [1.], vd)
        ax = pylab.subplot2grid((6,1), (vi*2, 0))
        ax.plot(d, linewidth=0.3)
        ax.set_title(var)
        ax2 = pylab.subplot2grid((6,1), (vi*2+1, 0))
        ax2.plot(vd, c='r', linewidth=0.3)
        ax2.plot(vd_smoothed, c='g', linewidth=0.3)
        

        ax2.set_ylabel(var + " vel")
        
    f.suptitle(basedir)
    pylab.savefig(stats, dpi=600)

if __name__ == "__main__":
    pipeline_run([agg_stats, sanity_check, difficult_regions,
                  measure_diode_params, plot_diode_params, 
                  motion_statistics],
                  multiprocess=4)

    
    
    
    
    
