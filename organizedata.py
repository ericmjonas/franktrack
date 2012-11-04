import numpy as np
import os
import glob
from ruffus import * 
import subprocess
import frankdata as fl
import tarfile
import cPickle as pickle
import tempfile

import scipy.ndimage

"""
animal_day_epoch_etc/:

positions.npy : front/back diode positions for each frame, npy

frames : a collection of 1000 frame chunks with the start frame #
"""

DATA_DIR = "data/fl"

VIDEOTEMP = "video.temp" # place for rendered frames

FRAMES_PER_TAR = 1000
FIGS_DIR = "figs"


# this code goes through and generates the files necessary for processing the data

def generate_files_fl():
    """
    Generate gives us (video, raw position, epochs_pfiles)
    """
    # walter's data

    BASEDIR = "original.data/Bukowski"
    for i in range(2, 5):
        name = "bukowski_%02d" % i
        # get the base from the mpeg
        p = glob.glob(os.path.join(BASEDIR, name, "*.mpeg"))
        basepath = p[0][:-5]

        mpeg = basepath + ".mpeg"
        postimestamp = basepath + ".postimestamp"
        epochs = glob.glob(basepath + "_*.p")
        awake_epochs = [e for e in epochs if 'sleep' not in e]

        params = {'field_dim_m' : (1.5, 2.0),  
                  'frame_dim_pix': (240, 320)}

        yield (mpeg, postimestamp, awake_epochs, name, params)

def generate_files_fl_proc():
    for mpeg, postimestamp, awake_epochs, name, params in generate_files_fl():
        infiles = [mpeg, postimestamp] +  awake_epochs
        outfiles = os.path.join(VIDEOTEMP, name)
        
        yield infiles, outfiles, name

def package_fl_proc():
    for mpeg, postimestamp, awake_epochs, name, params in generate_files_fl():
        for epoch in awake_epochs:
            infiles = [postimestamp, os.path.join(VIDEOTEMP, name), 
                       epoch]
            epoch_name = epoch[len(mpeg)-4:-2]
            outdir = os.path.join(DATA_DIR, name + "." + epoch_name)
            outfiles = [os.path.join(outdir, 'positions.npy'), 
                        outdir, 
                        os.path.join(outdir, 'config.pickle')]
            yield infiles, outfiles, name, epoch_name, params
                                  
                        
def get_frames(directory, frames):
    """
    Convenience function that reads in the config, 
    figures out the frame pos, and returns the N frames in 
    a numpy array
    frames is an array of absolute frame positions
    
    """

    cf = pickle.load(open(os.path.join(directory, "config.pickle")))
    start_f = cf['start_f']
    end_f = cf['end_f']
    FRAMEN = end_f - start_f + 1

    if (frames > (FRAMEN-1)).any():
        raise Exception("Requesting frame not in epoch, there are only %d frames here" % FRAMEN)
    
    sorted_idx = np.argsort(frames)
    frames_sorted = frames[sorted_idx]
    
    out_data = np.zeros((len(frames), cf['frame_dim_pix'][0], 
                         cf['frame_dim_pix'][1]), dtype=np.uint8)
    

    frame_tars = glob.glob(os.path.join(directory, "*.tar.gz"))
    frame_tars_filenames = [os.path.basename(f) for f in frame_tars]

    frame_tar_starts = np.sort([int(f[:-len(".tar.gz")]) for f in frame_tars_filenames])
    assert(frame_tar_starts[0] == start_f)

    frame_archive_src = np.searchsorted(frame_tar_starts, 
                                         frames_sorted + start_f, 
                                        side='right')-1
    # add in some caching
    tf = None
    tf_cached_name = None

    for fi, f in enumerate(frames_sorted):
        frame_archive_idx = frame_archive_src[fi]
        frame_archive_frame_no = frame_tar_starts[frame_archive_idx]

        # open the frame tarball
        tf_name = os.path.join(directory, 
                               "%08d.tar.gz" % frame_archive_frame_no)
        if tf_cached_name != tf_name:
            tf = tarfile.open(tf_name, "r:gz")
            tf_cached_name = tf_name

        frame_no = frames_sorted[fi] + start_f

        frame = tf.extractfile("%08d.jpg" % frame_no)
        # temporary write to disk
        tempf = tempfile.NamedTemporaryFile()
        tempf.write(frame.read())
        tempf.flush()
        f = scipy.ndimage.imread(tempf.name, flatten=True)
        out_data[sorted_idx[fi]] = f
        tempf.close()
    return out_data


    
    
@files(generate_files_fl_proc)
def mpeg_to_stills(infiles, outfiles, name):
    """
    Convert mpeg to the stand-alone series of jpegs
    before converting
    
    """
    mpeg_file = infiles[0]
    postimestamp = infiles[1]
    awake_epochs = infiles[2:]

    os.mkdir(outfiles)

    # now subprocess.call
    subprocess.call(["mplayer", "-vo", "jpeg:outdir=%s" % outfiles, 
                    mpeg_file])
    print outfiles
#def render_output([mpeg, postimestamp, awake_epochs], 

def package_frames(framedir, start_f, end_f, outdir):
    """
    Take a directory of frames, and package the frames
    in 1000-jpeg tar.gzs named with the first frame in the series
    start_f, end_f is inclusive

    """
    framepos = 0
    FRAME_N = end_f - start_f + 1

    archive_filenames = []
    
    fid = None

    while framepos < FRAME_N:
        frame_no = framepos + start_f
        if (framepos % FRAMES_PER_TAR) == 0:
            if fid != None:
                fid.close()
            # create a new tar
            tarfilename = os.path.join(outdir, "%08d.tar.gz" % frame_no)
            archive_filenames.append(tarfilename)
            
            fid = tarfile.open(tarfilename, 'w:gz')
        tgt_f = os.path.join(framedir, "%08d.jpg" % frame_no)
        fid.add(tgt_f, arcname= "%08d.jpg" % frame_no)
        framepos += 1
    fid.close()
    return archive_filenames
            
def pfile_to_pos_file(pfile_array, cam_dim, area_dim):
    """
    Convert timestamp - front - back to our custom per-frame masked-array
    format
    
    """
    N = len(pfile_array)
    d = np.ma.zeros(N, dtype=[('led_front', np.float32, 2), 
                              ('led_back', np.float32, 2), 
                              ('x', np.float32),
                              ('y', np.float32)])

    CAM_MAX_Y, CAM_MAX_X = cam_dim
    AREA_MAX_Y, AREA_MAX_X = area_dim

    # for each point:
    for ri, row in enumerate(pfile_array):
        fl_x, fl_y = row['front']
        bl_x, bl_y = row['back']
        if fl_x == 0 or bl_x == 0 or fl_y == 0 or bl_y == 0:
            d[ri] = np.ma.masked
        else:
            fl_pos = (float(fl_x)/CAM_MAX_X * AREA_MAX_X, 
                      float(fl_y)/CAM_MAX_Y * AREA_MAX_Y) 
            bl_pos = (float(bl_x)/CAM_MAX_X * AREA_MAX_X, 
                      float(bl_y)/CAM_MAX_Y * AREA_MAX_Y) 
            d['led_front'][ri] = fl_pos
            d['led_back'][ri] = bl_pos
            d['x'][ri] = np.mean([fl_pos[0], bl_pos[0]])
            d['y'][ri] = np.mean([fl_pos[1], bl_pos[1]])


    return d
            
    
@follows(mpeg_to_stills, mkdir(DATA_DIR))
@files(package_fl_proc)
def package_fl_frames((postimestamp, videodir, epoch), 
                      (positions_file, frames_tar_dir, config_pickle), 
                      name, epoch_name, params):
    """
    Take in the pos time stamp and the epoch
    figure out the frames for the epoch, tar them, and then save some
    meta data in the config

    """
    print name, videodir, frames_tar_dir, epoch_name
    epoch_p = fl.read_pfile(epoch)
    start_ts = epoch_p['timestamp'][0]
    end_ts = epoch_p['timestamp'][-1]

    pts = fl.read_postimestamp(postimestamp)
    
    # find the frame that starts this epoch
    start_f = np.searchsorted(pts['timestamp'], start_ts)
    end_f = np.searchsorted(pts['timestamp'], end_ts)
    
    FRAMENUM = end_f - start_f

    os.mkdir(frames_tar_dir)    
    
    # note that the franklab p-files are slightly different from our own
    # inference code
    pos_data_ma =  pfile_to_pos_file(epoch_p, params['frame_dim_pix'], 
                                     params['field_dim_m'])
    pos_data = pos_data_ma.filled(np.nan)

    np.save(positions_file, pos_data)
    package_frames(videodir, start_f, end_f, 
                   frames_tar_dir)

    config = {'start_ts' : start_ts, 
              'end_ts' : end_ts, 
              'epoch_name' : name, 
              'postimestamp_file' : postimestamp, 
              'start_f' : start_f, 
              'end_f' : end_f}

    config.update(params)
    pickle.dump(config, file(config_pickle, 'w'))
    

if __name__ == "__main__":    
    pipeline_run([package_fl_frames], multiprocess=3)
    
