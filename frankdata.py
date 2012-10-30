import cv
import numpy as np
from matplotlib import pylab
import os
import glob
import video
import videotools

def header_read(f):
    """
    Returns a file object where the header has been removed
    """
    fid = open(f, 'r')
    while fid.readline() != "%%ENDHEADER\n":
        pass
    return fid


BASEDIR = "/Users/jonas/projects/ssm/franklab/eric/Bukowski/bukowski_02/"

def b(f):
    return os.path.join(BASEDIR, f)

def read_possynctimes(f):
    """
    Return numpy array with possynctimes
    """

    d = np.fromfile(f, dtype=[("timestamp", np.uint32), ("transition", np.uint8)])
    
    return d
def read_pfile(f):
    """
    Read a position file
    """
    d = np.fromfile(f, dtype=[("timestamp", np.uint32), 
                              ("front", np.uint16, 2), 
                              ("back", np.uint16, 2)])

    return d

def read_postimestamp(f):
    d = np.fromfile(f, dtype=[('timestamp', np.uint32)])
    return d

def read_mpegoffset(f):
    d = np.fromfile(f, dtype=[('offset', np.uint64)])

    return d

def video_dump():
    r = os.path.join(BASEDIR, "possynctimes")
    pst = read_possynctimes(header_read(r))

    basename_dir = glob.glob(b("*.mpeg"))[0]
    basename = os.path.splitext(os.path.basename(basename_dir))[0]

    epochs = {}

    for fname in glob.glob(b("%s_*.p" % basename)):
        bn = os.path.basename(fname)
        epochname = bn[len(basename)+1:-2]
        print epochname

        linear_p = read_pfile(header_read(fname))
        ed = {'start' : linear_p['timestamp'][0],
              'end' : linear_p['timestamp'][-1]}
        ed['position'] = linear_p

        epochs[epochname] = ed

    tgt_epoch = 'linear'

    start_t = epochs[tgt_epoch]['start']
    end_t = epochs[tgt_epoch]['end']

    pts = read_postimestamp(header_read(b(basename + '.postimestamp')))

    # find the frame that starts this epoch
    start_f = np.searchsorted(pts['timestamp'], start_t)
    end_f = np.searchsorted(pts['timestamp'], end_t)

    mpeg_file = video.MPEGWrapper(b(basename + ".mpeg"))

    print "start time = ", start_t
    print "end time = ", end_t
    print "start_f = ", start_f
    print "end_f = ", end_f

    mpeg_file.seek(start_f)
    print "Seek to", start_f, "now at",  mpeg_file.get_pos()
    data = np.zeros((end_f - start_f, 240, 320), dtype=np.uint8)

    for i in range(start_f, end_f):
        print "frame", i, end_f
        #pos = epochs[tgt_epoch]['position'][i]
        #print pos['timestamp'], pts['timestamp'][start_f + i]

        f = mpeg_file.get_next_frame()
        # pyplot.figure()
        # pyplot.imshow(f)
        # pyplot.scatter(pos[1][0], pos[1][1])
        # pyplot.scatter(pos[2][0], pos[2][1], c='r')
        # pyplot.savefig('test%05d.png' % i)
        print f.shape

        data[i - start_f] = f[:, :, 0]
    np.save("video.npy", data)

    # I'm pretty sure postimestamp goes from frame# in file ->timestamp


    #mpeg_read()
    #mpeg_os = read_mpegoffset(header_read(b(basename + ".mpegoffset")))
    #print len(mpeg_os), np.min(mpeg_os['offset']), np.max(mpeg_os['offset'])

def alignment_test():
    basename_dir = glob.glob(b("*.mpeg"))[0]
    basename = os.path.splitext(os.path.basename(basename_dir))[0]
    
    pts = read_postimestamp(header_read(b(basename + '.postimestamp')))

    linear_p = read_pfile(header_read(b(basename + "_linear.p")))

    mpeg_file = video.MPEGWrapper(b(basename + ".mpeg"))
    print "FRAME COUNT=", mpeg_file.frame_count

    for F in  [11, 110, 200, 301, 400, 501, 601, 700, 800, 901]:
        start_t = linear_p[F]['timestamp']
        start_f = np.searchsorted(pts['timestamp'], start_t)



        # print pts.shape
        mpeg_file.seek(start_f)
        frame = mpeg_file.get_next_frame()
        pylab.figure()
        pylab.imshow(frame, interpolation='nearest', origin='lower')


        for f_os in range(F+1, F+15):
            pos = linear_p[f_os]
            pylab.scatter(pos['front'][0], pos['front'][1], c='r', alpha=0.5, linewidth=0)
            pylab.scatter(pos['back'][0], pos['back'][1], c='g', alpha=0.5, 
                          linewidth=0)
        pos = linear_p[F]
        pylab.scatter(pos['front'][0], pos['front'][1], c='r')
        pylab.scatter(pos['back'][0], pos['back'][1], c='g')

        pylab.title('ts=%d, frame=%d' % (start_t, start_f))
        pylab.savefig('test%05d.png' % F, dpi=300)
        # print mpeg_file.frame_count

alignment_test()
