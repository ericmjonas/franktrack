import cv
import numpy as np
from matplotlib import pyplot
import os
import glob

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


class MPEGWrapper(object):
    def __init__(self, filename):
        self.filename = filename
        self.cap = cv.CaptureFromFile(filename)

        self.frame_count = cv.GetCaptureProperty(self.cap, 
                                                 cv.CV_CAP_PROP_FRAME_COUNT)
        
        self.fps = cv.GetCaptureProperty(self.cap, 
                                         cv.CV_CAP_PROP_FPS)
        self.width = cv.GetCaptureProperty(self.cap, cv.CV_CAP_PROP_FRAME_WIDTH)
        self.height = cv.GetCaptureProperty(self.cap, cv.CV_CAP_PROP_FRAME_HEIGHT)
        
    def get_next_frame(self):
        i = cv.QueryFrame(self.cap)
        ai =  np.asarray(cv.GetMat(i))
        return ai

    def seek(self, id):
        cv.SetCaptureProperty(self.cap, cv.CV_CAP_PROP_POS_FRAMES, id)
    def get_pos(self):
        return cv.GetCaptureProperty(self.cap, cv.CV_CAP_PROP_POS_FRAMES)

        
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

mpeg_file = MPEGWrapper(b(basename + ".mpeg"))

print "start time = ", start_t
print "end time = ", end_t
print "start_f = ", start_f
print "end_f = ", end_f

mpeg_file.seek(start_f - 10)
print "Seek to", start_f, "now at",  mpeg_file.get_pos()

for i in range(100):
    pos = epochs[tgt_epoch]['position'][i]
    print pos['timestamp'], pts['timestamp'][start_f + i]


    f = mpeg_file.get_next_frame()
    pyplot.figure()
    pyplot.imshow(f)
    pyplot.scatter(pos[1][0], pos[1][1])
    pyplot.scatter(pos[2][0], pos[2][1], c='r')
    pyplot.savefig('test%05d.png' % i)

# I'm pretty sure postimestamp goes from frame# in file ->timestamp


#mpeg_read()
#mpeg_os = read_mpegoffset(header_read(b(basename + ".mpegoffset")))
#print len(mpeg_os), np.min(mpeg_os['offset']), np.max(mpeg_os['offset'])

