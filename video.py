import numpy as np
import os
import glob
import subprocess

class MPEGWrapper(object):
    """
    Simple class for reading in the mpegs
    """

    def __init__(self, filename):
        self.filename = filename
        self.cap = cv.CaptureFromFile(filename)

        self.frame_count = cv.GetCaptureProperty(self.cap, 
                                                 cv.CV_CAP_PROP_FRAME_COUNT)
        
        self.fps = cv.GetCaptureProperty(self.cap, 
                                         cv.CV_CAP_PROP_FPS)
        self.width = cv.GetCaptureProperty(self.cap, cv.CV_CAP_PROP_FRAME_WIDTH)
        self.height = cv.GetCaptureProperty(self.cap, cv.CV_CAP_PROP_FRAME_HEIGHT)
        
        self.internal_pos = 0
    def get_next_frame(self):
        i = cv.QueryFrame(self.cap)
        ai =  np.asarray(cv.GetMat(i))
        self.internal_pos += 1

        return ai

    def seek(self, id):
        while self.internal_pos < id:
            if self.internal_pos % 1000 == 0:
                print self.internal_pos, id
            self.get_next_frame()

        #cv.SetCaptureProperty(self.cap, cv.CV_CAP_PROP_POS_FRAMES, id)
    def get_pos(self):
        return cv.GetCaptureProperty(self.cap, cv.CV_CAP_PROP_POS_FRAMES)

def array2cv(a):
    """
    From Wiki
    http://opencv.willowgarage.com/wiki/PythonInterface
    """
    dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
        }
    try:
        nChannels = a.shape[2]
    except:
        nChannels = 1
    cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
                                 dtype2depth[str(a.dtype)],
                                 nChannels)
    cv.SetData(cv_im, a.tostring(),
               a.dtype.itemsize*nChannels*a.shape[1])
    return cv_im


def write_video(fname, array):
    """
    Write the array that is 8-bit pixel grey data [frame, rows, cols]
    to file name fname
    
    """
    N, H, W = array.shape
    format = cv.FOURCC('I', 'Y', 'U', 'V')
    vw = cv.CreateVideoWriter(fname, format, 
                              30.0, (W, H))
    for f in array:
        bitmap = array2cv(array)
        
        cv.WriteFrame(vw, bitmap)

def frames_to_mpng(frame_glob, outfile, fps=5):
    cmd = "mencoder mf://%s -mf fps=%d:type=png -ovc copy -oac copy -o %s" % (frame_glob, fps, outfile)
    subprocess.call(cmd, shell=True)
