import numpy as np
from nose.tools import *


import videotools

def test_writer():
    frames = np.random.random_integers(0,255,size=(50,100,200,3)).astype('uint8')
    vsnk = videotools.VideoSink('mymovie.avi',size=frames.shape[1:3],
                                colorspace='rgb24')
                                
    for frame in frames:
        vsnk(frame)
    vsnk.close()
                              
    a = np.zeros((1000, 240, 320), 
                 dtype = np.uint8)
    for i in range(1000):
        a[i, i % 240, i % 320]= 255
    videotools.dump_grey_movie('test.avi', a)

