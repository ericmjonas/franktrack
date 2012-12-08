import numpy as np
import cPickle as pickle
import os
import util2 as util
import measure
import methods
import organizedata
from ruffus import * 
import track

import cloud
from glob import glob


DATA_DIR = "data/fl"
def ddir(x):
    return os.path.join(DATA_DIR, x)

REPORT_DIR = "results"
def rdir(x):
    return os.path.join(REPORT_DIR, x)


dirs = glob(DATA_DIR + "/*")
datasets = [x[len(DATA_DIR)+1:] for x in dirs]

def per_frame_wrapper(dname):
    algo_data = track.per_frame(ddir(dname), methods.centroid_frame, 
                                {'thold' : 240})
    
    
jids = cloud.map(per_frame_wrapper, datasets, _type='f2', 
                 _vol="my-vol", _env='base/precise')

cloud.result(jids)
