from ruffus import *
import simulate
import os
import numpy as np
import util2 as util
import cPickle as pickle
import videotools

DATA_DIR = "data"
SYNTH_NAME = "synth"
SYNTH_DIR = os.path.join(DATA_DIR, SYNTH_NAME)

def synth_circle_noise_gen():
    for NOISE in [0, 100, 255]:
        base = os.path.join(SYNTH_DIR, "circle.%03d" % NOISE)
        yield [], [base + '.pickle', base +".avi"], NOISE

@follows(mkdir(SYNTH_DIR))
@files(synth_circle_noise_gen)
def synth_circle_noise(infiles, outfiles, NOISE):
    
    SIM_DURATION = 30.0
    TDELTA = 1/30.
    
    t = np.arange(0, SIM_DURATION, TDELTA)
    
    frames_to_skip = [20, 35, 36, 50, 51, 52, 53]
    
    env = util.Environmentz((1.5, 2), (240, 320))
    
    state = simulate.gen_track_circle(t, np.pi*2/10, env, circle_radius=0.5)
    images = simulate.render(env, state)
    new_images = simulate.add_noise_background(images, NOISE, NOISE, 
                                               frames_to_skip)
    pickle.dump({'state' : state, 
                 'video' : new_images, 
                 'noise' : NOISE, 
                 'frames_skipped' : frames_to_skip
                 }, 
                open(outfiles[0], 'w'))
    videotools.dump_grey_movie(outfiles[1], new_images)    
    
pipeline_run([synth_circle_noise], multiprocess=3)

