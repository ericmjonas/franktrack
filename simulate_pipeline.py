from ruffus import *
import tarfile
import os
import numpy as np
import scipy.ndimage
import simulate
import util2 as util
import cPickle as pickle
import videotools
import measure
from matplotlib import pylab


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
    

@follows(mkdir(os.path.join(SYNTH_DIR, "fl")))
@transform(os.path.join(DATA_DIR, "fl/*/positions.npy"), 
           regex(r".+/(.+)/positions.npy$"), 
           [os.path.join(SYNTH_DIR, "fl",  
                        r"\1.pickle"), 
           os.path.join(SYNTH_DIR, "fl",  
                        r"\1.avi")]
            )
def fl_to_sim(positions_file, (out_pickle, out_avi)):
    print "THIS IS", positions_file
    positions = np.load(positions_file)
    # frames thing
    directory = positions_file[:-len('positions.npy')]
    cf = pickle.load(open(os.path.join(directory, "config.pickle")))
    start_f = cf['start_f']
    # open the frame tarball
    tf = tarfile.open(os.path.join(directory, "%08d.tar.gz" % start_f),
                                   "r:gz")
    
    positions_interp, missing = measure.interpolate(positions)
    pos_derived = measure.compute_derived(positions_interp)
    

    N = len(positions)
    state = np.zeros(N, dtype=util.DTYPE_STATE)
    state['x'] = positions_interp['x']
    state['y'] = positions_interp['y']
    state['phi'] = pos_derived['phi']
    state['theta'] = np.pi/2.0

    env = util.Environmentz((1.5, 2), (240, 320))

    images = simulate.render(env, state[:100])
    NOISE = 0


    new_images = simulate.add_noise_background(images, NOISE, NOISE, 
                                               [])

    FN = 100
    pylab.figure()
    pylab.subplot(2, 1, 1)
    pylab.plot(state['x'][:FN])
    pylab.plot(state['y'][:FN])
    pylab.subplot(2, 1, 2)
    pylab.scatter(positions['led_front'][:FN, 0], positions['led_front'][:FN, 1], c='g')
    pylab.scatter(positions['led_back'][:FN, 0], positions['led_back'][:FN, 1], c='r')

    pylab.show()
    for fi in range(FN):

        frame_no = start_f + fi
        frame = tf.extractfile("%08d.jpg" % frame_no)
        open('/tmp/test.jpg', 'w').write(frame.read())
        f = scipy.ndimage.imread("/tmp/test.jpg")

        img_x, img_y = env.gc.real_to_image(state['x'][fi], 
                                            state['y'][fi])
        front_x, front_y = env.gc.real_to_image(*positions['led_front'][fi])
        back_x, back_y = env.gc.real_to_image(*positions['led_back'][fi])

        print "PHI =", state['phi'][fi]
        print "positions_interp", (positions['led_front'][fi], 
                                   positions['led_back'][fi])

        pylab.figure()
        pylab.subplot(1, 2, 1)
        pylab.imshow(f, interpolation='nearest')
        pylab.axvline(img_x, c='k')
        pylab.axhline(img_y, c='k')
        pylab.axvline(front_x, c='g')
        pylab.axhline(front_y, c='g')
        pylab.axvline(back_x, c='r')
        pylab.axhline(back_y, c='r')
        pylab.subplot(1, 2, 2)
        pylab.imshow(new_images[fi])
        pylab.show()


    
    # pickle.dump({'state' : state, 
    #              'video' : new_images, 
    #              'noise' : NOISE, 
    #              'frames_skipped' : [], 
    #              }, 
    #             open(out_pickle, 'w'))
    # videotools.dump_grey_movie(out_avi, new_images)    
    
    
pipeline_run([synth_circle_noise, fl_to_sim])

