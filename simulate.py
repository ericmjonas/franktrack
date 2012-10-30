import numpy as np
import likelihood
import cPickle as pickle
import util
import plotting
from matplotlib import pyplot
import videotools

"""
The simulation process:
a. generate true state vector 
b. render it to the images
c. add noise, distractions, etc. to the images. 
"""


def gen_track_circle(t, average_angular_velocity, env, 
                     xy_noise=0.001, phi_noise = 0.02, 
                     theta_noise=0.1, circle_radius = 1.0):
              
    """
    This has the animal running around in roughly a circle with some 
    head-direction wiggle, generally looking straight-ahead, and
    occasionally bobbing. 

    The animal starts at outer-circle-theta=0 moving ccw with head level
        
    t = temporal moments
    """
    N = len(t)
    state = np.zeros(N, dtype=util.DTYPE_STATE)

    R = circle_radius
    
    theta = t * average_angular_velocity

    x_noise = np.random.normal(0, xy_noise, N)
    y_noise = np.random.normal(0, xy_noise, N)

    phi_noise = np.random.normal(0, phi_noise, N)
    theta_noise = np.random.normal(0, theta_noise, N)

    room_center_xy = env.get_room_center_real_xy()

    state['x'] = R * np.cos(theta) + x_noise + room_center_xy[0]
    state['y'] = R * np.sin(theta) + y_noise + room_center_xy[1]
    state['phi'] = np.pi/2 + theta + phi_noise
    state['theta'] = np.pi/2 + theta_noise 
    
    return state

def render(env, state, DIODE_SEP = 10,
           FRONT_PIX = 4, BACK_PIX =2):

    gc = env.gc


    eo = likelihood.EvaluateObj(*env.image_dim)
    eo.set_params(DIODE_SEP, FRONT_PIX, BACK_PIX)

    images = np.zeros((len(state), env.image_dim[0], 
                       env.image_dim[1]), 
                      dtype=np.uint8)
    for si, s in enumerate(state):
        i_x, i_y = gc.real_to_image(s['x'], s['y'])
        img = eo.render_source(i_x, i_y, s['phi'], s['theta'])
        x = img*255
        x.flat[x.flat > 255] = 255
        images[si] = x

    return images


def add_noise_background(video, dc_noise_level, ac_noise_level):
    N, VIDEO_ROWS, VIDEO_COLS = video.shape
    dc_noise = (np.random.rand(VIDEO_ROWS, VIDEO_COLS) * dc_noise_level)

    new_video = np.zeros_like(video)

    for f in range(len(video)):
        ac_noise = np.random.rand(VIDEO_ROWS, VIDEO_COLS)*ac_noise_level
        new_f = video[f].astype(np.float32) + dc_noise + ac_noise
        new_f = np.minimum(new_f, 255)
        new_video[f] = new_f
    return new_video
    

def simple_gen():
    SIM_DURATION = 30.0
    TDELTA = 1/30.
    t = np.arange(0, SIM_DURATION, TDELTA)

    for NOISE in [0, 50, 100, 200, 255]:

        env = util.Environment((1.5, 2), (240, 320))

        state = gen_track_circle(t, np.pi*2/10, env, circle_radius=0.5)
        images = render(env, state)
        new_images = add_noise_background(images, NOISE, NOISE)
        pickle.dump({'state' : state, 
                     'video' : new_images}, 
                    open('simulate.%03d.pickle' % NOISE, 'w'))
        videotools.dump_grey_movie('test.%03d.avi' % NOISE, new_images)    
simple_gen()
