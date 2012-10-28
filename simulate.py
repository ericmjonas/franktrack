import numpy as np
import likelihood
import util
import plotting
from matplotlib import pyplot


def gen_track(T, tdelta, xy_noise, phi_noise):
              
    """
    This has the animal running around in roughly a circle with some 
    head-direction wiggle, generally looking straight-ahead, and
    occasionally bobbing. 

    The animal starts at outer-circle-theta=0 moving ccw with head level
        
    """
    tsteps = int(T/tdelta)
    
    state = np.zeros(tsteps, dtype=util.DTYPE_STATE)


    R = 1.0 # m
    
    T_circle = 30.0 # sec
    w = 2*np.pi / T_circle 
    
    t = np.arange(0, tsteps)*tdelta
    
    theta = t * w
    x_noise = np.random.normal(0, xy_noise, tsteps)
    y_noise = np.random.normal(0, xy_noise, tsteps)

    phi_noise = np.random.normal(0, phi_noise, tsteps)
    theta_noise = np.random.normal(0, 0.1, tsteps)

    state['x'] = R * np.cos(theta) + x_noise
    state['y'] = R * np.sin(theta) + y_noise
    state['phi'] = np.pi/2 + theta + phi_noise
    state['theta'] = np.pi/2 + theta_noise 
    
    return state


DIODE_SEP = 0.03 # m
state = gen_track(30, 0.1, 0.01, 0.5)
f = pyplot.figure()
ax = f.add_subplot(1, 1, 1)

plotting.plot_state_timeseries(state, DIODE_SEP)
pyplot.show()

#plotting.plot_state(ax, state, 10, DIODE_SEP)
#pyplot.show()
#plotting.animate_state(state, DIODE_SEP)
