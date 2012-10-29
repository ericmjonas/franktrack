import numpy as np
from matplotlib import pylab
import time
import util

def plot_state(ax, state, full_render_skip, 
               length):
    """
    take an axis and plot the state vector on it

    """
    
    ax.plot(state['x'], state['y'])

    front_pos, back_pos = util.compute_pos(length, state['x'], state['y'], 
                                           state['phi'], state['theta'])
    N = len(state)
    for p in range(0, N, full_render_skip):
        fp_x = front_pos[0][p]
        fp_y = front_pos[1][p]
        bp_x = back_pos[0][p]
        bp_y = back_pos[1][p]
        print bp_x, bp_y, fp_x, fp_y

        ax.plot([bp_x, fp_x], 
                [bp_y, fp_y], c='k')
        ax.scatter(bp_x, bp_y, c='r')
        ax.scatter(fp_x, fp_y, c='b')

        
def animate_state(state, length):
    pylab.ion()

    tstart = time.time()               # for profiling

    line, = pylab.plot([0, 1], 
                       [0, 1], c='k', linewidth=2)
    N = len(state)
    front_pos, back_pos = compute_pos(length, state['x'], state['y'], 
                                      state['phi'], state['theta'])

    for p in np.arange(N):
        fp_x = front_pos[0][p]
        fp_y = front_pos[1][p]
        bp_x = back_pos[0][p]
        bp_y = back_pos[1][p]
        
        line.set_xdata([bp_x, fp_x])
        line.set_ydata([bp_y, fp_y])  # update the data
        pylab.draw()                         # redraw the canvas

    print 'FPS:' , 200/(time.time()-tstart)

def plot_state_timeseries(state, length):
    
    ax_main = pylab.subplot2grid((8,8), (0, 0), colspan=4, rowspan=4)
    ax_main.grid(1)
    plot_state(ax_main, state, 10, length)

    for si, s in enumerate(['x', 'y', 'phi', 'theta']):
        ax_i = pylab.subplot2grid((8, 8), (4 + si, 0), colspan=8)
        ax_i.plot(state[s])
        ax_i.grid(1)
