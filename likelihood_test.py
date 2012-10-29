import numpy as np
import likelihood
import util
import plotting
from matplotlib import pylab
import videotools

def preprocess():
    from matplotlib import pylab
    x = np.load('video.npy')[:2000]

    print x.dtype
    avg = np.mean(x, axis=0, dtype=np.float32)
    print avg.dtype

    x_normed = x - avg
    x_normed.flat[x_normed.flat < 0] = 0

    
    # now save a few frames
    x_reg = x[::100]
    x_normed = x_normed[::100].astype(np.uint8)

    #pylab.imshow(x_normed[10], interpolation='nearest', 
    #             cmap=pylab.cm.gray)
    
    #pylab.show
    np.savez_compressed('frames.npz',  x_reg=x_reg, x_normed = x_normed)

# now 
#plotting.plot_state_timeseries(state, DIODE_SEP)
#pylab.show()
IMG_WIDTH = 320
IMG_HEIGHT = 240
PPM = 50
gc = util.GeomConverter((IMG_WIDTH, IMG_HEIGHT), 
                        (PPM, PPM), (0.0, 0.0))

eo = likelihood.EvaluateObj(IMG_WIDTH, IMG_HEIGHT)
eo.set_params(14, 4, 2)

frames = np.load('frames.npz')

phi_range = np.linspace(0.0, 2*np.pi, 8)
x_range = np.linspace(0.0, IMG_WIDTH/PPM*0.98, IMG_WIDTH/4)
y_range = np.linspace(0.0, IMG_HEIGHT/PPM*0.98, IMG_HEIGHT/4)

for frame_no, frame in enumerate(frames['x_normed']):
    print frame_no
    deltas = np.zeros((len(phi_range), len(x_range), len(y_range)), dtype=np.float)
    for phi_i, phi in enumerate(phi_range):
        for x_i, x in enumerate(x_range):
            for y_i, y in enumerate(y_range): 
                i_x, i_y = gc.real_to_image(x, y)

                img = eo.render_source(i_x, i_y, phi, np.pi/2)
                img_pix = img*255
                img_pix.flat[img_pix.flat > 255] = 255
                delta = (img_pix.astype(float) - frame.astype(float))
                deltas[phi_i, x_i, y_i] = np.sum((delta **2))
                # now diff with the frame

    idx = np.unravel_index(np.argmin(deltas), deltas.shape)
    print idx
    x_min=  x_range[idx[1]]
    y_min = y_range[idx[2]]
    pix = gc.real_to_image(x_min, y_min)
    print "min_meters = ", (x_min, y_min)
    print "min_pix = ", gc.real_to_image(x_min, y_min)
    print "min phi=", phi_range[idx[0]]

    pylab.figure()
    pylab.imshow(frame,
                 interpolation='nearest', cmap=pylab.cm.gray, 
                 origin='lower')
    pylab.axhline(pix[1])
    pylab.axvline(pix[0])

    front, back = util.compute_pos(14, pix[0], pix[1], phi_range[idx[0]], 
                                   np.pi/2)
    pylab.plot([front[0], back[0]], 
               [front[1], back[1]], c='r')
 
    pylab.savefig('tracked%06d.png' % frame_no)


#pylab.imshow(deltas, interpolation='nearest')            
#pylab.show()
# pylab.figure()
# pylab.imshow(x, interpolation='nearest', 
#               cmap = pylab.cm.gray)
# pylab.figure()
# pylab.imshow(frames['x_normed'][0], 
#              interpolation='nearest', cmap=pylab.cm.gray)

# pylab.show()

#videotools.dump_grey_movie('test.avi', images)
