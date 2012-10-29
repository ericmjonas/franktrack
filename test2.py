import scipy
import numpy as np
from matplotlib import pyplot
x = np.load('video.npy')

# compute the N-frame rolling average
f_avg = np.mean(x, axis=0)

for fi, f in enumerate(x[:3600]):
    y = f - f_avg
    
    pyplot.imsave('frame%08d.png' % fi, y, 
                  cmap=pyplot.cm.gray, 
                  vmin = 0)
    

    
    
