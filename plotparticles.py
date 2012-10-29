import numpy as np
import cPickle as pickle
from matplotlib import pylab


a = np.load('test.npz')

weights = a['weights']
particles = a['particles']
print particles
x = []
y = []
for p in particles:
    x.append([s['x'] for s in p])
    y.append([s['y'] for s in p])
x = np.array(x)
y = np.array(y)

print x.shape
xbar = np.mean(x, axis=1)
ybar = np.mean(y, axis=1)
wvar = np.var(weights, axis=1)

print xbar.shape
pylab.subplot(3,1, 1)
pylab.plot(xbar)
pylab.subplot(3,1, 2)
pylab.plot(ybar)
pylab.subplot(3,1, 3)
pylab.plot(wvar)
pylab.show()


