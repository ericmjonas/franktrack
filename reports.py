import pandas
import cPickle as pickle
import numpy as np
from matplotlib import pylab
df = pickle.load(open("particles.summary.pickle", 'r'))
x = df.pfmean - df.truth

df['err'] = x.apply(lambda a: np.sum(np.abs(a)))/df['particlen']

xe = df[df.variable == 'x'].groupby(['epoch', 'frame_start'])['err'].mean()
ye = df[df.variable == 'y'].groupby(['epoch', 'frame_start'])['err'].mean()
te = xe + ye
tec = te.copy()
tec.sort()


ax =pylab.subplot(1, 1, 1)

a = ax.hist(np.log(tec), bins=100)

bins = [0.0, 0.20, 0.40, 1000.0]
for b in bins:
    ax.axvline(np.log(b), c='r')
vals, bins_out = np.histogram(tec, bins)

for thold, v, name in zip(bins[:-1], vals, ['good', 'close', 'fail']):
    print "thold =", thold, "name=", name, "%02d" % (float(v) *100 / len(tec)), "%"

pylab.show()
