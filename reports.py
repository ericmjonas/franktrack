import pandas
import cPickle as pickle
import numpy as np
from matplotlib import pylab
df = pickle.load(open("particles.summary.pickle", 'r'))
x = df.pfmean - df.truth

df['err'] = x.apply(lambda a: np.sum(np.abs(a)))/(df['frame_end']-df['frame_start'] + 1)

xe = df[df.variable == 'x'].groupby(['epoch', 'frame_start'])['err'].mean()
ye = df[df.variable == 'y'].groupby(['epoch', 'frame_start'])['err'].mean()
te = xe + ye
tec = te.copy()
tec.sort()


ax =pylab.subplot(1, 1, 1)

a = ax.hist(np.log(tec), bins=100)

bins = [0.0, 0.10, 0.15, 100.0]
for b in bins:
    ax.axvline(np.log(b), c='r')
vals, bins_out = np.histogram(tec, bins)

for thold, v, name in zip(bins[:-1], vals, ['good', 'close', 'fail']):
    print "thold =", thold, "name=", name, "%02d" % (float(v) *100 / len(tec)), "%"

pylab.show()

# top N worst ones: 
N = 20
tl = len(tec)
a = tec.index[tl-N:]
pickle.dump({'bad_epochs' :  list(a), 
             }, open('report.badepochs.pickle', 'w'))


print "The 5 best are", tec.index[:5]
print

print "The 5 worst in the 'best' category are", tec[tec < bins[1]].index[-5:]
print

print "The first 5 ok are", tec[tec > bins[1]].index[:5]
print

print "The worst is", tec.index[-5:]
print
 
