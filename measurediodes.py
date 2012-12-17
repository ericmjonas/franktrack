"""
Code to use heuristics to track LEDs. This is a great time to use 
PF-mcmc but I'm lazy and feel like fucking around
"""

import numpy as np
import cPickle as pickle
from util2 import ddir, rdir
import util2 as util

from matplotlib import pylab

import os
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

import numpy as np
from scipy import ndimage
import skimage.measure
import skimage.feature
from matplotlib import pylab
import organizedata



def frame_clust_points(im, THOLD, min_distance, clust_eps, min_samples):
    im_thold = im.copy()
    im_thold[im_thold < THOLD] = 0
    coordinates = skimage.feature.peak_local_max(im_thold, 
                                                 min_distance=min_distance)
    
    D = distance.squareform(distance.pdist(coordinates.astype(float)))
    def to_sim(x):
        return 1.0 - x / 100. 
    S = to_sim(D)

    db = DBSCAN(eps=clust_eps, min_samples=min_samples).fit(S)
    labels = db.labels_

    num_clusters = len(np.unique(labels))
    if -1. in np.unique(labels):
        num_clusters -= 1

    cluster_centers = np.zeros((num_clusters, 2))
    for i in range(num_clusters):
        a = coordinates[np.argwhere(labels == float(i))]
        center = np.mean(a, axis=0)
        cluster_centers[i] = center
    return cluster_centers


def average_diode_sep():
    clust_eps = 0.2
    min_dist = 2.0
    min_samples = 3.0
    thold = 240
    
    FRAMES =  np.arange(4000)*2
    
    dataset = "bukowski_02.C"
    cf = pickle.load(open(os.path.join(ddir(dataset), 'config.pickle')))
    region = pickle.load(open(os.path.join(ddir(dataset), 'region.pickle')))
    
    env = util.Environmentz(cf['field_dim_m'], cf['frame_dim_pix'])
    x_min, y_min = env.gc.real_to_image(region['x_pos_min'], region['y_pos_min'])
    x_max, y_max = env.gc.real_to_image(region['x_pos_max'], region['y_pos_max'])
    print x_min, x_max
    print y_min, y_max
    if y_min < 0:
        y_min = 0
    frame_images = organizedata.get_frames(ddir(dataset), FRAMES)
    num_clusters = np.zeros(len(FRAMES))
    dists = []
    for fi, im in enumerate(frame_images):
        im = im[y_min:y_max+1, x_min:x_max+1]

        centers = frame_clust_points(im, 240, min_dist,
                                     clust_eps, min_samples)

        num_clusters[fi] = len(centers)
        if len(centers) == 2:
            dists.append(distance.pdist(centers)[0])
    dists = np.array(dists)
    pylab.hist(dists[dists < 50], bins=20)

    pylab.savefig("average_diode_sep.%s.png" % dataset, dpi=300)

def find_params():

    FRAMES =  np.arange(30)*100

    frame_images = organizedata.get_frames(ddir("bukowski_04.W2"), FRAMES)
    print "DONE READING DATA"

    CLUST_EPS = np.linspace(0, 0.5, 10)
    MIN_SAMPLES = [2, 3, 4, 5]
    MIN_DISTS = [2, 3, 4, 5, 6]
    THOLD = 240

    fracs_2 = np.zeros((len(CLUST_EPS), len(MIN_SAMPLES), len(MIN_DISTS)))

    for cei, CLUST_EP in enumerate(CLUST_EPS):
        for msi, MIN_SAMPLE in enumerate(MIN_SAMPLES):
            for mdi, MIN_DIST in enumerate(MIN_DISTS):
                print cei, msi, mdi
                numclusters = np.zeros(len(FRAMES))
                for fi, im in enumerate(frame_images):
                    centers = frame_clust_points(im, THOLD, MIN_DIST, 
                                                 CLUST_EP, MIN_SAMPLE)
                    # cluster centers
                    numclusters[fi] = len(centers)
                fracs_2[cei, msi, mdi] = float(np.sum(numclusters == 2))/len(numclusters)
    pylab.figure(figsize=(12, 8))
    for mdi, MIN_DIST in enumerate(MIN_DISTS):
        pylab.subplot(len(MIN_DISTS), 1, mdi+1)

        for msi in range(len(MIN_SAMPLES)):
            pylab.plot(CLUST_EPS, fracs_2[:, msi, mdi], label='%d' % MIN_SAMPLES[msi])
        pylab.title("min_dist= %3.2f" % MIN_DIST)
    pylab.legend()
    pylab.savefig('test.png', dpi=300)
    

if __name__ == "__main__":
    average_diode_sep()
