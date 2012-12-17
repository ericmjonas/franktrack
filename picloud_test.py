import numpy as np
import os
import cPickle as pickle

import bruteforcelikelihood as bfl
import cloud

#cloud.start_simulator()

dataset_name = "bukowski_04.W2"
dataset_dir = os.path.join(bfl.FL_DATA, dataset_name)
dataset_config_filename = os.path.join(dataset_dir, "config.pickle")
frame_hist_filename = os.path.join(dataset_dir, "framehist.npz")

cf = pickle.load(open(dataset_config_filename))


x_range = np.linspace(0, cf['field_dim_m'][1], 200)
y_range = np.linspace(0, cf['field_dim_m'][0], 200)
phi_range = np.linspace(0, 2*np.pi, 16)
theta_range = np.array([np.pi/2.])

sv = bfl.create_state_vect(y_range, x_range, phi_range, theta_range)

# now the input args
chunk_size = 10000
chunks = int(np.ceil(len(sv) / float(chunk_size)))
frames = np.arange(10)*50

args = []
for i in range(chunks):
    args += [  (i*chunk_size, (i+1)*chunk_size)]*len(frames)
               

print "THERE ARE", chunks, "CHUNKS"


CN = chunks
results = []
for fi, frame in enumerate(frames):
    jids = cloud.map(bfl.picloud_score_frame, [dataset_name]*CN,
                     [x_range]*CN, [y_range]*CN, 
                     [phi_range]*CN, [theta_range]*CN, 
                     args, [frame]*CN,  
                     _type='f2', _vol="my-vol", _env="base/precise")
    results.append(jids)

# get the results
[cloud.result(jids) for jids in results]
