import numpy as np
import os
import cPickle as pickle
import time
import cloud

#cloud.start_simulator()

N = 1e7
M = 40000
K = 1

def testfunc(a):
    x = np.arange(0, N, dtype=np.float32)
    y = np.arange(0, M, dtype=np.float32)
    z = np.convolve(x, y)
    q = np.sum(z)
    return q + a

def serial():

    t1 = time.time()
    for i in range(K):
        a = testfunc(i)
    t2 = time.time()
    return t2-t1

def picloud():
    t1 = time.time()
    jids = cloud.map(testfunc, np.arange(K), 
                     _type='f2', _vol="my-vol", _env="base/precise")
    # get the results
    cloud.result(jids) 
    t2 = time.time()
    return t2-t1

print "Local serial time=", serial()
print "picloud time=", picloud()
