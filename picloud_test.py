import numpy as np
import cPickle as pickle
import os
import util2 as util
import measure
import methods
import organizedata
import subprocess
from ruffus import * 

import random
import cloud


total_tests = 50000000

def monteCarlo(num_test):
  """
  Throw num_test darts at a square
  Return how many appear within the quarter circle
  """
  num_in_circle = 0
  for _ in xrange(num_test):
    x = random.random()
    y = random.random()
    if x*x + y*y < 1.0:  #within the quarter circle
      num_in_circle += 1
  return num_in_circle

def calcPi():
  num_jobs = 8
  tests_per_call = total_tests/num_jobs
  jids = cloud.map(monteCarlo,[tests_per_call]*num_jobs, _type='c2')  #argument list has 8 duplicate elements
  num_in_circle_list = cloud.result(jids) #get list of counts
  num_in_circle = sum(num_in_circle_list)   #add the list together
  pi = (4 * num_in_circle) / float(total_tests)
  return pi

def files(dirname):
    p = subprocess.Popen("find data/ | wc -l ", shell=True, 
                         stdout=subprocess.PIPE)
    s = p.stdout.read()
    p.wait()
    return s

def cloud_files():
    jids = cloud.map(files, ['none'], _type='c2', 
                     _vol="my-vol")
    return cloud.result(jids)[0]

if __name__ == '__main__':
    #pi = calcPi()
    #print 'Pi determined to be %s' % pi
    print "files=", files('none')
    print "cloud files=", cloud_files()
