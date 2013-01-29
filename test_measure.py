import numpy as np
import measure
import util2 as util

def test_compute_phi():
    front = np.array([[1, 0]])
    back = np.array([[0, 0]])
    phi = measure.compute_phi(front, back)
    print phi, np.pi / 2. 

    front = np.array([[0, 1]])
    back = np.array([[0, 0]])
    phi = measure.compute_phi(front, back)
    print phi, np.pi / 2. 

