import numpy as np

"""
Evaluation code

"""

def dist_delta(x, y, true_x, true_y):
    """
    """
    return np.sqrt((x-true_x)**2 + (y-true_y)**2)

def xy_compare(xyconf, xy_true):
    """
    Compare two x-y vectors returning
    """
    N = len(xyconf)
    delta = dist_delta(xyconf['x'], xyconf['y'], 
                       xy_true['x'][:N], xy_true['y'][:N])
    
    return delta
    
def avg_delta_conf_threshold(delta, conf, tholds):
    """
    For a list of confidences, return the average error of all points
    as or more confident than that thold

    returns a vector as long as tholds
    """
    N = len(delta)
    results = np.zeros(len(tholds), dtype=np.float32)
    fractions = np.zeros(len(tholds), dtype=np.float32)

    # stupid slow naive 
    for ti, thold in enumerate(tholds):
        idx = np.argwhere(conf >= thold)
        results[ti] = np.mean(delta[idx])
        fractions[ti] = len(idx) / float(N)
    return results, fractions
