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
    delta = dist_delta(xyconf['x'], xyconf['y'], 
                       xy_true['x'], xy_true['y'])
    
    return delta
    
def avg_delta_conf_threshold(delta, conf, tholds):
    """
    For a list of confidences, return the average error of all points
    as or more confident than that thold

    returns a vector as long as tholds
    """
    results = np.zeros(len(tholds))

    # stupid slow naive 
    for ti, thold in enumerate(tholds):
        results[ti] = np.mean(delta[conf >= thold])
    return results
