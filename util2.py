import numpy as np
import scipy.stats
import os

"""
UNITS: 

time is always in seconds
position is always in meters

we use a right-handed coordinate frame at all times, where the animal
is assumed to be moving in the x-y plane 

"""

DATA_DIR = "data/fl"
def ddir(x):
    return os.path.join(DATA_DIR, x)

REPORT_DIR = "results"
def rdir(x):
    return os.path.join(REPORT_DIR, x)


DTYPE_STATE = [('x', np.float32), 
               ('y', np.float32), 
               ('phi', np.float32), 
               ('theta', np.float32)]


class Environmentz(object):
    """ 
    By default, an environment's origin is in the lower-left corner
    """

    def __init__(self, room_dim, image_dim):
        self.room_dim = room_dim
        self.image_dim = image_dim
        if room_dim[0] > room_dim[1]:
            if image_dim[0] <= image_dim[1]:
                raise Exception("The image aspect ratio is very different from the room aspect ratio, are you sure you specified coordinates correctly?")

        self.room_lower_left = (0.0, 0.0)

        self.ppm = (float(image_dim[0]) / room_dim[0], 
                    float(image_dim[1])/ room_dim[1])

        self.gc = GeomConverter(self.image_dim, 
                                self.ppm, self.room_lower_left)

    def get_room_center_real_xy(self):
        return (self.room_lower_left[0] + self.room_dim[1]/2, 
                self.room_lower_left[1] + self.room_dim[0]/2)

def compute_pos(length, x, y, phi, theta):
    """
    x-y : center of dumbbell

    We use the standard physics coordinate system, modeling
    the big diode as a vector of r=length/2 centered at the
    head of the animal. Thus :

    phi is angle with X axis
    theta is angle with positive z axis (in plane is theta=pi/2)

    """
    r = length / 2

    vec_x = r * np.sin(theta) * np.cos(phi)
    vec_y = r * np.sin(theta) * np.sin(phi)
    vec_z = r * np.cos(theta)

    front_pos = (x + vec_x, y + vec_y, vec_z)
    back_pos = (x - vec_x, y - vec_y, -vec_z)

    return front_pos, back_pos


class GeomConverter(object):
    """
    convert from pix to real and vice versa
    """
    def __init__(self, image_geometry, 
                 pix_per_meter, lower_left):
        """
        image_geometry = (pixels_col, pixels_row)
        pix_per_meter = (pixels_per_vertical_meter, pixels_per_horizontal_meter)
        lower_left = (y_in_meters, x_in_meters)
        
        """
        self.image_geometry = image_geometry
        self.pix_per_meter = pix_per_meter
        self.lower_left = lower_left

    def real_to_image(self, x, y):
        orig_x = x - self.lower_left[1]
        orig_y = y - self.lower_left[0]
        
        return (orig_x * self.pix_per_meter[1], 
                orig_y * self.pix_per_meter[0])

    def image_to_real(self, pix_x, pix_y):
        y = float(pix_y ) / self.pix_per_meter[0]
        x = float(pix_x) / self.pix_per_meter[1]
        return self.lower_left[1] + x, self.lower_left[0] + y

        

def normal_rv(x, var):
    return np.random.normal(x, np.sqrt(var))

def sample_path_from_prior(model, N):
    """
    returns x, y

    """
    
    x = [model.sample_latent_from_prior()]  

    y= [model.sample_obs(x[0])]

    for n in range(1, N):
        x.append(model.sample_next_latent(x[-1], n))
        y.append(model.sample_obs(x[-1]))

    return x, y

def scores_to_prob(x):
    xn = x - np.max(x)
    a = np.exp(xn)
    s = np.sum(a)
    return a / s


def log_multinorm_dens(x, mu, cov):
    k = len(x)
    assert len(mu) == k
    assert cov.shape[0] == cov.shape[1]
    assert cov.shape[0] == k
    
    a = -k/2.0 * np.log(2*np.pi)
    b = -0.5 * np.log(np.linalg.det(cov))

    delta = x - mu
    inv_cov = np.linalg.inv(cov)
    c = -0.5 * np.dot(np.dot(delta.T, inv_cov), delta)

    return a + b + c

def credible_interval(samples, weights, 
                      lower=0.05, upper=0.95):
    """
    samples is  a vector of real-valued samples, weights
    are a weight associated with each sample

    """
    s_i = np.argsort(samples)
    s_sorted = samples[s_i]
    w_sorted = weights[s_i]
    cs = np.cumsum(w_sorted)
    creds = np.searchsorted(cs, [lower, upper])
    return np.array(s_sorted[creds])
    
def chunk(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

    
def extract_region_safe(a, r, c, n, defaultval = 0):
    """
    Extract the region centered at r, c with n pixels
    on each side, appropriately centered in the output array
    """
    
    region = np.zeros((n*2+1, n*2+1), dtype=a.dtype)
    region[:, :] = defaultval
    
    R, C = a.shape
    
    pix_left = min(c-n, n)
    if pix_left < 0: pix_left = 0

    pix_right = min(C - c -1, n)
    
    pix_top = min(r, n)
    pix_bot = min(R - r - 1, n)

    roi = a[r - pix_top:r+pix_bot+1, 
            c - pix_left:c + pix_right+1]

    try:
        region[(n - pix_top) : (n + pix_bot+ 1),
               (n - pix_left) : (n + pix_right + 1)] = roi
    except ValueError:
        s = "a.shape=(%d %d),  %d %d %d %d, %s, %d, %d" % (R, C, pix_left, pix_right, pix_top, pix_bot, roi.shape, r, c)
        raise Exception(s)

    return region

    
