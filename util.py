import numpy as np

"""
UNITS: 

time is always in seconds
position is always in meters

we use a right-handed coordinate frame at all times, where the animal
is assumed to be moving in the x-y plane 

"""

DTYPE_STATE = [('x', np.float32), 
               ('y', np.float32), 
               ('phi', np.float32), 
               ('theta', np.float32)]

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

