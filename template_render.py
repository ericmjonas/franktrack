"""
Assume that we have a template for tracking objects in a 2-d plane.

questions: do i want them to be point sources that I do the geometry
on?  Do I assume that the areas between are transparent or opaque? Can
the template specify?

Change to a body-centric axis

"""
import numpy as np
from matplotlib import pylab

# a template is just a R x C np MA of values between 0 and 1, 
# oriented along phi=0

R = 8
C = 16

template = np.ma.zeros((R, C), dtype=np.float)
template[3:5, 1:3] = 1.0
template[1:7, 10:16] = 1.0
# test gray border
template[0, :] = 0.5 
template[-1, :] = 0.5 
template[:, 0] = 0.5 
template[:, -1] = 0.5 

alpha = np.pi/4 # about z-axis
beta = np.pi/4 # about y-axis
gamma = 0.0 # about x-axis
cos_a = np.cos(alpha)
cos_b = np.cos(beta)
cos_y = np.cos(gamma)
sin_a = np.sin(alpha)
sin_b = np.sin(beta)
sin_y = np.sin(gamma)

rot_mat = [[cos_a*cos_b, cos_a*sin_b*sin_y - sin_a*cos_y, cos_a *sin_b * cos_y  + sin_a * sin_b], 
           [sin_a*cos_b, sin_a*sin_b*sin_y + cos_a*cos_y, sin_a *sin_b*cos_y - cos_a * sin_y], 
           [-sin_b, cos_b * sin_y, cos_b * cos_y]]

rot_mat = np.array(rot_mat, dtype=np.float32)
print "rotation matrix is", rot_mat
OS = 1

points = np.zeros((3, R*C*OS*OS), dtype=np.float32)
pix = np.zeros((R* C*OS*OS), dtype=np.float32)
pos = 0
for y in range(R*OS):
    for x in range(C*OS):
        points[:, pos] = x - (C*OS-1)/2., y-(R*OS-1)/2., 0.0
        pix[pos] = template[np.ceil(float(y)/OS), 
                            np.ceil(float(x)/OS)]
        pos += 1

# transformed 
rot_mat_inv = np.linalg.inv(rot_mat)
transformed = np.dot(rot_mat, points)

#pylab.scatter(transformed[0, :], transformed[1, :], 
#               c = pix)
#pylab.show()
BORDER = 3
TGTW = C * OS * BORDER
TGTH = R * OS * BORDER
tgt = np.zeros((TGTH, TGTW), dtype=np.float)

# compute the points corresponding to these
tgt_pts = np.len(tgt.flat)
pos = 0
for y in range(TGTH):
    for x in range(TGTW):
        tgt_pts[pos] = x - (TGTH-1)/2., y- (TGTH-1)/2.0, 0.0

tgt_p = np.dot(rot_mat_inv, tgt_pts)

for pi in xrange(len(tgt.flat)):
    x, y, z = tgt_p = 
    pixel = pix[pi]
    tgt_x, tgt_y = get_tgt_pt(x, y)
    print x, y, tgt_x, tgt_y
    print tgt_y, tgt_x, pixel
    if tgt_y < tgt.shape[0] and tgt_x < tgt.shape[1]:
        tgt[tgt_y, tgt_x] = pixel
pylab.imshow(tgt, interpolation='nearest', cmap=pylab.cm.gray)
pylab.show()
