import numpy as np
import template
from matplotlib import pylab

LENGTH = 14
FRONT_SIZE = 4
BACK_SIZE = 3

tr = template.TemplateRenderCircleBorder2(1.2, 0.0)

tr.set_params(LENGTH, FRONT_SIZE, BACK_SIZE)

phis = np.linspace(0, 2*np.pi, 12+1)
thetas = np.linspace(0., 
                    np.pi, 9)

pos = 1
for theta in thetas:
    for phi in phis:
        print theta, phi
        img= tr.render(phi, theta)
        ax = pylab.subplot(len(thetas), len(phis), pos)
        ax.imshow(img, interpolation='nearest', 
                  origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])

        pos +=1

pylab.show()

