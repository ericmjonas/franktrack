
"""
Proposal ideas: 
Just wider proposals

Something to help if tracking is lost

"""

import ssm.proposal
import numpy as np
import util2 as util
import model
import drift_reject
from matplotlib import pylab
import matplotlib

class HigherIsotropic(ssm.proposal.Proposal):
    """
    This just has a much-wider proposal than the underlying
    transition model

    """
    def __init__(self):
        self.DELTA_T = 1/30. 
        self.VELOCITY_NOISE_STD = 0.05
        self.POS_NOISE_STD = 0.01
        self.PHI_NOISE_STD = 0.3 # a good chunk of noise; units? 

        self.THETA_DRIFT_SIZE = 0.01
        self.THETA_ENVELOPE_SIZE = np.pi/32.
        self.THETA_OFFSET = np.pi/2.

    def sample(self, y, x_prev, n):
        """
        draw a sample from the proposal conditioned on current y and
        previous x
        
        """
        SN = 1
        x_next = np.zeros(SN, dtype=model.DTYPE_LATENT_STATE).view(np.recarray)
        x_det = x_prev['x'] + x_prev['xdot'] * self.DELTA_T
        y_det = x_prev['y'] + x_prev['ydot'] * self.DELTA_T

        norm = np.random.normal

        x_next['x']= x_det + norm(0, 
                                    self.POS_NOISE_STD, size=SN)
        x_next['y'] = y_det + norm(0, 
                                    self.POS_NOISE_STD, size=SN)
    
        x_next['xdot'] = norm(x_prev['xdot'], 
                                       self.VELOCITY_NOISE_STD, size=SN)

        x_next['ydot'] = norm(x_prev['ydot'], 
                                       self.VELOCITY_NOISE_STD, size=SN)

        x_next['phi'] = norm(x_prev['phi'], 
                                       self.PHI_NOISE_STD, size=SN)


        val, failures =  drift_reject.rej_sample(x_prev['theta'] - self.THETA_OFFSET, 
                                                 self.THETA_DRIFT_SIZE, 
                                                 self.THETA_ENVELOPE_SIZE)
        x_next['theta'] = self.THETA_OFFSET + val

        
        
        return x_next[0]
        
    def score(self, x, y, x_prev, n):
        """
        Score a particular proposal 

        """
        x_det = x_prev['x'] + x_prev['xdot'] * self.DELTA_T
        y_det = x_prev['y'] + x_prev['ydot'] * self.DELTA_T
        
        score = 0.0
        nd = ssm.util.log_norm_dens
        score += nd(x['x'], x_det, self.POS_NOISE_STD**2)
        score += nd(x['y'], y_det, self.POS_NOISE_STD**2)
        
        score += nd(x['xdot'], x_prev['xdot'], 
                                        self.VELOCITY_NOISE_STD**2)
        score += nd(x['ydot'], x_prev['ydot'], 
                                        self.VELOCITY_NOISE_STD**2)

        score += nd(x['phi'], x_prev['phi'], 
                    self.PHI_NOISE_STD)

        # now the theta likelihood is fun because it's like, the product
        # of two things

        t_o = x['theta'] - self.THETA_OFFSET
        tn_o = x_prev['theta'] - self.THETA_OFFSET
        score += nd(tn_o, t_o, self.THETA_DRIFT_SIZE) 
        score += nd(tn_o, 0, self.THETA_ENVELOPE_SIZE)


        return score

        
class HigherIsotropicAndData(object):
    """
    I want to pass in an existing higher-isotropic kernel but just say
    "only worry about non-xy terms" 

    But for the time being, we're living in copy-paste land

    This proposal kernel uses deterministic functions of the image to 
    postulate a gaussian based on the location of the filtered
    bright points in the image. 

    In some image regimes this will be great, in others it will 
    fail horribly; remember to just use as a proposal kernel inside a
    mixture with something that respects state dynamics better. 


    """

    
    def __init__(self, env,
                 img_to_points):
        """
        img_to_points is a feature-extractor that takes in a 
        image and returns a list of points in pixel coordiantes
        """

        self.DELTA_T = 1/30. 
        self.VELOCITY_NOISE_STD = 0.05
        self.POS_NOISE_STD = 0.01
        self.PHI_NOISE_STD = 0.3 # a good chunk of noise; units? 

        self.THETA_DRIFT_SIZE = 0.01
        self.THETA_ENVELOPE_SIZE = np.pi/32.
        self.THETA_OFFSET = np.pi/2.
    
        self.gaussians = {}
        self.img_to_points = img_to_points
        self.env = env

    def compute_gaussian_interest(self, img):
        """
        Use feature extractor to get points, then compute mean, var in both dim, 
        """
        
        fc = self.img_to_points(img)
        points = np.zeros((len(fc), 2), dtype=np.float32)
        if len(fc) > 0:
            points[:] = [self.env.gc.image_to_real(*x) for x in np.fliplr(fc)]
        
            means = np.mean(points, axis=0)
            vars = np.var(points, axis=0)
            vars = np.maximum(vars, np.ones_like(vars)*0.001)
            print "Variance is", vars
            return means, vars
        else:
            return None, None

    def cached_mean_var(self, y, n):
        if n in self.gaussians:
            return self.gaussians[n]
        else:
            m, v = self.compute_gaussian_interest(y)
            print "V=", v
            if v == None:
                self.gaussians[n] = None, None
                return None, None
            else:
                std = np.sqrt(v)
                pos_x, pos_y = self.env.gc.real_to_image(m[0], m[1])
                std_x, std_y = self.env.gc.real_to_image(std[0], std[1])
                # ax = pylab.subplot(1, 1, 1)
                # ax.imshow(y)
                # print "V=", v
                # e = matplotlib.patches.Ellipse((pos_x, pos_y), width= std_x, 
                #                                height=std_y)
                # ax.add_artist(e)
                # ax.axhline(pos_y)
                # ax.axvline(pos_x)
                # pylab.show()
                self.gaussians[n] = m, v
                return m, v
        
    def sample(self, y, x_prev, n):
        """
        draw a sample from the proposal conditioned on current y and
        previous x
        
        """
        est_mu, est_var = self.cached_mean_var(y, n)

        SN = 1
        x_next = np.zeros(SN, dtype=model.DTYPE_LATENT_STATE).view(np.recarray)
        x_det = x_prev['x'] + x_prev['xdot'] * self.DELTA_T
        y_det = x_prev['y'] + x_prev['ydot'] * self.DELTA_T

        norm = np.random.normal
        if est_mu != None and (est_var > 0.001).all():
            x_next['x']= norm(est_mu[0], np.sqrt(est_var[0]))
            x_next['y'] = norm(est_mu[1], np.sqrt(est_var[1]))
        else:
            x_next['x']= x_det + norm(0, 
                                      self.POS_NOISE_STD, size=SN)
            x_next['y'] = y_det + norm(0, 
                                       self.POS_NOISE_STD, size=SN)
    
        x_next['xdot'] = norm(x_prev['xdot'], 
                                       self.VELOCITY_NOISE_STD, size=SN)

        x_next['ydot'] = norm(x_prev['ydot'], 
                                       self.VELOCITY_NOISE_STD, size=SN)

        x_next['phi'] = norm(x_prev['phi'], 
                                       self.PHI_NOISE_STD, size=SN)


        val, failures =  drift_reject.rej_sample(x_prev['theta'] - self.THETA_OFFSET, 
                                                 self.THETA_DRIFT_SIZE, 
                                                 self.THETA_ENVELOPE_SIZE)
        x_next['theta'] = self.THETA_OFFSET + val

        
        
        return x_next[0]
        
    def score(self, x, y, x_prev, n):
        """
        Score a particular proposal 

        """
        est_mu, est_var = self.cached_mean_var(y, n)

        x_det = x_prev['x'] + x_prev['xdot'] * self.DELTA_T
        y_det = x_prev['y'] + x_prev['ydot'] * self.DELTA_T

        score = 0.0
        nd = ssm.util.log_norm_dens
        if est_mu != None and (est_var > 0.001).all():
            score += nd(x['x'], est_mu[0], est_var[0])
            score += nd(x['y'], est_mu[1], est_var[1])
        else:
            score += nd(x['x'], x_det, self.POS_NOISE_STD**2)
            score += nd(x['y'], y_det, self.POS_NOISE_STD**2)
            
        
        score += nd(x['xdot'], x_prev['xdot'], 
                                        self.VELOCITY_NOISE_STD**2)
        score += nd(x['ydot'], x_prev['ydot'], 
                                        self.VELOCITY_NOISE_STD**2)

        score += nd(x['phi'], x_prev['phi'], 
                    self.PHI_NOISE_STD)

        # now the theta likelihood is fun because it's like, the product
        # of two things

        t_o = x['theta'] - self.THETA_OFFSET
        tn_o = x_prev['theta'] - self.THETA_OFFSET
        score += nd(tn_o, t_o, self.THETA_DRIFT_SIZE) 
        score += nd(tn_o, 0, self.THETA_ENVELOPE_SIZE)


        return score
