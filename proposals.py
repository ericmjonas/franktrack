
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

class HigherIsotropic(ssm.proposal.Proposal):
    """
    This just has a much-wider proposal than the underlying
    transition model

    """
    def __init__(self):
        self.DELTA_T = 1/30. 
        self.VELOCITY_NOISE_STD = 0.05
        self.POS_NOISE_STD = 0.01
        self.PHI_NOISE_STD = 1.0 # a good chunk of noise; units? 

        self.THETA_DRIFT_SIZE = 0.1
        self.THETA_ENVELOPE_SIZE = np.pi/16.
        self.THETA_OFFSET = np.pi/2.

    @profile
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

        
