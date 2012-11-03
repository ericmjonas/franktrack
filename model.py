
import numpy as np


DTYPE_LATENT_STATE = [('x', np.float32), 
                      ('y', np.float32), 
                      ('xdot', np.float32), 
                      ('ydot', np.float32), 
                      ('phi', np.float32), 
                      ('theta', np.float32)]


class LinearModel(object):
    """
    Right now just enough parts for the bootstrap filter. 
    All units are MKS
    """
    
    def __init__(self, env, likelihood_evaluator, 
                 VELOCITY_NOISE_STD = 0.1, 
                 POS_NOISE_STD = 0.01):
        """
        This is a basic linear evolution model with
        a generative model for the observation likelihood. 
        Temporal evolution is modified by a random-walk model
        with gaussian noise. 

        TODO: Stick prior on velocity components, positon components, 
        phi. This will make scoring easy but sampling new latents hard. 
        
        """
        self.DELTA_T = 1/30. 
        self.VELOCITY_NOISE_STD = VELOCITY_NOISE_STD
        self.POS_NOISE_STD = POS_NOISE_STD
        self.PHI_NOISE_STD = 0.2 # a good chunk of noise; units? 
        self.THETA_NOISE_STD = 0.1 # 
        
        self.env = env
        self.likelihood_evaluator = likelihood_evaluator

    def sample_latent_from_prior(self, N=1):
        """
        For initialization
        """
        s = np.zeros(N, dtype=DTYPE_LATENT_STATE)

        s['x'] = np.random.rand(N) * self.env.room_dim[1]*0.99
        s['y'] = np.random.rand(N) * self.env.room_dim[0]*0.99
        s['xdot'] = np.random.normal(0, 0.01, N)
        s['ydot'] = np.random.normal(0, 0.01, N)
        s['phi'] = np.random.rand(N) * np.pi*2
        s['theta'] = np.random.normal(0, 0.1)  + np.pi/2
        if N == 1:
            return s[0]
        return s
        

    def score_obs(self, yn, xn):
        """
        Score the current latent state against
        the current observation
        """
        return self.likelihood_evaluator.score_state(xn, yn)


    def sample_next_latent(self, xn, n):
        """
        Return X_{n+1} | x_n

        """
        # right now this is totally linear, gaussian with
        # an identity covariance matrix
        SN = 1
        x_next = np.zeros(SN, dtype=DTYPE_LATENT_STATE).view(np.recarray)
        x_det = xn['x'] + xn['xdot'] * self.DELTA_T
        y_det = xn['y'] + xn['ydot'] * self.DELTA_T

        x_next['x']= x_det + np.random.normal(0, 
                                    self.POS_NOISE_STD, size=SN)
        x_next['y'] = y_det + np.random.normal(0, 
                                    self.POS_NOISE_STD, size=SN)
    
        x_next['xdot'] = np.random.normal(xn['xdot'], 
                                       self.VELOCITY_NOISE_STD, size=SN)

        x_next['ydot'] = np.random.normal(xn['ydot'], 
                                       self.VELOCITY_NOISE_STD, size=SN)

        x_next['phi'] = np.random.normal(xn['phi'], 
                                       self.PHI_NOISE_STD, size=SN)

        x_next['theta'] = np.random.normal(xn['theta'], 
                                       self.THETA_NOISE_STD, size=SN)
        
        
        return x_next[0]


class LinearStudentTModel(object):
    """
    Right now just enough parts for the bootstrap filter. 
    All units are MKS
    """
    
    def __init__(self, env, likelihood_evaluator, 
                 POS_NOISE_STD=0.01, VELOCITY_NOISE_STD=0.1):
        """
        This is a basic linear evolution model with
        a generative model for the observation likelihood. 
        Temporal evolution is modified by a random-walk model
        with gaussian noise. 

        TODO: Stick prior on velocity components, positon components, 
        phi. This will make scoring easy but sampling new latents hard. 
        
        """
        self.DELTA_T = 1/30. 
        self.VELOCITY_NOISE_STD = POS_NOISE_STD
        self.POS_NOISE_STD = VELOCITY_NOISE_STD
        self.PHI_NOISE_STD = 0.2 # a good chunk of noise; units? 
        self.THETA_NOISE_STD = 0.1 # 
        
        self.env = env
        self.likelihood_evaluator = likelihood_evaluator

    def sample_latent_from_prior(self, N=1):
        """
        For initialization
        """
        s = np.zeros(N, dtype=DTYPE_LATENT_STATE)

        s['x'] = np.random.rand(N) * self.env.room_dim[1]*0.99
        s['y'] = np.random.rand(N) * self.env.room_dim[0]*0.99
        s['xdot'] = np.random.normal(0, 0.01, N)
        s['ydot'] = np.random.normal(0, 0.01, N)
        s['phi'] = np.random.rand(N) * np.pi*2
        s['theta'] = np.random.normal(0, 0.1)  + np.pi/2
        if N == 1:
            return s[0]
        return s
        

    def score_obs(self, yn, xn):
        """
        Score the current latent state against
        the current observation
        """
        return self.likelihood_evaluator.score_state(xn, yn)


    def sample_next_latent(self, xn, n):
        """
        Return X_{n+1} | x_n

        """
        # right now this is totally linear, gaussian with
        # an identity covariance matrix
        SN = 1
        x_next = np.zeros(SN, dtype=DTYPE_LATENT_STATE).view(np.recarray)
        x_det = xn['x'] + xn['xdot'] * self.DELTA_T
        y_det = xn['y'] + xn['ydot'] * self.DELTA_T

        x_next['x']= x_det + np.random.standard_t(1, size=SN) * self.POS_NOISE_STD 
    
        x_next['y'] = y_det + np.random.standard_t(1, size=SN)* self.POS_NOISE_STD
    
        x_next['xdot'] = xn['xdot'] + np.random.standard_t(1, size=SN) * self.VELOCITY_NOISE_STD
                  

        x_next['ydot'] = xn['ydot'] + np.random.standard_t(1, size=SN) * self.VELOCITY_NOISE_STD

        x_next['phi'] = np.random.normal(xn['phi'], 
                                       self.PHI_NOISE_STD, size=SN)

        x_next['theta'] = np.random.normal(xn['theta'], 
                                       self.THETA_NOISE_STD, size=SN)
        
        
        return x_next[0]
