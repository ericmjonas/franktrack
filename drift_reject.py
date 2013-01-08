import numpy as np
import scipy.stats
from matplotlib import pylab


def norm_norm_peak(x_cur, sigma_d, sigma_e):
    """
    peak of N(x | x_cur, sigma_d^2)N(x | 0, sigma_e^2)
    """
    return x_cur / (1.0 + sigma_d**2/sigma_e**2)

#@profile
def rej_sample(x_cur, sigma_d, sigma_e):
    """
    returns sample from 
    N(x | x_cur, sigma_d^2)N(x | 0, sigma_e^2) 
    as well as the number of failed rejection attempts
    
    """

    lp_rv = scipy.stats.norm(0, sigma_e)
    norm_rv = scipy.stats.norm(x_cur, sigma_d)
    
    peak = norm_norm_peak(x_cur, sigma_d, sigma_e)
    def tgt_pdf(x):
        return lp_rv.pdf(x)*norm_rv.pdf(x)
    peak_val = tgt_pdf(peak)

    
    
    envelope_rv = scipy.stats.norm(peak, sigma_d)
    envelope_rv_val = envelope_rv.pdf(peak)
    
    failures = 0
    while True:
        x =  envelope_rv.rvs()
        u = np.random.rand()
        M =  1.0/envelope_rv_val * peak_val
        
        if u < (tgt_pdf(x) / (M * envelope_rv.pdf(x))):
            return x, failures
        else:
            failures += 1
    
def gen_plots():    
    MIN_X, MAX_X = (-10, 10)
    x = np.linspace(MIN_X, MAX_X, 1000)
    sigma_d = 1.0
    sigma_e = 2.0
    x_curs = [-8, -4, -1, 0, 1, 4, 8]
    for xi, x_cur in enumerate(x_curs):
        lp_rv = scipy.stats.norm(0, sigma_e)
        norm_rv = scipy.stats.norm(x_cur, sigma_d)
        pylab.subplot(len(x_curs), 2, 2*(xi)+1)
        pylab.title("x_cur= %f" % x_cur)
        def tgt_dist(x):
            return lp_rv.pdf(x)*norm_rv.pdf(x)
        pylab.plot(x, tgt_dist(x))
        peak = norm_norm_peak(x_cur, sigma_d, sigma_e)
        peak_val = tgt_dist(peak)
        
        
        envelope_rv = scipy.stats.norm(peak, sigma_d)
        envelope_rv_val = envelope_rv.pdf(peak)
        pylab.plot(x, envelope_rv.pdf(x)/ envelope_rv_val * peak_val, c='g')
        pylab.axvline(x_cur, c='r')
        pylab.axvline(peak, c='k')

        # now sample hist
        samps = [rej_sample(x_cur, sigma_d, sigma_e)[0] 
                 for _ in range(100000)]
        pylab.subplot(len(x_curs), 2, 2*xi+2)
        bins = np.linspace(MIN_X, MAX_X, 100+1)
        vals, dummy = np.histogram(samps, bins, normed=True)
        print vals
        pylab.scatter(bins[:-1], vals, c='r')
        td = tgt_dist(x)
        
        pylab.plot(x, td/np.sum(td)/(x[1]-x[0]), c='b')
        
    pylab.savefig('rej_attempts.pdf')

def sample():
    N = 100000
    x = np.zeros(N)
    x[0] = 0.0
    sigma_d = 0.1
    sigma_e = 2.0
    for i in range(1, N):
        x[i] = rej_sample(x[i-1], sigma_d, sigma_e)[0]
    pylab.subplot(2, 1, 1)
    pylab.plot(x)
    pylab.subplot(2, 1, 2)
    pylab.hist(x, normed=True, bins=50)
    pylab.savefig('drift.png', dpi=300)
        
#gen_plots()
#sample()
