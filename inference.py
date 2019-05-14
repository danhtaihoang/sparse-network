##========================================================================================
import numpy as np
from scipy import linalg

""" --------------------------------------------------------------------------------------
fit h0 and w based on Expectation Reflection
input: features x[l,n], target: y[l]
 output: h0, w[n]
"""
def fit(x,y,niter_max=100):    
    n = x.shape[1]
    
    x_av = np.mean(x,axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)
    c_inv = linalg.inv(c)

    # initial values
    h0 = 0.
    w = np.random.normal(0.0,1./np.sqrt(n),size=(n))
    
    cost = np.full(niter_max,100.)
    for iloop in range(niter_max):
        h = h0 + x.dot(w)
        y_model = np.tanh(h)    

        # stopping criterion
        cost[iloop] = ((y[:]-y_model[:])**2).mean()
        if iloop>0 and cost[iloop] >= cost[iloop-1]: break

        # update local field
        t = h!=0    
        h[t] *= y[t]/y_model[t]
        h[~t] = y[~t]

        # find w from h    
        h_av = h.mean()
        dh = h - h_av 
        dhdx = dh[:,np.newaxis]*dx[:,:]

        dhdx_av = dhdx.mean(axis=0)
        w = c_inv.dot(dhdx_av)
        h0 = h_av - x_av.dot(w)
        
    return h0,w
