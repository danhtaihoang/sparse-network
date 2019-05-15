##========================================================================================
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance

#=========================================================================================
def itab(n,m):    
    i1 = np.zeros(n)
    i2 = np.zeros(n)
    for i in range(n):
        i1[i] = i*m
        i2[i] = (i+1)*m

    return i1.astype(int),i2.astype(int)
#=========================================================================================
# generate coupling matrix w0: wji from j to i
def generate_interactions(n,m,g,sp):
    nm = n*m
    w = np.random.normal(0.0,g/np.sqrt(nm),size=(nm,nm))
    i1tab,i2tab = itab(n,m)

    # sparse
    for i in range(n):
        for j in range(n):
            if (j != i) and (np.random.rand() < sp): 
                w[i1tab[i]:i2tab[i],i1tab[j]:i2tab[j]] = 0.
    
    # sum_j wji to each position i = 0                
    for i in range(n):        
        i1,i2 = i1tab[i],i2tab[i]              
        w[:,i1:i2] -= w[:,i1:i2].mean(axis=1)[:,np.newaxis]            

    # no self-interactions
    for i in range(n):
        i1,i2 = i1tab[i],i2tab[i]
        w[i1:i2,i1:i2] = 0.   # no self-interactions

    # symmetry
    for i in range(nm):
        for j in range(nm):
            if j > i: w[i,j] = w[j,i]       
        
    return w
#=========================================================================================
def generate_external_local_field(n,m,g):  
    nm = n*m
    h0 = np.random.normal(0.0,g/np.sqrt(nm),size=nm)

    i1tab,i2tab = itab(n,m) 
    for i in range(n):
        i1,i2 = i1tab[i],i2tab[i]
        h0[i1:i2] -= h0[i1:i2].mean(axis=0)

    return h0
#=========================================================================================
# 2018.10.27: generate time series by MCMC
def generate_sequences(w,h0,n,m,l): 
    i1tab,i2tab = itab(n,m)
    
    # initial s (categorical variables)
    s_ini = np.random.randint(0,m,size=(l,n)) # integer values

    # onehot encoder 
    enc = OneHotEncoder(n_values=m)
    s = enc.fit_transform(s_ini).toarray()

    nrepeat = 5*n
    for irepeat in range(nrepeat):
        for i in range(n):
            i1,i2 = i1tab[i],i2tab[i]

            h = h0[np.newaxis,i1:i2] + s.dot(w[:,i1:i2])  # h[t,i1:i2]

            k0 = np.argmax(s[:,i1:i2],axis=1)
            for t in range(l):
                k = np.random.randint(0,m)                
                while k == k0[t]:
                    k = np.random.randint(0,m)
                
                if np.exp(h[t,k] - h[t,k0[t]]) > np.random.rand():
                    s[t,i1:i2],s[t,i1+k] = 0.,1.

        if irepeat%n == 0: print('irepeat:',irepeat) 

    return s

#===================================================================================================
# 2018.12.22: inverse of covariance between values of x
def cov_inv(x,y):
    l,mx = x.shape
    my = y.shape[1]
    cab_inv = np.empty((my,my,mx,mx))
    
    for ia in range(my):
        for ib in range(my):
            if ib != ia:
                eps = y[:,ia] - y[:,ib]

                which_ab = eps !=0.                    
                xab = x[which_ab]          
                xab_av = np.mean(xab,axis=0)
                dxab = xab - xab_av
                cab = np.cov(dxab,rowvar=False,bias=True)

                cab_inv[ia,ib,:,:] = linalg.pinv(cab,rcond=1e-15)    
                
    return cab_inv  
#=========================================================================================
# 2018.12.28: fit interaction to residues at position i
# additive update
def fit_additive(x,y,regu,nloop=10):        
    mx = x.shape[1]
    my = y.shape[1]
    
    # find elements having low probs, set w = 0
    #iprobs = y.sum(axis=0)/float(y.shape[0])
    #ilow = [i for i in range(my) if iprobs[i] < 0.02]
    #print(ilow)

    x_av = x.mean(axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)
    c_inv = linalg.pinvh(c)

    w = np.random.normal(0.0,1./np.sqrt(mx),size=(mx,my))
    h0 = np.random.normal(0.0,1./np.sqrt(mx),size=my)

    cost = np.full(nloop,100.)         
    for iloop in range(nloop):
        h = h0[np.newaxis,:] + x.dot(w)

        p = np.exp(h)
        p_sum = p.sum(axis=1)            
        p /= p_sum[:,np.newaxis]

        #cost[iloop] = ((y - p)**2).mean() + l1*np.sum(np.abs(w))
        cost[iloop] = ((y - p)**2).mean() + regu*np.sum(w**2)
        #print(iloop,cost[iloop])
        if iloop > 1 and cost[iloop] >= cost[iloop-1]: break

        h += y - p

        h_av = h.mean(axis=0)
        dh = h - h_av

        dhdx = dh[:,np.newaxis,:]*dx[:,:,np.newaxis]
        dhdx_av = dhdx.mean(axis=0)
        w = c_inv.dot(dhdx_av)            
        h0 = h_av - x_av.dot(w)

        #if len(ilow) > 0:
        #    w[:,ilow] = 0.
        #    h0[ilow] = 0.

        w -= w.mean(axis=0) 
        h0 -= h0.mean()

    return w,h0,cost,iloop  

#=========================================================================================
# 2019.02.25: fit interaction to residues at position i
# multiplicative update (new version, NOT select each pair as the old version)
def fit_multiplicative(x,y,nloop=10):
    mx = x.shape[1]
    my = y.shape[1]

    y2 = 2*y-1

    x_av = x.mean(axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)
    c_inv = linalg.pinvh(c)

    w = np.random.normal(0.0,1./np.sqrt(mx),size=(mx,my))
    h0 = np.random.normal(0.0,1./np.sqrt(mx),size=my)

    cost = np.full(nloop,100.)         
    for iloop in range(nloop):
        h = h0[np.newaxis,:] + x.dot(w)

        p = np.exp(h)
        
        # normalize
        p_sum = p.sum(axis=1)       
        p /= p_sum[:,np.newaxis]        
        h = np.log(p)
        
        #p2 = p_sum[:,np.newaxis] - p
        p2 = 1. - p
        h2 = np.log(p2)

        hh2 = h-h2
        model_ex = np.tanh(hh2/2)

        cost[iloop] = ((y2 - model_ex)**2).mean()
        if iloop > 0 and cost[iloop] >= cost[iloop-1]: break
        #print(cost[iloop])

        t = hh2 !=0    
        h[t] = h2[t] + y2[t]*hh2[t]/model_ex[t]
        h[~t] = h2[~t] + y2[~t]*2

        h_av = h.mean(axis=0)
        dh = h - h_av

        dhdx = dh[:,np.newaxis,:]*dx[:,:,np.newaxis]
        dhdx_av = dhdx.mean(axis=0)
        w = c_inv.dot(dhdx_av)            
        w -= w.mean(axis=0) 

        # 2019.03.29: ignore small w
        #for i in range(my):
        #    j = np.abs(w[:,i]) < fraction*np.mean(np.abs(w[:,i]))    
        #    w[j,i] = 0.

        h0 = h_av - x_av.dot(w)
        h0 -= h0.mean()
 
    return w,h0,cost,iloop

#=========================================================================================
# 2019.05.15: add ridge regression term to coupling w
def fit_multiplicative_ridge(x,y,nloop=10,lamda=0.1):
    mx = x.shape[1]
    my = y.shape[1]

    y2 = 2*y-1

    x_av = x.mean(axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)

    # 2019.05.15: ridge regression
    c += lamda*np.identity(mx)
    c_inv = linalg.pinvh(c)

    w = np.random.normal(0.0,1./np.sqrt(mx),size=(mx,my))
    h0 = np.random.normal(0.0,1./np.sqrt(mx),size=my)

    cost = np.full(nloop,100.)         
    for iloop in range(nloop):
        h = h0[np.newaxis,:] + x.dot(w)

        p = np.exp(h)
        
        # normalize
        p_sum = p.sum(axis=1)       
        p /= p_sum[:,np.newaxis]        
        h = np.log(p)
        
        #p2 = p_sum[:,np.newaxis] - p
        p2 = 1. - p
        h2 = np.log(p2)

        hh2 = h-h2
        model_ex = np.tanh(hh2/2)

        cost[iloop] = ((y2 - model_ex)**2).mean()
        if iloop > 0 and cost[iloop] >= cost[iloop-1]: break
        #print(cost[iloop])

        t = hh2 !=0    
        h[t] = h2[t] + y2[t]*hh2[t]/model_ex[t]
        h[~t] = h2[~t] + y2[~t]*2

        h_av = h.mean(axis=0)
        dh = h - h_av

        dhdx = dh[:,np.newaxis,:]*dx[:,:,np.newaxis]
        dhdx_av = dhdx.mean(axis=0)
        w = c_inv.dot(dhdx_av)            

        w -= w.mean(axis=0) 

        h0 = h_av - x_av.dot(w)
        h0 -= h0.mean()
 
    return w,h0,cost,iloop
#===================================================================================================
def dca(s0,theta=0.2,pseudo_weight=0.5):
#input: s0[L,n] (integer values, not one-hot)
#theta: threshold for finding similarity of sequences
#pseudo_weight = lamda/(lamda + pseudo_weight)
#output: w[mx_cumsum,mx_cumsum] coupling matrix ; di[n,n]: direct information

    n = s0.shape[1]
    mx = np.array([len(np.unique(s0[:,i])) for i in range(n)])

    mx_cumsum = np.insert(mx.cumsum(),0,0)
    i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T
      
    # hamming distance
    dst = distance.squareform(distance.pdist(s0, 'hamming'))
    ma = (dst <= theta).sum(axis=1).astype(float)
    Meff = (1/ma).sum()

    # convert to onehot
    onehot_encoder = OneHotEncoder(sparse=False)
    s = onehot_encoder.fit_transform(s0)
    
    fi_true = (s/ma[:,np.newaxis]).sum(axis=0)
    fi_true /= Meff

    fij_true = (s[:,:,np.newaxis]*s[:,np.newaxis,:]/ma[:,np.newaxis,np.newaxis]).sum(axis=0)
    fij_true /= Meff

    # add pseudo_weight
    fi = (1 - pseudo_weight)*fi_true + pseudo_weight/mx[0]        ## q = mx[0]
    fij = (1 - pseudo_weight)*fij_true + pseudo_weight/(mx[0]**2) ## q = mx[0]

    cw = fij - fi[:,np.newaxis]*fi[np.newaxis,:]
    cw_inv = -linalg.pinvh(cw)

    # set self-interations to be zeros
    for i0 in range(n):
        i1,i2 = i1i2[i0,0],i1i2[i0,1]
        cw_inv[i1:i2,i1:i2] = 0. 

    # normalize w
    w = cw_inv.copy()
    for i0 in range(n):
        i1,i2 = i1i2[i0,0],i1i2[i0,1]
        w[:,i1:i2] -= w[:,i1:i2].mean(axis=1)[:,np.newaxis]
        w[i1:i2,:] -= w[i1:i2,:].mean(axis=0)[np.newaxis,:]

    #----------------------------------------------------------------------------------------------- 
    # calculate direct information    
    ew_all = np.exp(w)
    di = np.zeros((n,n))

    tiny = 10**(-100.)
    diff_thres = 10**(-4.)

    for i0 in range(n-1):
        i1,i2 = i1i2[i0,0],i1i2[i0,1]
        fi0 = fi[i1:i2]

        for j0 in range(i0+1,n):
            j1,j2 = i1i2[j0,0],i1i2[j0,1]
            fj0 = fi[j1:j2]

            # fit h1[A] and h2[B] (eh = np.exp(h), ew = np.exp(w)) 
            ew = ew_all[i1:i2,j1:j2]

            diff = diff_thres + 1.
            # initial value
            eh1 = np.full(mx[i0],1./mx[i0])
            eh2 = np.full(mx[i0],1./mx[i0])
            for iloop in range(100):
                eh_ew1 = eh2.dot(ew.T)
                eh_ew2 = eh1.dot(ew)

                eh1_new = fi0/eh_ew1
                eh1_new /= eh1_new.sum()

                eh2_new = fi0/eh_ew2
                eh2_new /= eh2_new.sum()

                diff = max(np.max(np.abs(eh1_new - eh1)),np.max(np.abs(eh2_new - eh2)))

                eh1,eh2 = eh1_new,eh2_new    
                if diff < diff_thres: break        

            # direct information
            pdir = ew*((eh1.T).dot(eh2))
            pdir /= pdir.sum() 
            di[i0,j0] = (pdir*np.log(pdir+tiny/np.outer(fi0+tiny,fj0+tiny))).sum()
    
    # fill the lower triangular part
    di = di+di.T
    
    return w,di                                 
