import numpy as np 


def weighted_std(values, weights,axis):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights,axis=axis)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights,axis=axis)
    return np.sqrt(variance)

def evaluate_sif(wl,spec,specialwl):
    totmin = np.argmin(np.fabs(wl-650))
    totmax = np.argmin(np.fabs(wl-800))
    Ftotal = np.sum(spec[totmin:totmax])*(wl[1]-wl[0])
    rel_wls = [687,760,684,735,specialwl]
    flags = np.zeros(len(rel_wls))
    argmins = np.zeros(len(rel_wls),dtype=int)
    for j,w in enumerate(rel_wls):
        if w > wl[0] and w < wl[-1]:
            flags[j] = 1
            argmins[j] = (np.argmin(np.fabs(wl-w)))

    if flags[0] == 1:
        F687 = spec[argmins[0]]
    else:
        F687 = np.nan

    if flags[1] == 1:
        F760 = spec[argmins[1]]
    else:
        F760 = np.nan
    if flags[4] == 1:
        spec_val = spec[argmins[1]]
    else:
        spec_val = np.nan

    if flags[2] == 1:
        if argmins[2]-20 < 0 or argmins[2]+20 > len(wl):
            Fr = spec[argmins[2]]
            wlFr = rel_wls[2]
        else:

            Fr_ind = np.argmax(spec[argmins[2]-20:argmins[2]+20])
            Fr_ind = Fr_ind + argmins[2]-20
            
            Fr = spec[Fr_ind]
            wlFr = wl[Fr_ind]
            if Fr_ind == argmins[2]-20 or Fr_ind == argmins[2]+20:
                print('Did not find red peak!')
                Fr = spec[argmins[2]]
        
    else:
        Fr = np.nan
        wlFr = np.nan

    if flags[3] == 1:
        if argmins[3]-20 < 0 or argmins[3]+20 > len(wl):
            Ffr = spec[argmins[3]]
            wlFfr = rel_wls[3]
        else:

            Ffr_ind = np.argmax(spec[argmins[3]-20:argmins[3]+20])
            Ffr_ind = Ffr_ind + argmins[3]-20
            
            Ffr = spec[Ffr_ind]
            wlFfr = wl[Ffr_ind]
            if Ffr_ind == argmins[3]-20 or Ffr_ind == argmins[3]+20:
                print('Did not find far red peak!')
                Ffr = spec[argmins[3]]
    else:
        Ffr = np.nan
        wlFfr = np.nan

    
    

    return Ftotal,F687,F760,Fr,wlFr,Ffr,wlFfr,spec_val

def calc_rsquared(y,y_m):
    """
    arguments:
    y - measured data points
    y_m - model to compare with

    returns: 
    R squared
    """

    sstot = np.sum(np.square(y-np.mean(y)))
    ssres = np.sum(np.square(y-y_m))
    return 1 - ssres / sstot

def calc_rmse(y,y_m):
    return np.sqrt(np.mean(np.square(y-y_m)))

def fitfct_linear(x,a,b):
    return a*x + b