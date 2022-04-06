import numpy as np
import pywt 
from scipy import optimize
import matplotlib.pyplot as plt


def create_decomp_p(data,scales,level):
   
            
    data = np.array(data)

    
    
    j = scales[level]
  
    
    coef,freqs = pywt.cwt(data,j,'gaus2',method='conv')
    #coef,ssqscales = cwt(data,wavelet='cmhat',scales=j)
    
    return coef

def refl_tanh(x,a,b,c,d,e,f):
    xn = b * x + c 
    return a * ((np.exp(e * xn) - np.exp(-f * xn)) / (np.exp(xn) + np.exp(-xn))) + d 

def optimize_tanh(wl,ref,signal,scales,w0=675,w1=800):

    def diff_func(params,*args):
        level = args[0]
        R = fitfunc(wlnew,*params)
        diff = create_decomp_p(np.multiply(ref,R)-signal,scales,level)
        res = np.sum(np.square(diff))
        return res


    
    startind = np.argmin(np.fabs(wl-w0))
    endind = np.argmin(np.fabs(wl-w1))

    appref = np.divide(signal,ref)
    wlnew,signal,ref, appref = wl[startind:endind],signal[startind:endind],ref[startind:endind], appref[startind:endind]

    fitfunc = refl_tanh
    p_init = [0.5,1,-730,0.2,1,1]
    testfitparams,_ = optimize.curve_fit(fitfunc,wlnew,appref,p0=p_init)    
    results = []
    for i in range(len(scales)):
        result_nm = optimize.minimize(diff_func,testfitparams,args=(i))
        resfunc = fitfunc(wl,*result_nm.x)
        if np.isfinite(resfunc).all() == True:
            results.append(resfunc)
    results = np.array(results)
    
    return np.nanmean(results,axis=0)