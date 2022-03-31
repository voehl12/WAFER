from cgi import test
from statistics import mean
from zipapp import create_archive
import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.function_base import _msort_dispatcher
from scipy.interpolate.fitpack2 import UnivariateSpline 
import xarray as xr
from scipy.interpolate import interp1d
from hyplant_functions import get_wavelengths
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
import os
from ssqueezepy import ssq_cwt, ssq_stft,cwt,icwt,issq_cwt
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.special import gamma, factorial
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
from scipy.optimize import NonlinearConstraint
from scipy.optimize import least_squares
from scipy.optimize import fsolve
import warnings
from matplotlib import cm
from matplotlib.offsetbox import AnchoredText
import pywt
from scipy import interpolate
import pprocess
import time
from matplotlib import rc
from scipy.fft import fft, ifft, fftfreq
from collections import Counter
from datetime import datetime
import matplotlib.patches as patches
from itertools import permutations 
from matplotlib import rc
#plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

def plot_powerspectrum(wl,spectrum,scales,wlscales,signal,fname):
    figF,axF = plt.subplots(figsize=(8,6))
    normalizedsensorspec = [spectrum[i]/scales[i]**0.5 for i in range(len(spectrum))]
    normalizedsensorspec = np.array(normalizedsensorspec)
    plotspec = normalizedsensorspec[:-10]
    pF = axF.pcolor(wl,np.arange(len(plotspec)),plotspec,cmap='RdBu',shading='auto',vmin=-1,vmax=1)
    colF = figF.colorbar(pF,ax=axF,orientation='horizontal')
    ax2 = axF.twinx()
    ax2.pcolor(wl,np.arange(len(plotspec)),plotspec,cmap='RdBu',shading='auto',vmin=-1,vmax=1)
    ax2.set_yticks(axF.get_yticks())
    ax2.set_ylim(axF.get_ylim())
    figF.canvas.draw()
    axF.plot(wl,signal/np.max(signal)*30,color='black',linewidth=3)
    labels = [int(i.get_position()[1]) for i in ax2.get_yticklabels()]
    print(labels)
    labels_new = np.zeros(len(labels))
    for j,i in enumerate(labels):
        if i >= 0:
            labels_new[j] = np.round(wlscales[i],2)

    ax2.set_yticklabels(labels_new)
    ax2.set_ylabel(r'peak width [nm]')
    #rect = patches.Rectangle((wl[1], 29), wl[-2]-wl[1], 1, linewidth=1, edgecolor='r', facecolor='none')
    #axF.add_patch(rect)
    figname = fname
    axF.set_xlabel(r'$\lambda$ [nm]')
    axF.set_ylabel(r'level')
    figF.savefig(figname)


def icwavelet(wave,N,jmin,jmax,N_level,minrec=0,maxrec=0):
    scale = np.logspace(jmin,jmax,num=N_level,base=2.0)
    xdelta = np.zeros(N)
    xdelta[0] = 1
    Wdelta = create_decomp_p(xdelta,jmin,jmax,N_level)
    resdelta = np.zeros((len(Wdelta),N))
    for j in range(len(Wdelta)):
        resdelta[j] = np.divide(Wdelta[j],(scale[j])**0.5)#np.divide(Wdelta[j],(scale[j])**(1/2))
    recdelta = np.zeros(N)
    for i in range(N):
        recdelta = np.sum(resdelta,axis=0)
    cdelta = recdelta[0]
   
    dj = (jmax-jmin) / N_level
 
    coeff = 1.0 / (cdelta)
    
    oup = np.zeros(N)
    
    result = np.zeros((len(wave),N))
    jtot = len(wave)
    if jmax >= jtot:
        
        jmax = jtot
    for j in range(len(wave)):
        result[j] = wave[j]/scale[j]**0.5
    if maxrec == 0:
        for i in range(N):
            oup[i] = np.sum(result[minrec:,i]) 
    else:
        for i in range(N):
            oup[i] = np.sum(result[minrec:-maxrec,i]) 

    oup *= coeff
    return oup

def create_decomp_p(datapath,jmin,jmax,N_level,selscales='all'):
    data = []

    
    try:
        with open(datapath,'r') as d:
            for line in d:
                line = line.split()
                data.append(float(line[0]))
    except:
        data = datapath
    
    data = np.array(data)

    if selscales == 'all':
        j = np.logspace(jmin,jmax,num=N_level,base=2)
    else:
        j = np.array(np.logspace(jmin,jmax,num=N_level,base=2)[selscales])
    f = (pywt.scale2frequency('gaus2',j))**-1
    wav = pywt.ContinuousWavelet('gaus2')
    
    #coef,freqs = pywt.cwt(data,j,'gaus2',method='conv')
    coef,ssqscales = cwt(data,wavelet='cmhat',scales=j)

    return np.real(coef)

def push_down(wl,specdecomp,refdecomp,initknots,levels,apparentrefl,whitereference,scales):
    # initknots is an array of wavelengths (first row) and strengths (second row) of knots from initial guess fluorescence (zeroline)
    def diff_func(points,level,imcount,rangewl=np.arange(len(wl))):
        interp = interpolate.UnivariateSpline(wl_points,points,s=0)
        truetrans = apparentrefl - interp(wl)/whitereference
        #diff = np.fabs(np.multiply(refdecomp[level],truetrans) - specdecomp[level])
        diff = np.fabs(np.fabs(np.multiply(refdecomp[level],truetrans)) - np.fabs(specdecomp[level]))
        
        try:
            result = np.mean(diff[rangewl])/(scales[level])**0.5
        except: 
            if rangewl[0] > 500:
                result = np.mean(diff[:-30])/(scales[level])**0.5
            elif rangewl[0] < 500:
                result = np.mean(diff[:30])/(scales[level])**0.5
            else:
                print('Careful, could not find range for error calculation')
        return result
    
    # stepsize for sif-knot-shifting and maximal sif in mW
    fdelta = 0.1
    fmax = 10
    wmax = fmax/fdelta # maximum weights
    

    
    
    weights = np.ones((len(levels),len(initknots[0])-1)) # how many deltas is each knot pushed up? starting value is one.
    wl_points = initknots[0]
    inds = [np.argmin(np.fabs(wl-wl_points[i])) for i in range(len(wl_points))]
    knots = np.array(initknots[1])
    errors = []
    retlevels = []
    levelerrors = []
    acceptlevels = []
    knots = np.array(initknots[1])
    for k in range(len(levels)):
        imcount = 0
        levelerror = []
        level = levels[k]
        
        error = diff_func(knots,level,imcount)
        imcount += 1
        lasterror = error + 1
        
        
        """  while error < lasterror:
            lastdiff = diff
            lasterror = error
            
            knots[1:] -= deltas
            weights[k] += 1
            error = diff_func(knots,level,imcount) 
            diff = np.fabs(lasterror-error) 
            imcount += 1 """
            
        for i in range(len(knots)-1):
            error = diff_func(knots,level,imcount)#np.arange(inds[-(i+1)]-15,inds[-(i+1)]+15))
            imcount += 1
            lasterror = error + 1
                        
            while error < lasterror:
                if weights[k,-(i+1)] >= wmax:
                    break
                
                lasterror = error
                
                knots[-(i+1)] = knots[-(i+1)] + fdelta
                weights[k,-(i+1)] += 1
                
                error = diff_func(knots,level,imcount)#,np.arange(inds[-(i+1)]-15,inds[-(i+1)]+15))
                imcount += 1
                
                if error > lasterror:
                    knots[-(i+1)] = knots[-(i+1)] - fdelta
                    weights[k,-(i+1)] -= 1
                    levelerror.append(lasterror)
                    errors.append(lasterror)
                    retlevels.append(k)
                    break
                
                errors.append(error)
                levelerror.append(error)
                retlevels.append(k)
        minlevelerror = np.min(np.array(levelerror))
        
        if minlevelerror < 0.05 and minlevelerror > 0.0:
            acceptlevels.append(k)
        levelerrors.append(np.mean(np.array(levelerror)))
            
    
    
    meanweight = np.median(weights[acceptlevels],axis=0)
    meanweight = np.insert(meanweight,0,0.0)
    
    knots = initknots[1] + meanweight*fdelta
    return wl_points,knots

def prepare_arrays(cab,lai,feffef):
    completename = 'cwavelets/libradtranscope/floxseries_ae_oen/reflectance/radcomplete_{}_{:d}_{:d}_ae_conv.dat'.format(feffef,cab,lai)
    woFname = 'cwavelets/libradtranscope/floxseries_ae_oen/reflectance/radwoF_{}_{:d}_{:d}_ae_conv.dat'.format(feffef,cab,lai)
    reflname = 'reflectance/szamatch/rho_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    albedoname = 'reflectance/szamatch/albedo_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    scoperef = 'LupSCOPE/szamatch/Lup_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    Fname = 'cwavelets/libradtranscope/floxseries_ae_oen/reflectance/Fcomp_{}_{:d}_{:d}_ae.dat'.format(feffef,cab,lai) #'fluorescence/F_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    
    wlRname = 'reflectance/szamatch/wlR'
    wlFname = 'reflectance/szamatch/wlF'
    filenames = [completename,woFname,reflname,albedoname,Fname,wlRname,wlFname,scoperef]
    array_list = []
    for name in filenames:
        arr = []
        with open(name,'r') as f:
            for line in f:
                line = line.split()
                if len(line) > 1:
                    arr.append(float(line[1]))
                else: 
                    arr.append(float(line[0]))
        arr = np.array(arr)
        array_list.append(arr)
    return array_list

def prepare_arrays_hp():
    completename = 'cwavelets/Hyplant/spectrum_SIF_high.dat'
    
    
    filenames = [completename]
    array_list = []
    for name in filenames:
        arr = []
        with open(name,'r') as f:
            for line in f:
                line = line.split()
                if len(line) > 1:
                    arr.append(float(line[1]))
                else: 
                    arr.append(float(line[0]))
        arr = np.array(arr)
        array_list.append(arr)
    return array_list

def spec_respons(wl,rawdata,wlmin,wlmax,Nsamples,fwhm):
    wlfine = np.linspace(wl[0],wl[-1],num=100000)
    finessi = wlfine[1]-wlfine[0]
    intrad = interp1d(wl,rawdata,kind='nearest')
    finerad = intrad(wlfine)
    samples = np.linspace(wlmin,wlmax,num=Nsamples)
    pixelfwhm = fwhm/finessi
    sigma = pixelfwhm/2.355
    gauss = signal.gaussian(int(10*sigma),sigma)
    convolution = signal.convolve(finerad,gauss,mode='same') / np.sum(gauss)

    sampleinds = [np.argmin(np.fabs(wlfine-samples[i])) for i in range(len(samples))]
    convsignal = convolution[sampleinds]
    
    return samples,convsignal,gauss

def match_solspec(wl,fwhm,path='../Data/Sun/SUN001kurucz.dat'):
    swl = []
    srad = []
    with open(path,'r') as f:
        for line in f:
            line = line.split()
            try:
                freq = float(line[0])
                if freq == 0.0:
                    continue
            except:
                continue
            
            w = freq
            if w >= 200 and w <= 1000:
                swl.append(w)
                srad.append(float(line[1]) * 1000 / np.pi)
                # converted to mW/(m^2 nm ster)
            else:
                continue

    swl, srad = np.array(swl), np.array(srad)
    swl, sradnew, gaussf= spec_respons(swl,srad,wl[0],wl[-1],len(wl),fwhm)
    return sradnew,gaussf

# set wavelength

def level_selection(s,r,wl,wlscales):
    tarray_alllevel = np.zeros((len(s),len(wl)))
    difftoappstd = []
    levels = []
    
    fig, ax1 = plt.subplots()
    my_cmap = cm.get_cmap('jet', 25)
    print('Transmittance prediction accuracy and level selection...')
    for l in range(len(s)):
        points = []
        #sunt = 0.05*np.max(s[l])
        peakss, _ = find_peaks(-s[l])
        
        peaksr, _ = find_peaks(-r[l],height=0.05*np.max(-r[l]))
        tarray = np.zeros(len(wl))
        
        for i,pn in enumerate(peaksr):
            
            if pn in peakss:
                t = s[l,pn]/r[l,pn]
                tarray[pn] = t
                points.append(pn)
        tarray = np.array(tarray)
        levels.append(l)
        tarray_masked = np.ma.masked_where(tarray == 0.0, tarray)
        if len(tarray_masked.compressed()) > 50 and np.logical_and(np.min(tarray) > -0.1,np.ma.mean(tarray_masked.compressed()[:10]) < 0.2):
            if np.max(tarray) < 0.7 and wlscales[l] > 0.3:
                tarray_alllevel[l] = tarray
                #levels.append(l)
                ax1.plot(wl,tarray,'.',label='Level {:d}'.format(l),color=my_cmap(l))
        #ax2.plot(wl[points],level_difftoapp,'.',label='Level {:d}'.format(l),color=my_cmap(l))
      
        
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Reflectance')
    ax1.set_ylim(0,np.max(tarray_alllevel)+0.05)
    ax1.legend(ncol=2,loc=2,fontsize=7)
    
    fig.savefig(newdir+'/flowchart4_'+fileext+'.pdf')
    
    tarray_masked = np.ma.masked_where(tarray_alllevel == 0.0, tarray_alllevel)
    tarray_means = np.ma.mean(tarray_masked,axis=0)
    tarray_means = np.array(tarray_means)
    
    tarray_means = np.nan_to_num(tarray_means)
    tarray = [tarray_means[i] for i in range(len(wl)) if tarray_means[i] != 0.0]
    parray = [i for i in range(len(wl)) if tarray_means[i] != 0.0]
    warray = [wl[i] for i in parray]
    
       
    tarray = np.array(tarray)
    warray = np.array(warray)
    parray = np.array(parray)
   
    return parray,warray,tarray,levels

class MyBounds:

    def __init__(self, x0):

        self.xmax = np.array([val-0.0003 for val in x0])

        self.xmin = np.array([val-0.05 for val in x0])

    def __call__(self, **kwargs):

        x = kwargs["x_new"]

        tmax = bool(np.all(x <= self.xmax))

        tmin = bool(np.all(x >= self.xmin))

        return tmax and tmin

def callback(x,f,accept):
    print(f,accept)
    if int(accept) == 1 and f < 0.005:
        return True

def refl_tanh(x,a,b,c,d,e,f):
    xn = b * x + c 
    return a * ((np.exp(e * xn) - np.exp(-f * xn)) / (np.exp(xn) + np.exp(-xn))) + d   

def refl_logistic(w,a,b,c,d):
        wl0 = wl[0]
        return a*w + b/(1+c*np.exp(-d*(w-wl0)))

def refl_polynomial(w,*coeffs):
    
    interp = np.poly1d(coeffs)
    return interp(w)



def optimize_logistic(wl,signal,whiteref,appref,retlevels,w0=675,w1=800):

    def diff_func_exp(params,*args):
        R = fitfunc(wl,*params)
        diff = ((create_decomp_p(np.multiply(whiteref,R)-signal,jmin,jmax,nlevel,args[0])))
        return np.sqrt(np.sum((np.square(diff))/scale[args[0]]))

    """ def diff_func_wavelet(coeffs):
        newspec[i] = [basespec[i]*coeffs[i] for i in adaptlevels]
        R = icwt(res_ref_decomp,wavelet='cmhat',scales=sifscales,x_mean=np.mean(reflspline(wlnew)),recmin=30)
     """
    
    startind = np.argmin(np.fabs(wl-w0))
    endind = np.argmin(np.fabs(wl-w1))
    wl,signal,whiteref,appref = wl[startind:endind],signal[startind:endind],whiteref[startind:endind],appref[startind:endind]
    
    Rshape = 'tanh'
    if Rshape == 'tanh':
        fitfunc = refl_tanh
        p_init = [0.5,1,-730,0.2,1,1]
    elif Rshape == 'logistic':
        fitfunc = refl_logistic
        ainit = (appref[20]-appref[0])/(wl[20]-wl[0])
        binit = appref[-1]-ainit*wl[-1]
        cinit = binit/(appref[0]-ainit*wl[0]) - 1
        p_init = [ainit,binit,cinit,0.0001]
    elif Rshape == 'poly':
        fitfunc = refl_polynomial
        p_init = np.polyfit(wl,appref,5)
        
    scale = np.logspace(jmin,jmax,num=nlevel,base=2.0)
    
    
    testfitparams,_ = optimize.curve_fit(fitfunc,wl,appref,p0=p_init)
    print(testfitparams)

    plt.figure()
    plt.plot(wl,appref,label='appref')
    plt.plot(wl,fitfunc(wl,*testfitparams),label='initial guess')
    results = []
    for c,i in enumerate(retlevels):
        result_nm = optimize.minimize(diff_func_exp,testfitparams,args=(i,0))
        resfunc = fitfunc(wl,*result_nm.x)
        if np.isfinite(resfunc).all() == True:
            results.append(result_nm.x)
        plt.plot(wl,fitfunc(wl,*result_nm.x),label=i)
    plt.plot(wl,R_expect[startind:endind],label='expected')
    plt.legend()
    plt.show()
    results = np.array(results)
    print(results)
    print(np.nanmean(results,axis=0))
    print(wl)
    return startind,wl,np.nanmean(results,axis=0)


def optimize_knots(wl,comm_knots,decomp_inp,refdecomp,whiteref,appref,retlevels,signal):

    reslist = []
    scale = np.logspace(jmin,jmax,num=nlevel,base=2.0)
    exis_knots = []
    wl_points_arr = []
    coeffmeans = []
    for i in range(len(refdecomp)):
        
        peakinds,_ = find_peaks(-decomp_inp[i])
        peakpos = wl[peakinds]
        #peakvals = []
        knotinds = []
        for knot in comm_knots:
            closepeakind = np.argmin(np.fabs(knot-peakpos))
            
                
            totpeakind = peakinds[closepeakind]
            if np.fabs(wl[totpeakind] - knot) > 0.5*dwl:
                if knot == wl[-1]:
                    knotinds.append(np.argmin(np.fabs(wl-knot))-1)
                    #peakvals.append(decomp_inp[i,np.argmin(np.fabs(wl-knot))-1])
                elif knot == wl[0]:
                    knotinds.append(np.argmin(np.fabs(wl-knot))+1)
                    #peakvals.append(decomp_inp[i,np.argmin(np.fabs(wl-knot))+1])

                else:
                    knotinds.append(np.argmin(np.fabs(wl-knot)))
                    #peakvals.append(decomp_inp[i,np.argmin(np.fabs(wl-knot))])
                    
            elif totpeakind == len(wl)-1:
                knotinds.append(np.argmin(np.fabs(wl-knot)))
                #peakvals.append(decomp_inp[i,np.argmin(np.fabs(wl-knot))])
                print('closest peak was end of spectrum')

            

            else:
                #peakval = decomp_inp[i,totpeakind]
                knotinds.append(totpeakind)
                #peakvals.append(peakval)
        #knotinds.append(0)
        #knotinds.append(len(wl)-1)
        knotindsnew = knotinds.copy()
        notremoved = []
        
        for knotnum,val in enumerate(knotinds):
            
            if knotindsnew.count(val) == 1:
                notremoved.append(knotnum)
                
            else:
                
                knotindsnew.remove(val)
                
                
                
        
        exis_knots.append(notremoved)
        peakvals = decomp_inp[i,knotindsnew]
        wl_points_arr.append(knotindsnew)
        coeffmeans.append(np.fabs(peakvals)/np.sum(np.fabs(decomp_inp[i])))
        
        #coeffmeans[i,-2] = np.fabs(decomp_inp[i,0]/np.sum(np.fabs(decomp_inp[i])))
        #coeffmeans[i,-1] = np.fabs(decomp_inp[i,len(wl)-1]/np.sum(np.fabs(decomp_inp[i])))
    """ normalizedsensorspec = [refdecomp[i]/scale[i] for i in range(len(refdecomp))]
    normalizedsensorspec = np.array(normalizedsensorspec)

    
    for level in retlevels:
        levelmeans = np.zeros(len(wl_points_arr[0]))
        for i,knot in enumerate(wl_points_arr[level]):
            
            knotind = knot
            
            if len(normalizedsensorspec[level,knotind-30:knotind+30]) == 0:
                levelmeans[i] = 0
            else:
                meanpower = np.mean(np.fabs(normalizedsensorspec[level,knotind-30:knotind+30]))
                levelmeans[i] = meanpower
        coeffmeans[level] = levelmeans"""
    #coeffmeans = np.array(coeffmeans)
    exis_knots = np.array(exis_knots)
    
    weights = np.square(np.argsort(coeffmeans,axis=0))
    plt.figure()
    meancolors = cm.get_cmap('viridis', len(coeffmeans))
    for i in range(len(coeffmeans)):
        plt.plot(wl[wl_points_arr[i]],coeffmeans[i],'.',label=r'${:.2f}$ nm'.format(wlscales[i]),color=meancolors(i))
    plt.ylabel('Mean Normalized Coefficient Strength')
    plt.xlabel('Wavelength [nm]')
    plt.legend(ncol=2,loc=2,fontsize=7)
    plt.savefig(newdir+'/flowchart5b_'+fileext+'.pdf')

    """ normalizedsensorspec = [decomp_inp[i]/scale[i] for i in range(len(refdecomp))]
    normalizedsensorspec = np.array(normalizedsensorspec)

    coeffmeans = np.zeros((len(normalizedsensorspec),len(wl_points_arr[0])))
    for level in retlevels:
        levelmeans = np.zeros(len(wl_points_arr[0]))
        for i,knot in enumerate(wl_points_arr[level]):
            
            knotind = knot
            
            if len(normalizedsensorspec[level,knotind-30:knotind+30]) == 0:
                levelmeans[i] = 0
            else:
                meanpower = np.mean(np.fabs(normalizedsensorspec[level,knotind-30:knotind+30]))
                levelmeans[i] = meanpower
        coeffmeans[level] = levelmeans
    coeffmeans = np.array(coeffmeans)
    
    weights = np.square(coeffmeans)
    #weights = np.square(np.argsort(coeffmeans,axis=0))
    plt.figure()
    meancolors = cm.get_cmap('viridis', len(weights))
    for i in range(len(coeffmeans)):
        plt.plot(wl[wl_points_arr[i]],coeffmeans[i,:],'.',label='Level {:d}'.format(i),color=meancolors(i))
    plt.ylabel('Mean Normalized Coefficient Strength Signal')
    plt.xlabel('Wavelength [nm]')
    plt.legend(ncol=2,loc=2,fontsize=7)
    plt.savefig(newdir+'/coefficientstrengthsignal_'+fileext+'.pdf') """
    

    def diff_func(points,*args):
        """ ainit = (appref[20]-appref[0])/(wl[20]-wl[0])
        binit = appref[-1]-ainit*wl[-1]
        cinit = binit/(appref[0]-ainit*wl[0]) - 1
        p_init = [ainit,binit,cinit,0.0001]
        testfitparams,_ = optimize.curve_fit(refl_logistic,wl_points,points,p0=p_init) """
        #interp = interpolate.UnivariateSpline(wl_points,points,s=0)
        interppoints = np.append(points,apparentrefl[-1])
        interppoints = np.insert(interppoints,0,apparentrefl[0])
        interp = interp1d(wl_points,interppoints,kind='linear')
        #interp = interpolate.Akima1DInterpolator(wl_points,points)
        #pointins = np.array([np.argmin(np.fabs(wl-wl_points[i])) for i in range(len(points))])
        #polyfitcoeffs = np.polyfit(wl_points,points,5)
        #interp = np.poly1d(polyfitcoeffs)
        #diff = np.fabs(np.multiply(whitedecomp[args[0]],interp(wl)) - decomp_inp[args[0]])
        #diff = (np.fabs(create_decomp_p(np.multiply(whiteref,interp(wl)),jmin,jmax,nlevel,args[0]) - decomp_inp[args[0]]))
        diff = ((create_decomp_p(np.multiply(whiteref[wl_points_arr[args[0]][0]:wl_points_arr[args[0]][-1]],interp(wlnew))-signal[wl_points_arr[args[0]][0]:wl_points_arr[args[0]][-1]],jmin,jmax,nlevel,args[0])))
        #diff = (np.fabs(create_decomp_p(np.multiply(whiteref[comm_knots_inds[0]:comm_knots_inds[-1]],interp(wlnew))-signal[comm_knots_inds[0]:comm_knots_inds[-1]],jmin,jmax,nlevel,args[0])))
        levelres.append((np.sum((np.square(diff))/scale[args[0]])))
        #diff = np.fabs(np.multiply(whitedecomp,interp(wl))[level] - decomp_inp[level])
        finalfunc = interp(wlnew)
        #finalfunc = refl_logistic(wlnew,*testfitparams)

        
        reslist.append(finalfunc)
        return (np.sum((np.square(diff))/scale[args[0]]))
    
        


    
    shiftfig,shiftax = plt.subplots()
    reflfig, reflax = plt.subplots()
     
    
    
    shiftresults = []
    shiftresults_nm = []
    finaldev = []
    lens = [len(wl_points_arr[i]) for i in range(len(wl_points_arr))]
    shiftresults_arr = np.zeros((len(normalizedsensorspec),np.max(lens)))
    weights = np.zeros((len(normalizedsensorspec),np.max(lens)))
    my_cmap = cm.get_cmap('viridis', len(retlevels))
    wl_points_arr = np.array(wl_points_arr)

    comm_knots_inds = [np.argmin(np.fabs(wl-comm_knots[i])) for i in range(len(comm_knots))]
    comm_knots_inds = np.array(comm_knots_inds)
    allresiduals = []
    perfresiduals = []
    figknots,axknots = plt.subplots()
    init_points_fix = appref[comm_knots_inds]
    axknots.plot(wl,appref,label='Apparent Reflectance',color='tab:blue',alpha=0.5)
    axknots.plot(wl,interRa_smooth(wl),label='Smooth',color='tab:blue')
    axknots.plot(wl[comm_knots_inds],init_points_fix,'o',color='red',label='Initial knots')
    axknots.legend()
    
    
    
    
    figres,axres = plt.subplots()
    for c,i in enumerate(retlevels):
        levelres = []
        init_points = appref[wl_points_arr[i]]
        upperbound = apparentrefl[wl_points_arr[i]]
        #init_points = appref[comm_knots_inds]
        wl_points = wl[wl_points_arr[i]]
        axknots.plot(wl_points,init_points,'.',color=my_cmap(c),label=r'${:.2f}$ nm'.format(wlscales[i]))
        #wl_points = comm_knots
        wlnew = wl[wl_points_arr[i][0]:wl_points_arr[i][-1]]
        #wlnew = wl[comm_knots_inds[0]:comm_knots_inds[-1]]
        #appreflfit = interp1d(wl_points,init_points,kind='cubic')(wl)
        #appref_diff = np.mean(np.fabs(appreflfit-appref))
        #print(appref_diff)
        """ if appref_diff > 0.005:
            continue """
        
        parameterBounds = optimize.Bounds(np.array(upperbound-10/whiteref[wl_points_arr[i]])[1:-1],upperbound[1:-1])
        
        minimizer_kwargs = {"args": (i,[])}
        #minimizer_kwargs = [i]
        #result = least_squares(diff_func,init_points,method='trf',max_nfev=100,args=minimizer_kwargs)
        #result = basinhopping(diff_func,init_points,niter=500,minimizer_kwargs=minimizer_kwargs,callback=callback,accept_test=parameterBounds,seed=3,stepsize=-0.01)
        result_nm = optimize.minimize(diff_func,init_points[1:-1], bounds=parameterBounds, args=(i,0))#, xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=callback, initial_simplex=None)
        
        shiftresults_arr[i,0] = apparentrefl[0]
        shiftresults_arr[i,-1] = apparentrefl[-1]
        #shiftresults.append(result.x)
        shiftresults_arr[i,exis_knots[i,1:-1]] = result_nm.x 
        shiftresults_nm.append(result_nm.x)
        perfpoints = transferf[wl_points_arr[i]]
        perfspline = interp1d(wl_points,perfpoints,kind='quadratic')
        reflax.plot(wl,transferf)
        reflax.plot(wl_points,perfpoints,'o',color=my_cmap(c))
        reflax.plot(wlnew,perfspline(wlnew),color=my_cmap(c))
        
        #perfresiduals.append(diff_func(perfpoints,i))
        allresiduals.append(levelres)
        finaldev.append(result_nm.fun)
        its = len(allresiduals[c])
        """ axres.plot(np.arange(its),allresiduals[c],color=my_cmap(c),label=r'${:.2f}$ nm'.format(wlscales[i]))
        axres.plot(np.arange(its),perfresiduals[c]*np.ones(its),'--',color=my_cmap(c))
        axres.set_yscale('log') """
        shiftax.plot(wl_points[1:-1],result_nm.x,'.',label=r'${:.2f}$ nm'.format(wlscales[i]),color=my_cmap(c))
        weights[c,exis_knots[c]] = np.square(coeffmeans[c])
    shiftax.plot(wl,appref,label='Apparent Reflectance')
    shiftax.plot(wl,R_expect,label='True Reflectance')
    shiftax.set_xlabel('Wavelength [nm]')
    shiftax.set_ylabel('Reflectance')
    #plt.ylim(0.02,0.5)
    """ axres.set_ylabel('Residual')
    axres.set_xlabel('Function Calls')
    axres.legend() """
    shiftax.legend(ncol=2,loc=2,fontsize=7)
    shiftfig.savefig(newdir+'/flowchart5a_'+fileext+'.pdf')
    #figres.savefig(newdir+'/residuals_quadr.pdf')
    
    
    reslen = np.max(lens)
    
     
    shiftresults = np.array(shiftresults_nm)
    diffs = []
    
    argmin = np.argmin(finaldev)
    print('level with minimum residual: {:d}'.format(retlevels[argmin]))
    #meanresult = np.average(shiftresults,weights=weights,axis=0)
    meanresult = np.zeros(reslen)
    stdtvs = np.zeros(reslen)
    wl_final = np.zeros(reslen)
    wl_array = np.zeros((len(shiftresults_arr),np.max(lens)))
    for i in range(len(wl_array)):
        wl_array[i,exis_knots[i]] = wl[wl_points_arr[i]]
    for i in range(len(shiftresults_arr[0])):
        if np.sum(weights[:,i]) != 0:
            meanresult[i] = np.average(shiftresults_arr[:,i],weights = weights[:,i])
            #meanresult = np.average(shiftresults_arr,weights=weights,axis=0)
            stdtvs[i] = weighted_std(shiftresults_arr[:,i],weights[:,i],0)
            wl_final[i] = np.mean(wl_array,axis=0)[i]
        elif i == 0:
            wl_final[i] = wl[0]
        
            meanresult[i] = appref[0]
            stdtvs[i] = 0
    meanresult_nz_inds = np.nonzero(meanresult)
    meanresult_nz = meanresult[meanresult_nz_inds]
    wl_nz = wl_final[np.where(meanresult != 0,wl_final,0) != 0]
    stdvs_nz = stdtvs[np.where(meanresult != 0,stdtvs,100) != 100]
    
    
    return wl_nz,meanresult_nz,stdvs_nz,finaldev

def diff_func_single(wl_points,points,level):
        #all_points = np.insert(points,0,appref[0])
        #all_points = np.append(all_points,appref[-1])
        interp = interpolate.UnivariateSpline(wl_points,points,s=0)
        #interp = interp1d(wl_points,points,kind='cubic')
        #diff = np.fabs(np.multiply(whitedecomp[args[0]],interp(wl)) - decomp_inp[args[0]])
        diff = np.square(np.fabs(create_decomp_p(np.multiply(whitereference,interp(wl)),jmin,jmax,nlevel,level) - sensorspec[level]))
        #diff = np.fabs(np.multiply(whitedecomp,interp(wl))[level] - decomp_inp[level])
        finalfunc = interp(wl)
        
        return np.fabs(np.mean(diff))/scales[level]**0.5

def weighted_std(values, weights,axis):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights,axis=axis)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights,axis=axis)
    return np.sqrt(variance)

def FLOX_SpecFit_6C_funct(x,L0,wvl,sp,inp,knots):


    # --- RHO

    knotvals   = x[2:]
    interp = interpolate.UnivariateSpline(knots,knotvals,s=0)
    RHO          = interp(wvl)



    # --- FLUORESCENCE

    # -- FAR-RED
    u2       = (wvl - 735)/25.
    FFAR_RED = x[1]/(u2**2 + 1.)

    # -- RED
    u2       = (wvl - 684)/10.
    FRED     = x[0]/(u2**2 + 1.)

    # -- FULL SPECTRUM
    F        = np.multiply((FFAR_RED + FRED),(1.-(1.-RHO)))
    #F        = (FFAR_RED + FRED)

    # -- UPWARD RADIANCE
    y        = np.multiply(RHO,L0) + F
    return y-inp


def FLOX_SpecFit_6C(wvl, L0, LS, fsPeak, w, alg, oWVL):

    # Apparent Reflectance - ARHO -
    ARHO          = np.divide(LS,L0)
    
    # Excluding O2 absorption bands
    mask = np.logical_and(np.logical_or(wvl < 686, wvl > 692),np.logical_or(wvl < 758, wvl > 773))
    mask = np.array(mask)                                                                      # np.logical_or(wl < 688, wl > 705)     
    mask = mask > 0
    
    wvl = np.array(wvl)
    wvlnoABS      = wvl[mask]
    ARHOnoABS     = ARHO[mask]
    # knots vector for piecewise spline
    inds = np.linspace(1,len(wvlnoABS)-2,20-1+4,dtype=int) #[int(number) for number in np.linspace(1,len(wvlnoABS)-2,20-1+4)]
    knots         = wvlnoABS[inds]
    
    # piece wise cubic spline
    sp = interpolate.LSQUnivariateSpline(wvlnoABS,ARHOnoABS,knots)
    firsspline = sp(wvl)
    p_r           = sp(wvl)[inds]
    
    # --- FIRST GUESS VECTOR

    x0            = [fsPeak[0],fsPeak[1]]
    for val in p_r:
        x0.append(val)
    
    
    
    # --- WEIGHTING SCHEME

    LS_w          = np.multiply(LS,w)   # with weight
    # --- OPTIMIZATION

    if alg == 'trf':
        res = least_squares(FLOX_SpecFit_6C_funct,x0,method='trf',max_nfev=100,args=(L0,wvl,sp,LS_w,knots))     #lsqcurvefit(Fw,x0,L0,LS_w,[],[], options)
        x, resnorm,residual,exitflag,nfevas = res.x,res.cost,res.fun,res.status,res.nfev
        
        
        if exitflag == -1:
            resnorm = np.NaN

    elif alg == 'lm':
        res = least_squares(FLOX_SpecFit_6C_funct,x0,method='lm',max_nfev=6,args=(L0,wvl,sp,LS_w,knots))     #lsqcurvefit(Fw,x0,L0,LS_w,[],[], options)
        x, resnorm,residual,exitflag,nfevas = res.x,res.cost,res.fun,res.status, res.nfev

    else:
        print('Check Optimization algorithm')
        exit(0)
          
    

    # --- OUTPUT SPECTRA

    # -- Reflectance 
    knotvals   = x[2:]
    interp = interpolate.UnivariateSpline(knots,knotvals,s=0)
    RHO          = interp(wvl)
    r_wvl        = RHO


    # -- Sun-Induced Fluorescence
    # - FAR-RED 
    u2       = (oWVL - 735)/25.
    FFAR_RED = x[1]/(u2**2 + 1.)

    # - RED 
    u2       = (oWVL - 684)/10.
    FRED     = x[0]/(u2**2 + 1.)

    # - FULL SPECTRUM
    f_wvl    = np.multiply((FFAR_RED + FRED),(1-(1-RHO)))
    #f_wvl    = (FFAR_RED + FRED)


    # -- At-sensor modeled radiance
    LSmod        = FLOX_SpecFit_6C_funct(x,L0,wvl,sp,LS_w,knots)


    # --  RETRIEVAL STATS
    residual     = LSmod - LS
    rmse         = np.sqrt(np.sum(np.square(LSmod - LS))/len(LS))
    rrmse        = np.sqrt(np.sum(np.square(np.divide((LSmod - LS),LS)))/len(LS))*100. 






    return x, f_wvl, r_wvl, resnorm, exitflag, nfevas

#def fld():

def add_noise(data,snr,switch):
    sigmas = []
    if switch == 0:
        sigmas = np.zeros(len(data))
        return sigmas,data
    else:
        noisydata = np.zeros((10,len(data)))
        for j in range(10):

            s0 = 0
            newdata = np.zeros(len(data))
            #SNR = []
            for i,val in enumerate(data):
                sigma = val/snr
                #sigma = 0.3
                #sigma = 1.0
                #snr = val/sigma
                s0 = np.random.normal(0.0, sigma)
                #SNR.append(snr)
                sigmas.append(s0)
                newval = val + s0
                newdata[i] = newval
            noisydata[j] = newdata
        newdata = np.mean(noisydata,axis=0)
        return sigmas,newdata

def adjust_smoothing(wl,refl,initsmoothing,minknots,maxknots):
    smoothing = initsmoothing
    interRa = interpolate.UnivariateSpline(wl,refl,s=smoothing,k=1)
    initknots = interRa.get_knots()
    
    if len(initknots) < minknots:
        while len(interRa.get_knots()) < minknots:
            step = smoothing / 10
            smoothing -= step
            interRa = interpolate.UnivariateSpline(wl,refl,s=smoothing,k=1)
    
    elif len(initknots) > maxknots:
        while len(interRa.get_knots()) > maxknots:
            step = smoothing / 10
            smoothing += step
            interRa = interpolate.UnivariateSpline(wl,refl,s=smoothing,k=1)

    #print('Final Smoothing: {:.7f}'.format(smoothing))
    return interRa,smoothing

def evaluate_sif(wl,spec):
    dt = wl[3]-wl[2]
    totmin = np.argmin(np.fabs(wl-670))
    totmax = np.argmin(np.fabs(wl-800))
    Ftotal = np.sum(spec[totmin:totmax])*dt
    ind_687 = np.argmin(np.fabs(wl-687))
    ind_760 = np.argmin(np.fabs(wl-760))
    F687 = spec[ind_687]
    F760 = spec[ind_760]
    ind_684 = np.argmin(np.fabs(wl-684))
    ind_735 = np.argmin(np.fabs(wl-735))
    Fr_ind = np.argmax(spec[ind_684-50:ind_684+60])
    Fr_ind = Fr_ind + ind_684-50
    Ffr_ind = np.argmax(spec[ind_735-50:ind_735+60])
    Ffr_ind = Ffr_ind + ind_735-50
    Fr = spec[Fr_ind]
    wlFr = wl[Fr_ind]
    Ffr = np.max(spec)
    wlFfr = wl[np.argmax(spec)]
    if Fr_ind == ind_684-50 or Fr_ind == ind_684+59:
        print('Did not find red peak!')
        Fr = 0.0
    
    if Ffr_ind == ind_735-50 or Ffr_ind == ind_735+60:
        print('Did not find far-red peak!')
        Ffr = 0.0

    return Ftotal,F687,F760,Fr,wlFr,Ffr,wlFfr

def fsolve_solver(sig,ref,init,scale):
    def ffunc(r):
        integ = sig-np.multiply(r,ref)
        coefs = create_decomp_p(integ,jmin,jmax,nlevel,scale)
        return coefs[0]

    rsolve = fsolve(ffunc,init)

    return rsolve
        




newdir = str(datetime.now().date())
os.system('mkdir {}'.format(newdir))


resultfilename = newdir+'/flox_proposal_oensmodel_quadrmean_1015_nonoise'
sifresults = newdir+'/SIF_spectralshapes_oensmodel_quadrmean_1015_nonoise'
specificsfile = newdir+'/parameters'

testdata = 'scope'
noise = 0

if testdata == 'scope':
    # SCOPE series:
    sifeffec = '004'
    #CAB = []
    """ CAB = [int(5),int(10)]
    for i in range(2,9):
        CAB.append(i*10)

    LAI = [1,2,3,4,5,6,7] """
    CAB = [30,40]
    LAI = [5,4]

    with open(resultfilename,'w') as res:
        print('CAB','LAI','Ftotal_m','F687_m','F760_m','Fr_m','wlFr_m','Ffr_m','wlFfr_m','Ftotal_wm','F687_wm','F760_wm','Fr_wm','wlFr_wm','Ffr_wm','wlFfr_wm','Ftotal_sfm','F687_sfm','F760_sfm','Fr_sfm','wlFr_sfm','Ffr_sfm','wlFfr_sfm',sep= '  ',file=res)

    with open(sifresults,'w') as Fres:
        print('Cab','LAI',sep='  ',file=Fres)

    wl = []
    with open('cwavelets/libradtranscope/floxseries_ae/radcomplete_004_5_7_ae_conv.dat','r') as g:
        for line in g:
            line = line.split()
            wl.append(float(line[0]))
    wl = np.array(wl)

elif testdata == 'flox' or testdata == 'hyplant':    
    with open(resultfilename,'w') as res:
        print('num','Ftotal_wm','F687_wm','F760_wm','Fr_wm','wlFr_wm','Ffr_wm','wlFfr_wm','Ftotal_sfm','F687_sfm','F760_sfm','Fr_sfm','wlFr_sfm','Ffr_sfm','wlFfr_sfm',sep= '  ',file=res)
    if testdata == 'flox':
        day = '2021-05-30'
        timestampbegin = day+' 05:00:00'
        timestampend = day+' 17:00:00'
        #datapath = "../FloX_Davos/SDcard/FloX_JB038AD_S20210603_E20211119_C20211208.nc"
        datapath = "../FloX_Davos/FloX_JB023HT_S20210326_E20210610_C20210615.nc"
        fluodata = xr.open_dataset(datapath, group="FLUO")
        metadata = xr.open_dataset(datapath,group='METADATA')
        wlfluo = np.array(fluodata["wavelengths"])
        upseries = fluodata["upwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").mean("time")
        downseries = fluodata["downwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").mean("time")
        uperrors = fluodata["upwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").std("time")
        downerrors = fluodata["downwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").std("time")
        iflda_ref = metadata['SIF_A_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").mean("time")
        iflda_err = metadata['SIF_A_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").std("time")
        ifldb_ref = metadata['SIF_B_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").mean("time")
        ifldb_err = metadata['SIF_B_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").std("time")
        int_num = len(upseries)

    if testdata == 'hyplant':
        fileext = 'hyplant'
        wl = get_wavelengths('../Data/hyplant2020_wl.csv')
        wl = np.array(wl)
        refname = 'cwavelets/Hyplant/compare_reference.dat'
        whitereference = []
        with open(refname,'r') as wf:
            for k,line in enumerate(wf):
                line = line.split()
                whitereference.append(float(line[0]))
        whitereference = np.array(whitereference)









comm_knots = [] 
with open('common_reflectance_knots','r') as ck:
    for val in ck:
        if float(val) < 720 or float(val) > 743:
            if float(val) < 670 or float(val) > 682:
                comm_knots.append(float(val))

    

figtr, axtr = plt.subplots()
########################### Load data ##############################################
###################################################################################

for cab in CAB:
    for lai in LAI:

#for mn in range(int_num):




        
        if testdata == 'scope':
            ################ data selection Scope simulation
            fileext = testdata+'_oensquadrres1015_'+str(cab)+'_'+str(lai)
            wl = []
            noconvwl = []
            with open('cwavelets/libradtranscope/series/wl_array','r') as ncw:
                for line in ncw:
                    line = line.split()
                    noconvwl.append(float(line[0]))
            noconvwl = np.array(noconvwl)
            with open('cwavelets/libradtranscope/floxseries_ae_oen/radcomplete_004_5_7_ae_conv.dat','r') as g:
                        for line in g:
                            line = line.split()
                            wl.append(float(line[0]))
            wl = np.array(wl)
            # select simulation: first number - Ch content, second number - LAI
            arrays = prepare_arrays(cab,lai,sifeffec)
            refname = 'cwavelets/libradtranscope/floxseries_ae_oen/whiteref_ae_conv.dat'
            whitereference = []
            with open(refname,'r') as wf:
                for k,line in enumerate(wf):
                    line = line.split()
                    whitereference.append(float(line[0]))
            whitereference = np.array(whitereference)
            
            atsensornonoise = arrays[0]
            
            atsensor_noSIF = arrays[1]
            reflectance = arrays[2]
            wlR = arrays[5]
            interR = interp1d(wlR, reflectance,kind='cubic')
            R = interR(wl)
            albedo = arrays[3]
            interA = interp1d(wlR,albedo,kind='cubic')
            A = interA(wl)
            A_adj = A/0.3
            F_input = arrays[4]
            F_input_nonsmoothed = F_input
            wlF = noconvwl
            interF = interp1d(wlF, F_input,kind='cubic')
            F_input = interF(wl)
            scopref = arrays[7]
            sunspectrum,gaussf = match_solspec(wl,0.3)
            
            #atsensor_woF = atsensornonoise-F_input_nonsmoothed # maybe subtract real at sensor SIF here (including O2 reabsorption) - done!
            atsensor_woF = atsensor_noSIF
            R_expect = np.divide(atsensor_woF,whitereference)
            refnoise, whitereference = add_noise(whitereference,1000,noise)
            sensornoise, atsensornonoise = add_noise(atsensornonoise,1000,noise)
            

            fig,ax = plt.subplots()
            ax.plot(wl,atsensornonoise,color='tab:green',linewidth=0.8,label='TOC radiance')
            ax.plot(wlR,scopref,color='limegreen',linewidth=0.8,label='SCOPE TOC compare')
            ax2 = ax.twinx()
            ax2.plot(wl,F_input,color='tab:red',linewidth=0.8,label='SIF')
            ax.set_xlabel(r'Wavelength [nm]')
            ax.set_ylabel(r'Radiance [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
            ax2.set_ylabel(r'Radiance [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]',color='tab:red')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.set_xlim(wl[0],wl[-1])
            fig.savefig('example_radF.pdf')
            plt.show()
           
            
            
            

        elif testdata == 'flox':
            ################ data selection FloX data
            
            
            floxup = upseries[:,mn].data
            floxdown = downseries[:,mn].data
            errorup = uperrors[:,mn].data
            errordown = downerrors[:,mn].data
            floxtime = str(upseries.time[mn].data)[11:16]
            iflda_ref_i = iflda_ref[mn].data
            iflda_err_i = iflda_err[mn].data
            ifldb_ref_i = ifldb_ref[mn].data
            ifldb_err_i = ifldb_err[mn].data
            print(floxtime)
            fileext = 'diffmethod_oens_'+floxtime
            
            
            wl = wlfluo[10:-10]
            print(wl)
            
            dataerror = np.sqrt(np.square(np.divide(errorup,floxup)) + np.square(np.divide(errordown,floxdown)))[10:-10]
            print('Mean Standard Deviation FloX Acquisition: {:.2%}'.format(np.mean(dataerror[10:-10])))
            plt.figure()
            plt.plot(wl,np.divide(errorup,floxup)[10:-10],label='up')
            plt.plot(wl,np.divide(errordown,floxdown)[10:-10],label='down')
            #plt.ylim(0,1)
            plt.legend()
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Relative Standard Deviation over 15min')
            plt.savefig(newdir+'/noise_'+fileext+'.pdf')
            #plt.show()
            whitereference = floxdown[10:-10]*1000
            atsensornonoise = floxup[10:-10]*1000
            R_expect = np.divide(atsensornonoise,whitereference)


        elif testdata == 'hyplant':
            atsensornonoise = prepare_arrays_hp()[0]
            atsensornonoise = np.array(atsensornonoise)
            R_expect = np.divide(atsensornonoise,whitereference)
        else:
            print('Please select testdata scope, flox or hyplant.')
            #exit(0)

        ###################################################################################
        ###################################################################################


        ########## set global parameters ###################################

        global N
        N = len(wl)
        dwl = wl[1]-wl[0]
        wl_lb = np.argmin(np.fabs(wl-670))
        wl_ub = np.argmin(np.fabs(wl-800))
        fwhm = 0.3

        ##### set wavelet decomposition params #############################

        """ nlevel = 81
        jmin = -2.7222
        jmax = 17.7778 """
        wav = pywt.ContinuousWavelet('gaus2')
      

        # print the range over which the wavelet will be evaluated
        print("Continuous wavelet will be evaluated over the range [{}, {}]".format(
        wav.lower_bound, wav.upper_bound))

        width = wav.upper_bound - wav.lower_bound 


        nlevel = 10
        jmin = 1.3
        #jmin = 0.5
        jmax = 2.2
        blubb = np.logspace(jmin,jmax,num=nlevel,base=2)
        #scales = [1,2,3,6,9,12,24,36,90,300]
        #coefs,freqs = pywt.cwt(atsensornonoise,scales,'gaus2',method='conv')
        f = pywt.scale2frequency('gaus2', blubb)
        wlscales = 1/f*dwl
        print(wlscales)


        yf = fft(atsensornonoise)

        xf = fftfreq(N, dwl)[:N//2]
        scales = 1/fftfreq(N)[:N//2]
        f = pywt.scale2frequency('gaus2', scales)/dwl


        """ print(1/f)
        print(1/xf)
        print(2.0/N * np.abs(yf[0:N//2]))
        plt.figure()
        plt.plot(1/xf,2.0/N * np.abs(yf[0:N//2]))
        plt.show() """


        ###################################################################################


        ########## preparation, initial conditions ###################################
        smoothing = 0.0001 # as power of 10 # smoothing factor for the spline to apparent reflectance (0.0005 for downwelling reference) - 0.001 standard for simulations
        # Davos: 0.00002
        # Oensingen: 0.001
        sunspectrum,gaussf = match_solspec(wl,0.3)
        sunmins = find_peaks(-sunspectrum)
        sigmins = find_peaks(-atsensornonoise)


        #whitereference = sunspectrum
        if testdata == 'scope':

            transferf = np.divide(atsensor_woF,whitereference)
            
            
            
            
            
            R_expect = np.divide(atsensor_woF,whitereference)
            interR_expect = interp1d(wl,R_expect,kind='cubic')
        minknots = 13
        maxknots = 14
        apparentrefl = np.divide(atsensornonoise,whitereference)
        interRa = interpolate.UnivariateSpline(wl,apparentrefl,s=smoothing,k=3)
        appknots = interRa.get_knots()
        print('Number of knots used for Apparent Reflectance Spline: {:d}'.format(len(appknots)))
        if len(appknots) > maxknots or len(appknots) < minknots:  # 17-35 for reflectance based
            print('Adjusting spline representation...')
            interRa, smoothing = adjust_smoothing(wl,apparentrefl,smoothing,minknots,maxknots)
            appknots = interRa.get_knots()
        print('Number of knots used for Apparent Reflectance Spline: {:d}'.format(len(appknots)))
        appderiv = np.diff(interRa(wl))/dwl

        smoothappref = interRa(wl)
        rededge_min = np.argmin(np.fabs(wl-690))
        rededge_max = np.argmin(np.fabs(wl-750))
        rededgemax = np.argmax(appderiv[rededge_min:rededge_max])
        tot_rededgemax = rededge_min+rededgemax
        plt.figure()
        plt.plot(wl,interRa(wl))
        plt.plot(appknots,interRa(appknots),'.')
        plt.plot(wl[:-1],np.diff(interRa(wl))/dwl/np.max(np.diff(interRa(wl))/dwl)*np.max(interRa(wl)))
        plt.plot(wl[tot_rededgemax],appderiv[tot_rededgemax],'o',color='red')

        minpeak = np.argmin(np.fabs(wl-757))
        maxpeak = np.argmin(np.fabs(wl-768))
        nopeak_appref = []
        nopeak_wl = []
        for i in range(len(wl)):
            if i < minpeak or i > maxpeak:
                nopeak_appref.append(apparentrefl[i])
                nopeak_wl.append(wl[i])

        interRa_smooth = interpolate.UnivariateSpline(nopeak_wl,nopeak_appref,s=smoothing,k=1)
        smooth_knots = interRa_smooth.get_knots()

        print('Number of knots used for Smooth Reflectance Spline: {:d}'.format(len(smooth_knots)))
        if len(smooth_knots) > maxknots or len(smooth_knots) < minknots:  # 17-35 for reflectance based
            print('Adjusting spline representation...')
            interRa_smooth, smoothing = adjust_smoothing(nopeak_wl,nopeak_appref,smoothing,minknots,maxknots)
            smooth_knots = interRa_smooth.get_knots()
        print('Number of knots used for Smooth Reflectance Spline: {:d}'.format(len(smooth_knots)))
        appderiv = np.array(np.diff(interRa_smooth(wl))/dwl)
        deriv_cond = np.logical_or(np.fabs(appderiv) > 0.2*np.max(appderiv), appderiv < 0.2*np.max(appderiv))   #0.4 and 0.2 used for proposal results

        comm_knots = []
        for i,knot in enumerate(smooth_knots):
            wl_ind = np.argmin(np.fabs(wl-knot))
            if wl_ind == len(wl)-1:
                comm_knots.append(knot)
            elif deriv_cond[wl_ind] == True or testdata == 'flox':
                comm_knots.append(knot)
        #comm_knots.append(761)
        transferspline = interpolate.UnivariateSpline(wl,transferf,s=smoothing,k=2)
        print(len(transferspline.get_knots()))
        axtr.plot(wl,transferspline(wl),linewidth=0.5)
        axtr.set_ylabel(r'Transfer Function Smooth')
        axtr.set_xlabel(r'Wavelength [nm]')
        
        fig,ax= plt.subplots()
        ax2 = ax.twinx()
        ax.plot(wl,apparentrefl)
        ax.plot(wl,interRa_smooth(wl))
        ax.plot(comm_knots,interRa_smooth(comm_knots),'.')
        ax2.plot(wl[:-1],appderiv)

        ####### find multiplicative wavelet differences between smooth apparent reflectance and input :

        sifjmin = -3
        sifjmax = 13
        sifnlevel = 80
        sifscales = np.logspace(sifjmin,sifjmax,num=sifnlevel,base=2)
        appref_decomp = create_decomp_p(interRa_smooth(wl),sifjmin,sifjmax,sifnlevel)
        Rexpect_decomp = create_decomp_p(R,sifjmin,sifjmax,sifnlevel)

        """ coeff_multdiff = np.divide(appref_decomp,Rexpect_decomp)
        plt.figure()
        for j in range(len(coeff_multdiff)):
            plt.plot(wl,coeff_multdiff[j],label=j)
        plt.ylim(-3,3)
        plt.legend()
        plt.show() """
        num_manlevels = 20
        starting_level = 30
        def r_difffunc(mult_coeffs):
            newcoeffs = appref_decomp.copy()
            for i in range(len(mult_coeffs)-1):
                newcoeffs[i+starting_level] = (appref_decomp[i+starting_level]*mult_coeffs[i])
            newcoeffs = np.array(newcoeffs)
            R_new = icwt(newcoeffs,wavelet='cmhat',scales=sifscales,x_mean=np.mean(interRa_smooth(wl)))
            return np.sqrt(np.sum(np.square(R_new-R)))

        multcoeffs_init = np.ones(num_manlevels)
        #multcoeffs_init[-1] = np.mean(interRa_smooth(wl))
        plt.figure()
        for i in range(num_manlevels):
            plt.plot(wl,np.divide(appref_decomp[starting_level+i],Rexpect_decomp[starting_level+i]),'.',label=i)
        plt.legend()
        result_mult = optimize.minimize(r_difffunc,multcoeffs_init)
        print(result_mult.x)
        resdecomp = appref_decomp.copy()
        for i in range(len(result_mult.x)-1):
            resdecomp[i+starting_level] = (appref_decomp[starting_level+i]*result_mult.x[i])
        resdecomp = np.array(resdecomp)
        resR = icwt(resdecomp,wavelet='cmhat',scales=sifscales,x_mean=np.mean(interRa_smooth(wl)))
        plt.figure()
        plt.plot(wl,R,label='Input')
        plt.plot(wl,resR,label='Coefficient Optimized')
        plt.plot(wl,interRa_smooth(wl),label='Smooth Apparent Refl')
        plt.legend()
        plt.show()





        """ Fsolve_R = fsolve_solver(atsensornonoise,whitereference,interRa_smooth(wl)-0.2,2)
        plt.figure()
        plt.plot(wl,R_expect,label='expected')
        plt.plot(wl,Fsolve_R,label='fsolve')
        plt.legend()
        plt.show() """

        """ interR = interpolate.UnivariateSpline(wl,R_expect,s=0,k=2)
        ind_diff5 = int(7/0.17)
        knotinds_new = [tot_rededgemax,tot_rededgemax-ind_diff5,tot_rededgemax-2*ind_diff5,tot_rededgemax+ind_diff5,tot_rededgemax+2*ind_diff5,0,len(wl)-1,ind_diff5,2*ind_diff5,len(wl)-1-ind_diff5,len(wl)-1-2*ind_diff5]
        knots_new = wl[knotinds_new]
        comm_knots = knots_new
        plt.figure()
        plt.plot(wl,interRa(wl))
        plt.plot(comm_knots,interRa(comm_knots),'.')
        plt.show() """
        #R_expect = interR(wl)

        ###################################################################################


        ########## plotting input spectra #############################
        plt.figure()
        plt.plot(wl,atsensornonoise,label='Upwelling Radiance')
        plt.plot(wl,whitereference,label='Downwelling Radiance')
        plt.legend()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Radiance [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
        plt.savefig(newdir+'/flowchart1_'+fileext+'.pdf')

        plt.figure()
        if testdata == 'scope':
            plt.plot(wl,R_expect,label='Transfer Function',color='tab:orange')
        plt.plot(wl,apparentrefl,label='Apparent Reflectance',color='tab:red',alpha=0.5)
        plt.plot(wl,interRa(wl),label='Apparent Reflectance Spline',color='tab:red')
        plt.plot(interRa.get_knots(),interRa(interRa.get_knots()),'.',color='tab:red')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Reflectance')
        plt.savefig(newdir+'/flowchart3_'+fileext+'.pdf')
        plt.legend()

        apparentrefl = interRa(wl)
        ###################################################################################


        ########## Wavelet Decompositions, Reconstructions and Reconstruction errors #############################
        # a) reference spectrum
        refspec = create_decomp_p(whitereference,jmin,jmax,nlevel)
        refrecon = icwavelet(refspec,N,jmin,jmax,nlevel)
        fig,ax = plt.subplots()

        ax.plot(wl,refrecon)
        ax2 = ax.twinx()
        ax2.plot(wl,whitereference,alpha=0.5,color='red')
        fig.savefig('Lowlevel_recon.pdf')

        #partrefrecon = icwavelet(refspec,N,jmin,jmax,nlevel,minrec=11,maxrec=60)
        refrecon_res = whitereference-refrecon
        spec_refrecerror = np.divide(np.fabs(refrecon_res),whitereference)
        refrecerror = np.mean(np.divide(np.fabs(refrecon_res),whitereference))

        # b) at sensor spectrum
        sensorspec = create_decomp_p(atsensornonoise,jmin,jmax,nlevel)
        nkl = len(comm_knots)

        #plot_powerspectrum(wl,sensorspec.real,blubb,wlscales,atsensornonoise,'ssqcwt_test.pdf')
        #plt.show()

        """ knotinds_num = Counter(shiftknotsarr)
        shiftknots_single = [val for val in knotinds_num]
        print(shiftknots_single) """

        #plot_powerspectrum(wl,sensorspec,scales,wlscales)

        sensorrecon = icwavelet(sensorspec,N,jmin,jmax,nlevel)
        #partsensorrecon = icwavelet(sensorspec,N,jmin,jmax,nlevel,minrec=11,maxrec=60)
        sensorrecon_res = atsensornonoise-sensorrecon
        spec_sensrecerror = np.divide(np.fabs(sensorrecon_res),atsensornonoise)
        sensrecerror = np.mean(np.divide(np.fabs(sensorrecon_res),atsensornonoise))
        normalizedsensorspec = [sensorspec[i]/scales[i]**0.5 for i in range(len(sensorspec))]
        normalizedsensorspec = np.array(normalizedsensorspec)

        print('Reference Reconstruction Accuracy: {:%}, Atsensor Reconstruction Accuracy:{:%}'.format(refrecerror,sensrecerror))
        if refrecerror > 0.01 or sensrecerror > 0.01:
            print('The reconstruction error is large, please check the decomposition!')

        # expected error for the reflectance, based on the reconstruction error (note: as the reconstruction is not used for the retrieval, this is really only an estimate of the goodness of the decomposition)
        reflerror_progn = np.sqrt(2*np.square(spec_sensrecerror))

        # if input known: plot F decomposition:
        sifjmin = -2
        sifjmax = 17
        sifnlevel = 80



        """ F_decomp = create_decomp_p(F_input,sifjmin,sifjmax,sifnlevel)
        Fscales = np.logspace(sifjmin,sifjmax,num=sifnlevel,base=2.0)
        siff = pywt.scale2frequency('gaus2', Fscales)
        Fwlscales = 1/siff*dwl """
        #plot_powerspectrum(wl,F_decomp,Fscales,Fwlscales,F_input)
        ###################################################################################


        ####### plot error prognosis and reconstruction errors ############################
        plt.figure()
        plt.plot(wl,reflerror_progn)

        plt.figure()
        plt.plot(wl,refrecon_res,label=r'Reference Residual, mean {:.2f} $\%$'.format(refrecerror*100))
        plt.plot(wl,sensorrecon_res,label=r'Sensor Residual, mean {:.2f} $\%$'.format(sensrecerror*100))
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Radiance [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
        plt.legend()
        plt.savefig('flowchart2a_'+fileext+'.pdf')


        plt.figure()
        plt.plot(wl,R_expect,label='True Reflectance')
        plt.plot(wl,apparentrefl,label='Apparent Reflectance')
        plt.fill_between(wl, R_expect-np.multiply(reflerror_progn,R_expect), R_expect+np.multiply(reflerror_progn,R_expect), alpha = 0.5,label='Expected Error')
        plt.legend()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Reflectance')
        plt.savefig(newdir+'/flowchart2b_'+fileext+'.pdf')
        #plt.show()
        ###################################################################################



        ###################### Calculate initial differences of the at-sensor and to be optimized decompositions ###################

        """ # set reflectance knots to be used in optimization
        shiftknots = appknots
        # often, there is no knot after 770nm and only one at the end, as this is a straight line, but having a knot here makes the final result more robust
        if shiftknots[-2] < 770:
            insindex = np.argmin(np.fabs(wl-(wl[-1]+shiftknots[-2])*0.5))
            shiftknots = np.insert(shiftknots,-1,wl[insindex])

        mindiff = []
        mindiff_appref = []
        rmmask = []
        R_best_int = interp1d(shiftknots,interR(shiftknots),kind='quadratic')
        R_best = R_best_int(wl)

        Rapp_best_int = interp1d(shiftknots,interRa(shiftknots),kind='quadratic')
        Rapp_best = Rapp_best_int(wl)
        plt.figure()
        plt.plot(wl,R_best,label='best expected with these knots')
        plt.plot(wl,interR(wl),label='best expected with all knots')
        plt.legend()
        #plt.show()

        for i in range(len(refspec)):
            diff = np.square(np.fabs(create_decomp_p(np.multiply(whitereference,R_best),jmin,jmax,nlevel,i) - sensorspec[i]))
            meandiff = np.mean(diff)
            mindiff.append(np.fabs(meandiff)/scales[i])
            
            diff_apprefl = np.square(np.fabs(create_decomp_p(np.multiply(whitereference,Rapp_best),jmin,jmax,nlevel,i) - sensorspec[i]))
            meandiff_appref = np.mean(diff_apprefl)
            
            mindiff_appref_l = meandiff_appref/scales[i]
            mindiff_appref.append(np.fabs(mindiff_appref_l))

            levdiff = np.mean(np.square(refspec[i]-sensorspec[i]))/scales[i]
            if levdiff < mindiff_appref_l:
                print('Kicked out level {:d}'.format(i))
                rmmask.append(i)
        mindiff_appref = np.array(mindiff_appref)
        if len(rmmask) > 0:
            print('Need to reduce levels for level selection...')
            exit(0)
        #retlevels = np.delete(retlevels,rmmask) """

        ###################################################################################


        ############ level selection ######################################################
            
        initinds,initwl,initr,retlevels = level_selection(sensorspec,refspec,wl,wlscales)
        retlevels = np.array(retlevels,dtype=int)
        print('Chosen levels:')
        print(retlevels)
        #plt.show()
        ###################################################################################


        ############### optimization process ##############################################
        """ appinit = interRa(shiftknots)
        print('Using {:d} knots for optimization!'.format(len(shiftknots)))
        """
        beginindex, wlnew, tanhparams = optimize_logistic(wl,atsensornonoise,whitereference,apparentrefl,retlevels[:-4],690,810)
        R_final = refl_tanh(wl,*tanhparams)
        wl_points, final_reflpoints, stddvs, finaldev = optimize_knots(wl,comm_knots[:nkl],sensorspec,refspec,whitereference,R_final,retlevels,atsensornonoise)

        #errexpect = np.square(np.fabs(create_decomp_p(np.multiply(whitereference,R_best),jmin,jmax,nlevel,0) - sensorspec[0]))
        #print('Smallest Residual:')
        #print(errexpect)

        ###################################################################################

        ########### Check algorithm performance (did the value of the difference function improve for the different levels?) ##################
        """ mindiff_retl = [mindiff[l] for l in retlevels]
        mindiff_a_retl = [mindiff_appref[l] for l in retlevels]
        optm_performance = ((np.array(finaldev) - np.array(mindiff_retl)))
        nooptm_diff = ((np.array(mindiff_a_retl) - np.array(mindiff_retl)))
        plt.figure()
        plt.plot(retlevels,optm_performance,'.',label='Result to expected minimum')
        plt.plot(retlevels,nooptm_diff,'.',label='Input (apparent reflectance) to expected minimum')
        plt.plot(retlevels,-optm_performance,'.',color='tab:blue',alpha=0.5)
        plt.ylabel('Absolute difference')
        plt.xlabel('Level')
        plt.legend()
        plt.savefig(newdir+'/flowchart5c_'+fileext+'.pdf') """
        ###################################################################################

        """  ################# error estimation #################################################
        errormin, errormax = np.argmin(np.fabs(wl-wl_points[0])), np.argmin(np.fabs(wl-wl_points[-1]))
        wlerror = wl[errormin:errormax]
        print(stddvs)
        #errorinterp = interp1d(wl_points,stddvs,kind='quadratic')
        #totRerror = np.sqrt(np.square(errorinterp(wlerror)))   #reflerror_progn[errormin:errormax]
        plt.figure()
        plt.plot(wl_points,stddvs,'.')
        #plt.plot(wl[errormin:errormax], errorinterp(wl[errormin:errormax]))

        ###################################################################################
        """

        ################### reflectance interpolation #######################################
        wl_sorting = np.argsort(wl_points)
        wl_points = wl_points[wl_sorting]
        final_reflpoints = final_reflpoints[wl_sorting]
        reflspline = interp1d(wl_points,final_reflpoints,kind='linear')
        """ polyfitcoeffs = np.polyfit(wl_points,final_reflpoints,6)
        reflspline = np.poly1d(polyfitcoeffs)
        sortedknots = np.sort(comm_knots[:nkl])
        print(wl_points)
        print(sortedknots) """
        #reflspline = interpolate.LSQUnivariateSpline(wl_points[1:-1],final_reflpoints[1:-1],sortedknots)
    
        wlpoints_inds = [np.argmin(np.fabs(wl-wl_points[i])) for i in range(len(wl_points))]
        print(wlpoints_inds)
        wlnew = wl[wlpoints_inds[0]+1:wlpoints_inds[-1]]
        wlnew_argmin = wlpoints_inds[0]+1
        wlnew_argmax = wlpoints_inds[-1]
        ################################################################################### 


        ################# SFM for comparison ##############################################
        wlSFM = wl[wl_lb:wl_ub]
        w_F = np.ones(len(wlSFM))
        Lin = whitereference[wl_lb:wl_ub]
        Lup = atsensornonoise[wl_lb:wl_ub]
        opt_alg = 'trf'
        x, f_wvl, r_wvl, resnorm, exitflag, nfevas = FLOX_SpecFit_6C(wlSFM,Lin,Lup,[1,1],w_F,opt_alg,wlSFM)
        ###################################################################################



        ######################### reflectance plotting ############################################################
        
        #reflspline = transferspline
        plt.figure()
        plt.plot(wl_points,final_reflpoints,'.',label='Final Knot Positions',color='tab:blue')
        plt.plot(wlnew,reflspline(wlnew),label='Interpolation',color='tab:blue')
        #plt.fill_between(wlerror, reflspline(wlerror)-totRerror, reflspline(wlerror)+totRerror, alpha = 0.5,label='Expected Error',color='tab:blue')
        if testdata == 'scope':
            plt.plot(wl,transferf,label='Transfer Function',color='tab:red')
        plt.plot(wl,apparentrefl,label='Apparent Reflectance',color='tab:orange')
        plt.plot(wlSFM,r_wvl,label='SFM reflectance',color='tab:green')
        #plt.plot(comm_knots[:15],interRa(comm_knots[:15]),'.',color='tab:pink')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Reflectance')
        #lt.ylim(0.40,0.55)
        #plt.xlim(740,800)
        plt.legend()
        plt.savefig(newdir+'/flowchart6_'+fileext+'.pdf')
        ###################################################################################


        ################## SIF extraction ###################################################
        sifjmin = -3
        sifjmax = 13.5
        sifnlevel = 80
        sifscales = np.logspace(sifjmin,sifjmax,num=sifnlevel,base=2.0)
        f = pywt.scale2frequency('gaus2', sifscales)
        sifwlscales = 1/f*(wlnew[1]-wlnew[0])
        res_ref_decomp = create_decomp_p(reflspline(wlnew),sifjmin,sifjmax,sifnlevel)
        R_expect_decomp = create_decomp_p(interR_expect(wlnew),sifjmin,sifjmax,sifnlevel)
        #plot_powerspectrum(wlnew,res_ref_decomp-R_expect_decomp,sifscales,sifwlscales,reflspline(wlnew),'powspec_rres.pdf')
        R_recon =icwt(res_ref_decomp,wavelet='cmhat',scales=sifscales,x_mean=np.mean(reflspline(wlnew)),recmin=30)
        res_ref_decomp_a = res_ref_decomp.copy()
        for i in range(30,70):
            res_ref_decomp_a[i] = res_ref_decomp_a[i]*0.5
        R_recon_change = icwt(res_ref_decomp_a,wavelet='cmhat',scales=sifscales,x_mean=np.mean(reflspline(wlnew)),recmin=30)
        signal_woF = np.multiply(whitereference[wlnew_argmin:wlnew_argmax],reflspline(wlnew))
        #signal_woF = np.multiply(whitereference[wlpoints_inds[0]:wlpoints_inds[-1]],R_best_int(wlnew))
        F = atsensornonoise[wlnew_argmin:wlnew_argmax]-signal_woF
        #F = atsensornonoise[beginindex:]-signal_woF
        F_res_decomp = create_decomp_p(F,sifjmin,sifjmax,sifnlevel)
        #F_rec = icwavelet(F_res_decomp,len(wlnew),sifjmin,sifjmax,sifnlevel,minrec=34) # 31
        F_rec = icwt(F_res_decomp,wavelet='cmhat',scales=sifscales,x_mean=np.mean(F),partial=True,recmin=40)
        plt.figure()
        plt.plot(wlnew,reflspline(wlnew))
        plt.plot(wlnew,R_recon)
        plt.plot(wlnew,R_recon_change,label='changed components')
        plt.plot(wlnew,interR_expect(wlnew))
        plt.legend()
        plt.show()
        ###################################################################################


        ############################ SIF plotting ##################################################

        plt.figure()
        if testdata == 'scope':
            plt.plot(wl,F_input,label='Input',color='limegreen')
        plt.plot(wlnew,F,label='Wavelet Method',color='tab:red',alpha=0.5,linewidth=3)
        plt.plot(wlnew,F_rec,label='Large Scale Reconstruction',color='tab:red')
        #plt.fill_between(wlerror, F_rec[errormin:errormax]-np.multiply(whitereference[errormin:errormax],totRerror),F_rec[errormin:errormax]+np.multiply(whitereference[errormin:errormax],totRerror),alpha=0.5,label='Estimated retrieval error')
        if testdata == 'flox':
            plt.fill_between(wlnew,F_rec-np.multiply(dataerror[wlnew_argmin:wlnew_argmax],F_rec), F_rec+np.multiply(dataerror[wlnew_argmin:wlnew_argmax],F_rec),alpha=0.5,color='forestgreen',label='Estimated STD of data')
            plt.plot(760,iflda_ref_i,'o',color='tab:blue',label='iFLD')
            plt.errorbar(760,iflda_ref_i, yerr=iflda_err_i,color='tab:blue')
            plt.plot(687,ifldb_ref_i,'o',color='tab:blue')
            plt.errorbar(687,ifldb_ref_i, yerr=ifldb_err_i,color='tab:blue')
        
        plt.plot(wlSFM,f_wvl,label='SFM',color='tab:green')
        

        plt.legend()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Radiance [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
        plt.savefig(newdir+'/flowchart7_'+fileext+'.pdf')

        """ ###################################################################################
        if testdata == 'scope':
            print('Final SIF deviation: {:.2f}'.format(np.mean(np.fabs(F_rec-F_input[wlnew_argmin:wlnew_argmax]))))
            Ftotal_m,F687_m,F760_m,Fr_m,wlFr_m,Ffr_m,wlFfr_m = evaluate_sif(wl,F_input)
        Ftotal_wm,F687_wm,F760_wm,Fr_wm,wlFr_wm,Ffr_wm,wlFfr_wm = evaluate_sif(wlnew,F_rec)
        Ftotal_sfm,F687_sfm,F760_sfm,Fr_sfm,wlFr_sfm,Ffr_sfm,wlFfr_sfm = evaluate_sif(wlSFM,f_wvl)
        
        with open(sifresults,'a') as Fres:
            print(fileext,file=Fres,sep='  ')
            if testdata == 'scope':
                for val in wl:
                    print(val,file=Fres,sep='  ',end='  ')
                print(file=Fres) 
                for val in F_input:
                    print(val,file=Fres,sep='  ',end='  ')
            print(file=Fres)
            for val in wlnew:
                print(val,file=Fres,sep='  ',end='  ')
            print(file=Fres)
            for val in F_rec:
                print(val,file=Fres,sep='  ',end='  ')
            print(file=Fres)

        #Ftoterr_wm = np.fabs(Ftotal_m-Ftotal_wm) / Ftotal_m
        #Ftoterr_sfm = np.fabs(Ftotal_m-Ftotal_sfm) / Ftotal_m
        if testdata == 'scope':

            with open(resultfilename,'a') as res:
                print(cab,lai,Ftotal_m,F687_m,F760_m,Fr_m,wlFr_m,Ffr_m,wlFfr_m,Ftotal_wm,F687_wm,F760_wm,Fr_wm,wlFr_wm,Ffr_wm,wlFfr_wm,Ftotal_sfm,F687_sfm,F760_sfm,Fr_sfm,wlFr_sfm,Ffr_sfm,wlFfr_sfm,sep= '  ',file=res)
        elif testdata == 'flox' or testdata == 'hyplant':
            with open(resultfilename,'a') as res:
                print(mn,Ftotal_wm,F687_wm,F760_wm,Fr_wm,wlFr_wm,Ffr_wm,wlFfr_wm,Ftotal_sfm,F687_sfm,F760_sfm,Fr_sfm,wlFr_sfm,Ffr_sfm,wlFfr_sfm,sep= '  ',file=res)
                     """

        plt.show()