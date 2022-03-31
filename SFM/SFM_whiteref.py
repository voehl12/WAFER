import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.function_base import _msort_dispatcher
from scipy.interpolate.fitpack2 import UnivariateSpline 
import xarray as xr
from scipy.interpolate import interp1d
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.special import gamma, factorial
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
from scipy.optimize import NonlinearConstraint
from scipy.optimize import least_squares
import warnings
from matplotlib import cm
from matplotlib.offsetbox import AnchoredText
import pywt
from scipy import interpolate
import pprocess
import time
from matplotlib import rc

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


global N 
N = 1024

global dt 
dt = 0.20


def wl_grid(resolution,length):
    wl0 = 640.0
    grid = []
    while len(grid) < length:
        grid.append(wl0)
        wl0 += resolution
    return grid

def prepare_arrays(cab,lai,feffef):
    completename = '../cwavelets/libradtranscope/series/res02/radcomplete_{}_{:d}_{:d}.dat'.format(feffef,cab,lai)
    woFname = '../cwavelets/libradtranscope/series/res02/radwoF_{}_{:d}_{:d}.dat'.format(feffef,cab,lai)
    reflname = '../reflectance/rho_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    albedoname = '../reflectance/albedo_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    Fname = '../fluorescence/F_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    wlRname = '../reflectance/wlR'
    wlFname = '../reflectance/wlF'
    filenames = [completename,woFname,reflname,albedoname,Fname,wlRname,wlFname]
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

def evaluate_sif(wl,spec):
    totmin = np.argmin(np.fabs(wl-650))
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

def bspleval(x, knots, coeffs, order, debug=False):
    '''
    Evaluate a B-spline at a set of points.

    Parameters
    ----------
    x : list or ndarray
        The set of points at which to evaluate the spline.
    knots : list or ndarray
        The set of knots used to define the spline.
    coeffs : list of ndarray
        The set of spline coefficients.
    order : int
        The order of the spline.

    Returns
    -------
    y : ndarray
        The value of the spline at each point in x.
    '''

    k = order
    t = knots
    m = len(t)
    npts = len(x)
    B = np.zeros((m-1,k+1,npts))

    if debug:
        print('k=%i, m=%i, npts=%i' % (k, m, npts))
        print('t=', t)
        print('coeffs=', coeffs)

    ## Create the zero-order B-spline basis functions.
    for i in range(m-1):
        B[i,0,:] = np.float64(np.logical_and(x >= t[i], x < t[i+1]))

    if (k == 0):
        B[m-2,0,-1] = 1.0

    ## Next iteratively define the higher-order basis functions, working from lower order to higher.
    for j in range(1,k+1):
        for i in range(m-j-1):
            if (t[i+j] - t[i] == 0.0):
                first_term = 0.0
            else:
                first_term = ((x - t[i]) / (t[i+j] - t[i])) * B[i,j-1,:]

            if (t[i+j+1] - t[i+1] == 0.0):
                second_term = 0.0
            else:
                second_term = ((t[i+j+1] - x) / (t[i+j+1] - t[i+1])) * B[i+1,j-1,:]

            B[i,j,:] = first_term + second_term
        B[m-j-2,j,-1] = 1.0

    if debug:
        plt.figure()
        for i in range(m-1):
            plt.plot(x, B[i,k,:])
        plt.title('B-spline basis functions')

    ## Evaluate the spline by multiplying the coefficients with the highest-order basis functions.
    y = np.zeros(npts)
    for i in range(m-k-1):
        y += coeffs[i] * B[i,k,:]

    if debug:
        plt.figure()
        plt.plot(x, y)
        plt.title('spline curve')
        plt.show()

    return(y)

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
    F        = FFAR_RED + FRED
    #np.multiply((FFAR_RED + FRED),(1.-(1.-RHO)))


    # -- UPWARD RADIANCE
    y        = np.multiply(RHO,L0) + F
    return y-inp


def FLOX_SpecFit_6C(wvl, L0, LS, fsPeak, w, alg, oWVL):

    
    

    # Apparent Reflectance - ARHO -
    ARHO          = np.divide(LS,L0)
    
    # Excluding O2 absorption bands
    mask1 = np.logical_or(wvl < 686, wvl > 692)
    mask2 = np.logical_or(wvl < 758, wvl > 773)
    id_m = np.argmin(np.fabs(wvl-710))
    mask = np.zeros(len(wvl))
    mask[:id_m] = mask1[:id_m]
    mask[id_m:] = mask2[id_m:]
    mask = np.array(mask)
    mask = mask > 0
    wvl = np.array(wvl)
    wvlnoABS      = wvl[mask]
    ARHOnoABS     = ARHO[mask]
    # knots vector for piecewise spline
    inds = [int(number) for number in np.linspace(1,len(wvlnoABS)-2,20-1+4)]
    knots         = wvlnoABS[inds]
    print(inds)
    # piece wise cubic spline
    sp = interpolate.LSQUnivariateSpline(wvlnoABS,ARHOnoABS,knots)
    firsspline = sp(wvl)
    p_r           = sp(wvl)[inds]
    
    interp = bspleval(wvl,knots,p_r,3,debug=False)
   



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
    f_wvl    = FFAR_RED + FRED
    # np.multiply((FFAR_RED + FRED),(1-(1-RHO)))


    # -- At-sensor modeled radiance
    LSmod        = FLOX_SpecFit_6C_funct(x,L0,wvl,sp,LS_w,knots)


    # --  RETRIEVAL STATS
    residual     = LSmod - LS
    rmse         = np.sqrt(np.sum(np.square(LSmod - LS))/len(LS))
    rrmse        = np.sqrt(np.sum(np.square(np.divide((LSmod - LS),LS)))/len(LS))*100. 






    return x, f_wvl, r_wvl, resnorm, exitflag, nfevas

def add_noise(data,snr,switch):
    if switch == 0:
        return data
    else:

        newdata = np.zeros(len(data))
        for i,val in enumerate(data):
            sigma = val/snr
            s = np.random.normal(0.0, sigma)
            newval = val + s
            newdata[i] = newval
        return newdata

SNR = 500
noise_on = 0

wl = []
with open('../cwavelets/libradtranscope/series/res02/wl_array','r') as g:
            for line in g:
                wl.append(float(line))
            wl = np.array(wl)
wl = np.array(wl)
wl_complete = wl

t2 = []
with open('../cwavelets/libradtranscope/trans2.dat','r') as tf:
    for line in tf:
            line = line.split()
            t2.append(float(line[0]))
wlold = wl_grid(0.1,2048)
t2 = np.array(t2)
intert = interp1d(wlold,t2,kind='cubic')
t2 = intert(wl)

refl_dir = '../reflectance/'
refname = '../../libRadtran-2.0.4/whitealbedos/a03res02.dat'

whitereference = []
with open(refname,'r') as wf:
     for k,line in enumerate(wf):
        line = line.split()
        whitereference.append(float(line[0]))
whitereference = np.array(whitereference)

opt_alg = 'trf'


sifeffec = '004'
CAB = [int(5),int(10)]
for i in range(2,9):
    CAB.append(i*10)

LAI = [1,2,3,4,5,6,7]

wl_lb = np.argmin(np.fabs(wl-670))
wl_ub = np.argmin(np.fabs(wl-780))
wl = wl[wl_lb:wl_ub]
w_F = np.ones(len(wl))
ind_760 = np.argmin(np.fabs(wl-760))
ind_687 = np.argmin(np.fabs(wl-687))


for cab in CAB:
    for lai in LAI:

        arrays = prepare_arrays(cab,lai,sifeffec)
        
        const = np.ones(N)
        atsensor = arrays[0]
        atsensor = add_noise(atsensor,SNR,noise_on)
        atsensor_noSIF = arrays[1]
        atsensor_noSIF = add_noise(atsensor_noSIF,SNR,noise_on)
        reflectance = arrays[2]
        wlR = arrays[5]
        interR = interp1d(wlR, reflectance,kind='cubic')
        R = interR(wl)
        albedo = arrays[3]
        interA = interp1d(wlR,albedo,kind='cubic')
        A = interA(wl)
        A_adj = A/0.3
        F_input = arrays[4]
        wlF = arrays[6]
        interF = interp1d(wlF, F_input,kind='cubic')
        F_input = interF(wl_complete)
        apparentrefl = np.divide(atsensor,whitereference)
        Ftotal_m,F687_m,F760_m,Fr_m,wlFr_m,Ffr_m,wlFfr_m = evaluate_sif(wl,F_input)
        R_input = np.divide(atsensor_noSIF,whitereference)
        
        Lin = whitereference[wl_lb:wl_ub]
        Lup = atsensor[wl_lb:wl_ub]

        x, f_wvl, r_wvl, resnorm, exitflag, nfevas = FLOX_SpecFit_6C(wl,Lin,Lup,[1,1],w_F,opt_alg,wl)

        fig,(ax1,ax2) = plt.subplots(2,sharex=True)
        ax1.plot(wl,f_wvl,label='retrieved')
        ax1.plot(wl_complete,F_input,label='input')
        ax1.legend()
        ax2.plot(wl,r_wvl,label='retrieved')
        ax2.plot(wl_complete,R_input,label='input')
        ax2.plot(wl_complete,apparentrefl,label='apparent reflectance')
        ax2.legend()

        plt.show()