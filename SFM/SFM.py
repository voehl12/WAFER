from scipy import interpolate
from scipy.optimize import least_squares
import numpy as np 

#### Python adapted from https://gitlab.com/ltda/flox-specfit

def FLOX_SpecFit_6C_funct(x,L0,wvl,sp,inp,knots):

    """ 
    Input:

    x: fit parameters, i.e. peak heights of fluorescence and heights of all reflectance knots
    L0: downwelling radiance
    wvl: corresponding wavelength array
    sp: spline to apparent reflectance
    inp: upwelling radiance
    knots: wavelength positions of the knots

    Returns:

    Difference of modelled and input radiance (this is what is going to be minimized)
    """

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


    # -- UPWARD RADIANCE
    y        = np.multiply(RHO,L0) + F
    return y-inp


def FLOX_SpecFit_6C(wvl, L0, LS, fsPeak, w, oWVL, alg = 'lm'):

    """ 
    Input: 
    
    wvl: wavelength-array, 
    L0: downwelling radiance, 
    LS: reflected radiance,
    --> lengths need to match up
    initial peak height fsPeak: array of first guesses in units of input radiances ([1,1] should suffice), 
    weights: array of length LS (can also be set to one), 
    oWVL is a wavelength array for final Fluorescence but it actually needs to be the 
    same as the reflectance wavelength range, which is the original one (just 
    left this as in the original in order for it to be as similar as possible),
    algorithm type ('lm', Levenberg-Marquardt, is default). 
    

    Returns:

    x: Solution vector (peak heights of fluorescence followed by heights of all reflectance knots)
    f_wvl, r_wvl: Spectral fluorescence and reflectance over input wl array
    resnorm, exitflag, nfevas: Costfunction, success of convergence and number of function evaluations 

    """

    # Apparent Reflectance - ARHO -
    ARHO          = np.divide(LS,L0)
    
    # Excluding O2 absorption bands
    mask = np.logical_and(np.logical_or(wvl < 686, wvl > 692),np.logical_or(wvl < 758, wvl > 773))
    mask = np.array(mask)                                                                      
    mask = mask > 0
    
    wvl = np.array(wvl)
    wvlnoABS      = wvl[mask]
    ARHOnoABS     = ARHO[mask]
    # knots vector for piecewise spline
    inds = np.linspace(1,len(wvlnoABS)-2,20-1+4,dtype=int) 
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
        res = least_squares(FLOX_SpecFit_6C_funct,x0,method='trf',max_nfev=100,args=(L0,wvl,sp,LS_w,knots))     
        x, resnorm,residual,exitflag,nfevas = res.x,res.cost,res.fun,res.status,res.nfev
        
        
        if exitflag == -1:
            resnorm = np.NaN

    elif alg == 'lm':
        res = least_squares(FLOX_SpecFit_6C_funct,x0,method='lm',max_nfev=6,args=(L0,wvl,sp,LS_w,knots))    
        x, resnorm,residual,exitflag,nfevas = res.x,res.cost,res.fun,res.status, res.nfev

    else:
        print('Check Optimization algorithm')
        exit(0)
          
    

    # --- OUTPUT SPECTRA

    # -- Reflectance 
    knotvals   = x[2:]
    interp = interpolate.UnivariateSpline(knots,knotvals,s=0)
    RHO          = interp(oWVL)
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


    # -- At-sensor modeled radiance
    LSmod        = FLOX_SpecFit_6C_funct(x,L0,wvl,sp,LS_w,knots)


    # --  RETRIEVAL STATS
    residual     = LSmod - LS
    rmse         = np.sqrt(np.mean(np.square(LSmod - LS)))
    rrmse        = np.sqrt(np.mean(np.square(np.divide((LSmod - LS),LS))))*100. 






    return x, f_wvl, r_wvl, resnorm, exitflag, nfevas, residual