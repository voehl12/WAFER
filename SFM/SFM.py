from scipy import interpolate
from scipy.optimize import least_squares
import numpy as np 

#### Python adapted from https://gitlab.com/ltda/flox-specfit

def SpecFit_funct(x,rad_d,wvl,inp,knots):

    """ 
    Input:

    x: fit parameters, i.e. peak heights of fluorescence and heights of all reflectance knots
    L0: downwelling radiance
    wvl: corresponding wavelength array
    inp: upwelling radiance
    knots: wavelength positions of the knots

    Returns:

    Difference of modelled and input radiance (this is what is going to be minimized)
    """

    #### reflectance model ####

    knotvals   = x[2:]
    interp = interpolate.UnivariateSpline(knots,knotvals,s=0)
    refl         = interp(wvl) 

    #### fluorescence model ####

    # red peak
    lorentz_x  = (wvl - 684)/10.
    f_red = x[0]/(lorentz_x**2 + 1.)

    # far red peak
    lorentz_x       = (wvl - 735)/25.
    f_fred = x[1]/(lorentz_x**2 + 1.)

    # full fluorescence, modulated by reflectance
    fluo = np.multiply(f_fred + f_red,refl)

    # forward model upwelling radiance
    rad_u_m = np.multiply(refl,rad_d) + fluo

    # return the residual (to be minimized):
    return rad_u_m-inp


def SpecFit(wvl,rad_d,rad_u,fg_peak,w,owl,numknots,alg = 'lm'):

    """ 
    Input: 
    
    wvl: wavelength-array, 
    rad_d: downwelling radiance, 
    rad_u: upwelling radiance,
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

    # Apparent Reflectance 
    app_refl          = np.divide(rad_d,rad_u)
    
    # create mask to exclude O2 bands
    mask = np.logical_and(np.logical_or(wvl < 686, wvl > 690),np.logical_or(wvl < 759, wvl > 768))
    mask = np.array(mask)                                                                      
    mask = mask > 0
    
    wvl = np.array(wvl)
    wvlnoO2 = wvl[mask]
    app_reflnoO2 = app_refl[mask]
    
    # knots vector for piecewise spline
    inds = np.linspace(1,len(wvlnoO2)-2,numknots,dtype=int) 
    knots = wvlnoO2[inds]
    
    # piece wise cubic spline
    sp = interpolate.LSQUnivariateSpline(wvlnoO2,app_reflnoO2,knots)
    p_r = sp(wvl)[inds]
    
    # first guesses of peak values and reflectance knots:
    x0 = [fg_peak[0],fg_peak[1]]
    for val in p_r:
        x0.append(val)
    
    # weight for the upwelling radiance (if any)
    rad_u_w = np.multiply(rad_u,w)   # with weight
    
    # optimization part: least squares minimization of the residual returned by SpecFit_funct:
    if alg == 'trf':
        res = least_squares(SpecFit_funct,x0,method='trf',max_nfev=100,args=(rad_d,wvl,rad_u_w,knots))     
        x, resnorm,residual,exitflag,nfevas = res.x,res.cost,res.fun,res.status,res.nfev
        if exitflag == -1:
            resnorm = np.NaN
    elif alg == 'lm':
        res = least_squares(SpecFit_funct,x0,method='lm',max_nfev=6,args=(rad_d,wvl,rad_u_w,knots))    
        x, resnorm,residual,exitflag,nfevas = res.x,res.cost,res.fun,res.status, res.nfev
    else:
        print('Check Optimization algorithm')
        exit(0)
          
    # optimized spectra for output:

    #### reflectance model #### 
    knotvals = x[2:]
    interp = interpolate.UnivariateSpline(knots,knotvals,s=0)
    refl = interp(owl)
    
    #### fluorescence model ####

    # red peak
    lorentz_x = (owl - 684)/10.
    f_red = x[0]/(lorentz_x**2 + 1.)
    # far red peak
    lorentz_x = (owl - 735)/25.
    f_fred = x[1]/(lorentz_x**2 + 1.)

    # fluorescence spectrum, modulated with reflectance
    fluo = np.multiply(f_fred + f_red,refl)

    # final at-sensor modeled radiance
    rad_u_m        = SpecFit_funct(x,rad_d,wvl,rad_u_w,knots)

    # summary statistics
    residual     = rad_u_m - rad_u
    rmse         = np.sqrt(np.mean(np.square(rad_u_m - rad_u)))
    rrmse        = np.sqrt(np.mean(np.square(np.divide((rad_u_m - rad_u),rad_u))))*100. 

    return x, fluo, refl, resnorm, exitflag, nfevas, residual