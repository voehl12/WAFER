import numpy as np 
from scipy import optimize
from scipy import interpolate
from utils import wavelets

"""
functions to determine the reflectance from given down- and upwelling radiance as a piece-wise spline (linear, quadratic or cubic can be chosen)
currently not the default for the wavelet method as the spline has too many free parameters and is less robust. Functions probably need some adjustment.

"""


def setup_B(x,knots,order):

    '''
    Set up a B-spline matrix.

    Parameters
    ----------
    x : list or ndarray
        The set of points at which to evaluate the spline.
    knots : list or ndarray
        The set of knots used to define the spline.
    order : int
        The order of the spline.

    Returns
    -------
    B : ndarray
        The B-matrix to be multiplied with spline coefficients.
    '''

    k = order
    t = knots
    m = np.alen(t)
    npts = np.alen(x)
    B = np.zeros((m-1,k+1,npts))

    
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

    return B




def bspleval(coeffs,B):
    '''
    Evaluate a B-spline at a set of points.

    Parameters
    ----------
    coeffs : list of ndarray
        The set of spline coefficients.
    B : ndarray
       The basis matrix of the B-spline.

    Returns
    -------
    y : ndarray
        The value of the spline at each point in x.
    '''

 
    m = len(B) + 1
    k = len(B[0]) - 1
    npts = len(B[0,0])


    ## Evaluate the spline by multiplying the coefficients with the highest-order basis functions.
    y = np.zeros(npts)
    for i in range(len(B)):
        y += coeffs[i] * B[i,k,:]

    return y






def optimize_coeffs(wl,ref, signal, knots, initial_guess,sigdecomp,order,lbl=0,low_init=0):
    """
    Parameters
    ---------
    wl: ndarray
        wavelength array over which R should be derived
    ref_decomp: ndarray
                Wavelet decomp of reference (levels to analyze)
    signal_decomp: ndarray
                Same for signal
    knots: ndarray
            Spline knots as wl positions
    inital_guess: ndarray
            initial guess of spline coefficients (i.e. knot heights)
    Returns
    -------
    coeffs: ndarray
    """

    

    def diff_func_spl(coeffs,*args):

        level = args[0].optlevel
        ncoeffs = np.zeros(len(B))
        ncoeffs[:len(coeffs)] = coeffs
        refl = bspleval(ncoeffs,B) 
        masks = args[0].masks
        scales = args[0].scales
  
       
        diff = wavelets.create_decomp_p(np.multiply(ref,refl),scales,level) - args[0].comps[level]
        #subtracting noise for both reference and signal on all scales would not change anything for the optimization
        
        if isinstance(level,int):
            diff = diff/scales[level]**0.5
            diff = diff[0,masks[level].mask]
            squaredsum = np.sum(np.square(diff))
            
            
        else:
            diff = [np.divide(diff[i],scales[i]**0.5)[masks[i].mask] for i in range(len(level))]
            squaredsum = sum([ele for sub in np.square(diff)*np.sqrt(np.square(ref)+ np.square(signal)) for ele in sub])
            
    
        res = np.sqrt(squaredsum)


        return res
    N = len(wl)
    # initiate B-matrix:


    B = setup_B(wl,knots,order)
    nz_inds = np.nonzero(np.array(initial_guess))
    
   
    nz_initial = np.array(initial_guess)[nz_inds]
    nz_initial_up = nz_initial
    nz_initial_down = nz_initial-0.5
    initial_spline = bspleval(initial_guess,B)
    if low_init == 1:
        initial_guess = nz_initial_down
        lowerBound = initial_guess
        upperBound = nz_initial
    else: 
        initial_guess = nz_initial_up
        lowerBound = np.zeros(len(initial_guess))
        lowerBound[0:2] = initial_guess[0:2]-0.00002
        lowerBound[3:-1] = initial_guess[3:-1]-0.3
        lowerBound[-1] = initial_guess[-1]-0.00002
        upperBound = initial_guess
    
  
    # create decomposition and masks of the signal
    sigdecomp.create_comps(signal)
    sigdecomp.calc_mask(signal)
    
    parameterBounds = optimize.Bounds(lowerBound,upperBound)
    
    # run the optimization:
    results = []
    ress = np.ones((len(sigdecomp.scales),2))
    if lbl == 1:

        for i in range(len(sigdecomp.scales)):
            sigdecomp.optlevel = i
            ress[i,0] = diff_func_spl(initial_guess,sigdecomp)
            result_nm = optimize.minimize(diff_func_spl,initial_guess, bounds=parameterBounds, args=sigdecomp)
            ress[i,1] = result_nm.fun
            results.append(result_nm.x)
    else: 
        kwargs = sigdecomp,0
        result_nm = optimize.minimize(diff_func_spl,initial_guess, bounds=parameterBounds, args=sigdecomp)
        results = result_nm.x
        # todo: add functionality to return proper residuals for this case as well

    return np.array(results),ress

def determine_weights(wl,knots,signal,scales):
    """
    function to determine knot position dependent weights - currently not implemented in the optimization

    """
    nz_knots = knots[np.nonzero(knots)]
   
    coef_inds = [np.argmin(np.fabs(wl-knots[i])) for i in range(len(nz_knots))]

    sigdecomp = wavelets.create_decomp_p(signal,scales)
    

    
    
    
    masks = []
    for i in range(len(scales)):
        negmask = np.ma.masked_where(sigdecomp[i] <= 0, sigdecomp[i])
        
        levelmask = np.ma.masked_where(sigdecomp[i] <= -np.median(np.fabs(sigdecomp[i,negmask.mask]))/0.6745, sigdecomp[i])
   
        masks.append(np.ma.masked_where(sigdecomp[i] <= -np.median(np.fabs(sigdecomp[i,negmask.mask]))/0.6745, sigdecomp[i]))
    
    
    contrib_counts = np.ones((len(scales),len(nz_knots)))
    for i,val in enumerate(coef_inds):
        if i == 0:
            rangemin = 0
            rangemax = val + (coef_inds[i+1] - val)//2
        elif i == len(coef_inds)-1:
            rangemin = val - (val-coef_inds[i-1])//2
            rangemax = val
        else:
            rangemin = val - (val-coef_inds[i-1])//2
            rangemax = val + (coef_inds[i+1] - val)//2
        
        for j,scale in enumerate(scales):
            scalecounts = 0
            for k in masks[j][rangemin:rangemax].mask:
                if k == True:
                    scalecounts += 1
            
          
            contrib_counts[j,i] += scalecounts
    
    return contrib_counts




def adjust_smoothing(wl,refl,initsmoothing,numknots,order):
    """
    function to determine initial piece wise spline for a given number of knots (usually used to determine initial guess)
    """
    smoothing = initsmoothing
    interRa = interpolate.UnivariateSpline(wl,refl,s=smoothing,k=order)
    initknots = interRa.get_knots()

    while len(interRa.get_knots()) != numknots:
        if len(interRa.get_knots()) < numknots:
            #step = smoothing / 10
            while len(interRa.get_knots()) < numknots:
                
                smoothing *= 0.999
                interRa = interpolate.UnivariateSpline(wl,refl,s=smoothing,k=order)
    
        elif len(interRa.get_knots()) > numknots:
            #step = smoothing / 10
            while len(interRa.get_knots()) > numknots:
                
                smoothing *= 1.001
                interRa = interpolate.UnivariateSpline(wl,refl,s=smoothing,k=order)

    #print('Final Smoothing: {:.7f}'.format(smoothing))
    return interRa,smoothing






