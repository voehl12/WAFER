import numpy as np 
from scipy import optimize
from scipy import interpolate
from utils import wavelets


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






def optimize_coeffs(wl,ref, signal, knots, initial_guess,scales,order,lbl=0):
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

        level = args[0]
        ncoeffs = np.zeros(len(B))
        ncoeffs[:len(coeffs)] = coeffs
        refl = bspleval(ncoeffs,B) 
  
       
        diff = wavelets.create_decomp_p(np.multiply(ref,refl),scales,level) - sigdecomp[level]
        #subtracting noise for both reference and signal on all scales would not change anything for the optimization
        
        if isinstance(level,int):
            diff = diff/scales[level]**0.5
            diff = diff[0,masks[level].mask]
            squaredsum = np.sum(np.square(diff))
            #squaredsum = np.sum(np.square(diff)*np.sqrt(np.square(ref[masks[level].mask])+ np.square(signal[masks[level].mask])))
            
        else:
            diff = [np.divide(diff[i],scales[i]**0.5)[masks[i].mask] for i in range(len(level))]
            squaredsum = sum([ele for sub in np.square(diff)*np.sqrt(np.square(ref)+ np.square(signal)) for ele in sub])
            
    
        res = np.sqrt(squaredsum)


        return res
    N = len(wl)
    # initiate B-matrix:


    B = setup_B(wl,knots,order)
    nz_inds = np.nonzero(np.array(initial_guess))
    
    appref = np.divide(signal,ref)
    nz_initial = np.array(initial_guess)[nz_inds]
    nz_initial_up = nz_initial
    nz_initial_down = nz_initial-0.5
    initial_spline = bspleval(initial_guess,B)
    noisedecomp = wavelets.create_decomp_p(signal-np.multiply(ref,initial_spline),scales)

    coef_inds = [np.argmin(np.fabs(wl-knots[i])) for i in range(len(nz_initial))]

    sigdecomp = wavelets.create_decomp_p(signal,scales)
    
    
    diff_func = diff_func_spl
    lowerBound = np.zeros(len(nz_initial))
    lowerBound[0:2] = nz_initial[0:2]-0.00002
    lowerBound[3:-1] = nz_initial[3:-1]-0.3
    lowerBound[-1] = nz_initial[-1]-0.00002

    parameterBounds = optimize.Bounds(lowerBound,nz_initial_up)
    
    
    
    masks = []
    for i in range(len(scales)):
        negmask = np.ma.masked_where(sigdecomp[i] <= 0, sigdecomp[i])
        #print(-np.median(np.fabs(sigdecomp[i,negmask.mask]))/0.6745)
        levelmask = np.ma.masked_where(sigdecomp[i] <= -np.median(np.fabs(sigdecomp[i,negmask.mask]))/0.6745, sigdecomp[i])
        #print(len(sigdecomp[i,levelmask.mask]))
        masks.append(np.ma.masked_where(sigdecomp[i] <= -np.median(np.fabs(sigdecomp[i,negmask.mask]))/0.6745, sigdecomp[i]))
    
    # thresholding: wavelet tour, page 565
    

    
  
    results = []
    if lbl == 1:

        for i in range(len(scales)):
            kwargs = (i)

            result_nm = optimize.minimize(diff_func,nz_initial_down, bounds=parameterBounds, args=kwargs)
            #print(result_nm.x)
            results.append(result_nm.x)
    else: 
        kwargs = range(len(scales))
        result_nm = optimize.minimize(diff_func,nz_initial, bounds=parameterBounds, args=kwargs)
        results = result_nm.x

    return np.array(results)

def determine_weights(wl,knots,signal,scales):
    nz_knots = knots[np.nonzero(knots)]
   
    coef_inds = [np.argmin(np.fabs(wl-knots[i])) for i in range(len(nz_knots))]

    sigdecomp = wavelets.create_decomp_p(signal,scales)
    

    
    
    
    masks = []
    for i in range(len(scales)):
        negmask = np.ma.masked_where(sigdecomp[i] <= 0, sigdecomp[i])
        #print(-np.median(np.fabs(sigdecomp[i,negmask.mask])))
        levelmask = np.ma.masked_where(sigdecomp[i] <= -np.median(np.fabs(sigdecomp[i,negmask.mask]))/0.6745, sigdecomp[i])
        #print(len(sigdecomp[i,levelmask.mask]))
        masks.append(np.ma.masked_where(sigdecomp[i] <= -np.median(np.fabs(sigdecomp[i,negmask.mask]))/0.6745, sigdecomp[i]))
    
    # thresholding: wavelet tour, page 565
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
            #print(scalecounts)
          
            contrib_counts[j,i] += scalecounts
    
    return contrib_counts




def adjust_smoothing(wl,refl,initsmoothing,numknots,order):
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






