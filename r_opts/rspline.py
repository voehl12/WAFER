import numpy as np 
import pywt
from ssqueezepy import cwt
from scipy import optimize
from scipy import interpolate


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

def create_decomp_p(data,scales,level):
   
            
    data = np.array(data)

    
    
    j = scales[level]
        
    coef,freqs = pywt.cwt(data,j,'gaus2',method='fft')
    #coef,ssqscales = cwt(data,wavelet='cmhat',scales=j)
    
    return coef




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

    def diff_func(coeffs,*args):

        level = args[0]
        ncoeffs = np.zeros(len(B))
        ncoeffs[:len(coeffs)] = coeffs
        refl = bspleval(ncoeffs,B) 
  
       
        diff = create_decomp_p(np.multiply(ref,refl)-signal,scales,level)
        res = np.sqrt(np.sum(np.square(diff)))


        return res
    nz_inds = np.nonzero(np.array(initial_guess))
    
    appref = np.divide(signal,ref)
    nz_initial = np.array(initial_guess)[nz_inds]

    coef_inds = [np.argmin(np.fabs(wl-knots[i])) for i in range(len(nz_initial))]

    # initiate B-matrix:
    B = setup_B(wl,knots,order)

    lowerBound = np.zeros(len(nz_initial))
    lowerBound[:-1] = 0.8*nz_initial[:-1]
    lowerBound[-1] = 0.99*nz_initial[-1]

    parameterBounds = optimize.Bounds(lowerBound,0.99*nz_initial)
    results = []
    if lbl == 1:

        for i in range(len(scales)):
            kwargs = (i)

            result_nm = optimize.minimize(diff_func,nz_initial, bounds=parameterBounds, args=kwargs)

            results.append(result_nm.x)
    else: 
        kwargs = range(len(scales))
        result_nm = optimize.minimize(diff_func,nz_initial, bounds=parameterBounds, args=kwargs)
        results = result_nm.x

    return results


def adjust_smoothing(wl,refl,initsmoothing,minknots,maxknots,order):
    smoothing = initsmoothing
    interRa = interpolate.UnivariateSpline(wl,refl,s=smoothing,k=order)
    initknots = interRa.get_knots()
    
    if len(initknots) < minknots:
        while len(interRa.get_knots()) < minknots:
            step = smoothing / 10
            smoothing -= step
            interRa = interpolate.UnivariateSpline(wl,refl,s=smoothing,k=order)
    
    elif len(initknots) > maxknots:
        while len(interRa.get_knots()) > maxknots:
            step = smoothing / 10
            smoothing += step
            interRa = interpolate.UnivariateSpline(wl,refl,s=smoothing,k=order)

    #print('Final Smoothing: {:.7f}'.format(smoothing))
    return interRa,smoothing






