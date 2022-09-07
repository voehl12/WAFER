import numpy as np 
from scipy import optimize
from scipy import interpolate
from utils import wavelets




def optimize_coeffs(wl,ref, signal, initial_guess,scales,lbl=0):
    """
    Parameters
    ---------
    wl: ndarray
        wavelength array over which R should be derived
    ref_decomp: ndarray
                Wavelet decomp of reference (levels to analyze)
    signal_decomp: ndarray
                Same for signal
    
    inital_guess: ndarray
            initial guess of polynomial coefficients 
    Returns
    -------
    coeffs: ndarray
    """

    def diff_func_poly(coeffs,*args):
        level = args[0]
        interp = np.poly1d(coeffs)
        refl = interp(wl)
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

    sigdecomp = wavelets.create_decomp_p(signal,scales)
    
 
    diff_func = diff_func_poly
    nz_initial = initial_guess
    
    lowerBound = np.full((len(nz_initial)),-np.inf)
    upperBound = np.full((len(nz_initial)),np.inf)

    # these bounds depend on whether the last coefficient of the polynomial has been pushed down as initial guess or not! if yes (p_init[-1] = p_init[-1] - 0.3), the first set is correct.
    #upperBound[-1] = nz_initial[-1]+ 0.2999
    #lowerBound[-1] = nz_initial[-1]
    upperBound[-1] = nz_initial[-1]-0.0000001
    lowerBound[-1] = nz_initial[-1]-0.2
    
 
    """upperBound[-2] = nz_initial[-2]+0.0001
    lowerBound[-2] = nz_initial[-2]-0.0001
    if len(nz_initial) > 2:

        upperBound[-3] = nz_initial[-3]+0.0001
        lowerBound[-3] = nz_initial[-3]-0.0001"""
    parameterBounds = optimize.Bounds(lowerBound,upperBound)
    masks = []
    for i in range(len(scales)):
        negmask = np.ma.masked_where(sigdecomp[i] <= 0, sigdecomp[i])
        #print(-np.median(np.fabs(sigdecomp[i,negmask.mask]))/0.6745)
        levelmask = np.ma.masked_where(sigdecomp[i] <= -np.median(np.fabs(sigdecomp[i,negmask.mask]))/0.6745, sigdecomp[i])
        #print(len(sigdecomp[i,levelmask.mask]))
        masks.append(np.ma.masked_where(sigdecomp[i] <= -np.median(np.fabs(sigdecomp[i,negmask.mask]))/0.6745, sigdecomp[i]))
    
    # thresholding: wavelet tour, page 565
    

    
  
    results = []
    ress = np.ones((len(scales),2))
    if lbl == 1:

        for i in range(len(scales)):
            kwargs = (i)
            ress[i,0] = diff_func_poly(nz_initial,i)
            result_nm = optimize.minimize(diff_func,nz_initial, bounds=parameterBounds, args=kwargs)
            ress[i,1] = result_nm.fun
            #print(result_nm.x)
            results.append(result_nm.x)
    else: 
        kwargs = range(len(scales))
        result_nm = optimize.minimize(diff_func,nz_initial, bounds=parameterBounds, args=kwargs)
        results = result_nm.x

    return results,ress

