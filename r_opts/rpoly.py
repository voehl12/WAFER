import numpy as np 
from scipy import optimize
from scipy import interpolate
from utils import wavelets




def optimize_coeffs(wl:np.array,ref:np.array, signal:np.array, initial_guess:np.array,sigdecomp,lbl=1,low_init=0) -> (np.array,np.array):
    """
    Parameters
    ---------
    wl: ndarray
        wavelength array over which R should be derived
    
    ref: ndarray
        reference radiance, same length as wl
    signal: ndarray
        radiance including SIF, same lenght as wl
    
    inital_guess: ndarray
            initial guess of polynomial coefficients for reflectance (usually obtained from apparent reflectance)

    sigdecomp: class member of utils.wavelets.decomp

    optional: lbl: 1 if optimization is performed level-wise, 0 if the optimization should be performed on the entire decomposition (not recommended)
                low_init: 1 if the initial guess should be below the expected reflectance
    Returns
    -------
    results: ndarray
            array containing optimal reflectance polynome coefficients for each level
    ress: ndarray
            residuals of initial guess and for each level according to difference function
    """

    def diff_func_poly(coeffs,*args):
        # get current reflectance:
        interp = np.poly1d(coeffs)
        refl = interp(wl)
        level = args[0].optlevel
        masks = args[0].masks
        scales = args[0].scales
        
        if len(args) == 2:
            diff = wavelets.create_decomp_p(np.multiply(ref,refl),scales,level='all') - args[0].comps
            diff = [np.divide(diff[i],scales[i]**0.5)[masks[i].mask] for i in range(len(level))]
            squaredsum = sum([ele for sub in np.square(diff)*np.sqrt(np.square(ref)+ np.square(signal)) for ele in sub])

        else:
            # option if residual is calculated level by level (default)
            
            diff = wavelets.create_decomp_p(np.multiply(ref,refl),scales,level) - args[0].comps[level]
            #subtracting noise for both reference and signal on all scales would not change anything for the optimization
            #diff = diff/scales[level]**0.5 # normalization not really necessary for optimization 
            diff = diff[0,masks[level].mask]
            squaredsum = np.sum(np.square(diff))

              
    
        res = np.sqrt(squaredsum)


        return res

    # create decomposition and masks of the signal
    sigdecomp.create_comps(signal)
    sigdecomp.calc_mask(signal)
 
    # initialize and set the boundary conditions:      
    lowerBound = np.full((len(initial_guess)),-np.inf)
    upperBound = np.full((len(initial_guess)),np.inf)

    # these bounds depend on whether the last coefficient of the polynomial has been pushed down as initial guess or not! if yes (p_init[-1] = initial_guess[-1] - 0.3), the first set is correct.
    if low_init == 1:
        initial_guess[-1] -= 0.3
        upperBound[-1] = initial_guess[-1]+ 0.2999
        lowerBound[-1] = initial_guess[-1]

    else:
        upperBound[-1] = initial_guess[-1]-0.0000001
        lowerBound[-1] = initial_guess[-1]-0.2
    

    parameterBounds = optimize.Bounds(lowerBound,upperBound)
    
    # run the optimization:
    results = []
    ress = np.ones((len(sigdecomp.scales),2))
    if lbl == 1:

        for i in range(len(sigdecomp.scales)):
            sigdecomp.optlevel = i
            ress[i,0] = diff_func_poly(initial_guess,sigdecomp)
            result_nm = optimize.minimize(diff_func_poly,initial_guess, bounds=parameterBounds, args=sigdecomp)
            ress[i,1] = result_nm.fun
            results.append(result_nm.x)
    else: 
        kwargs = sigdecomp,0
        result_nm = optimize.minimize(diff_func_poly,initial_guess, bounds=parameterBounds, args=kwargs)
        results = result_nm.x
        # todo: add functionality to return proper residuals for this case as well

    return np.array(results),ress

