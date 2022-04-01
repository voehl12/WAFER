from bz2 import decompress
from scipy import interpolate
from scipy import optimize
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from ssqueezepy import ssq_cwt, ssq_stft,cwt,icwt,issq_cwt
import pywt


def prepare_arrays(cab,lai,feffef):
    completename = '../cwavelets/libradtranscope/floxseries_ae_oen/reflectance/radcomplete_{}_{:d}_{:d}_ae_conv.dat'.format(feffef,cab,lai)
    woFname = '../cwavelets/libradtranscope/floxseries_ae_oen/reflectance/radwoF_{}_{:d}_{:d}_ae_conv.dat'.format(feffef,cab,lai)
    reflname = '../reflectance/szamatch/rho_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    albedoname = '../reflectance/szamatch/albedo_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    scoperef = '../LupSCOPE/szamatch/Lup_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    Fname = '../cwavelets/libradtranscope/floxseries_ae_oen/reflectance/Fcomp_{}_{:d}_{:d}_ae.dat'.format(feffef,cab,lai) #'fluorescence/F_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    
    wlRname = '../reflectance/szamatch/wlR'
    wlFname = '../reflectance/szamatch/wlF'
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




def refl_tanh(x,a,b,c,d,e,f):
    xn = b * x + c 
    return a * ((np.exp(e * xn) - np.exp(-f * xn)) / (np.exp(xn) + np.exp(-xn))) + d   

def refl_logistic(w,a,b,c,d):
    # logistic plus linear contribution from soil
        wl0 = wl[0]
        return a*w + b/(1+c*np.exp(-d*(w-wl0)))

def refl_polynomial(w,*coeffs):
    
    interp = np.poly1d(coeffs)
    return interp(w)

def refl_piecewise(w,wls,vals,order):
    if order == 1:
        kind = 'linear'
    elif order == 2:
        kind = 'quadratic'
    elif order == 3:
        kind = 'cubic'
    else:
        print('Specify order 1, 2 or 3!')
        exit(0)
    interp = interpolate.interp1d(wls,vals,kind=kind)
    return interp(w)

def refl_legendre(w,*coef):
    legpoly = np.polynomial.legendre.Legendre(coef)
    return(legpoly(w))



def diff_func_exp(params,*args):
        R = fitfunc(wl,*params)
        diff = ((create_decomp_p(np.multiply(whiteref,R)-signal,jmin,jmax,nlevel,args[0])))
        return np.sqrt(np.sum((np.square(diff))/scale[args[0]]))

def diff_stats(R,fit):
    diff = R-fit
    rmse = np.sqrt(np.mean(np.square(diff)))
    rrmse = np.sqrt(np.mean(np.square(np.divide(diff,R))))
    return diff,rmse,rrmse

def create_decomp_p(data,scales,selscales='all'):
   
            
    data = np.array(data)

    
    j = scales
    
  
    

    coef,ssqscales = cwt(data,wavelet='cmhat',scales=j)
    sellevel = np.real(coef)[selscales]

    return sellevel

def residual(Fguess,level,scales):
               
        diff = create_decomp_p(Fguess,scales,level)
        
        return (np.sum((np.square(diff))/scales[level]))


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
    m = np.alen(t)
    npts = np.alen(x)
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

""" Rshape = 'poly'
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
    p_init = np.polyfit(wl,appref,5) """


wl = []
with open('../cwavelets/libradtranscope/floxseries_ae_oen/radcomplete_004_5_7_ae_conv.dat','r') as g:
    for line in g:
        line = line.split()
        wl.append(float(line[0]))
    wl = np.array(wl)

refname = '../cwavelets/libradtranscope/floxseries_ae_oen/whiteref_ae_conv.dat'
whitereference = []
with open(refname,'r') as wf:
    for k,line in enumerate(wf):
        line = line.split()
        whitereference.append(float(line[0]))
whitereference = np.array(whitereference)

sifeffecs = ['002']
  
CAB = [int(5),int(10)]
for i in range(2,8):
    CAB.append(i*10) 
#CAB = [int(10)]
LAI = [1,2,3,4,5,6,7]
my_cmap = cm.get_cmap('viridis', len(CAB)*len(LAI)*len(sifeffecs))
l = 0
fitfunc = refl_polynomial
#p_init = [0.5,1,-730,0.2,1,1]
fig,(ax1,ax2) = plt.subplots(2)
resfig,resax = plt.subplots()
rrmses = []
rmses = []
F_meandiffs = []
F_diffs = []
for sifeffec in sifeffecs:
    for cab in CAB:
        for lai in LAI:
            print(cab,lai)
            arrays = prepare_arrays(cab,lai,sifeffec)
            atsensornonoise = arrays[0]

            reflectance = arrays[2]
            wlR = arrays[5]
            interR = interpolate.interp1d(wlR, reflectance,kind='cubic')
            R = interR(wl)

            apparentrefl = np.divide(atsensornonoise,whitereference)
            startind = np.argmin(np.fabs(wl-675))
            endind = np.argmin(np.fabs(wl-800))
            R = R[startind:endind]
            apparentrefl = apparentrefl[startind:endind]
            atsensornonoise = atsensornonoise[startind:endind]
            wlnew = wl[startind:endind]
            whiterefnew = whitereference[startind:endind]

            ainit = (apparentrefl[20]-apparentrefl[0])/(wlnew[20]-wlnew[0])
            binit = apparentrefl[-1]-ainit*wlnew[-1]
            cinit = binit/(apparentrefl[0]-ainit*wlnew[0]) - 1
            #p_init = [ainit,binit,cinit,0.0001]
            p_init = np.polyfit(wlnew,apparentrefl,7)
            #p_init = np.polynomial.legendre.legfit(wlnew,apparentrefl,5)
            minpeak = np.argmin(np.fabs(wlnew-757))
            maxpeak = np.argmin(np.fabs(wlnew-768))
            nopeak_appref = []
            nopeak_wl = []
            for i in range(len(wlnew)):
                if i < minpeak or i > maxpeak:
                    nopeak_appref.append(apparentrefl[i])
                    nopeak_wl.append(wlnew[i])
            initsmooth = 0.001
            interRa_smooth = interpolate.UnivariateSpline(nopeak_wl,nopeak_appref,s=initsmooth,k=2)
            smooth_knots = interRa_smooth.get_knots()
            

            smooth_appref = interRa_smooth(wlnew)

            
            #testfitparams,_ = optimize.curve_fit(fitfunc,wlnew,smooth_appref,p0=p_init)
            """ R_fitparams, fitcov = optimize.curve_fit(fitfunc,wlnew,R,p0=p_init)
            paramerrors = np.sqrt(np.diag(fitcov))
            R_final = fitfunc(wlnew,*R_fitparams)  """


            interRa, smoothing = adjust_smoothing(wlnew,R,initsmooth,10,11,2)
            plt.figure()
            plt.plot(wlnew,interRa(wlnew),label='Original Spline')
            #R_final = refl_piecewise(wlnew,interRa.get_knots(),interR(interRa.get_knots()),3)
            print(interRa._data[9])
            print(len(interRa._eval_args))
            print(len(interRa.get_knots()))
            splinecoeffs = interRa._data[9]
            knots = interRa._data[8]
            change = (np.random.rand(len(splinecoeffs))-0.5)/10
            splinecoeffs += change
            Rnew = bspleval(wlnew, knots, splinecoeffs, 2, debug=False)
            plt.plot(wlnew,Rnew,label='Changed Coefficients')
            plt.legend()
            plt.show()
            R_final = interRa(wlnew)
            shiftedknots = interR(interRa.get_knots())
            shiftedknots[-2] -= 0.03
            shiftedknots[-6] -= 0.1
            R_shiftknot = refl_piecewise(wlnew,interRa.get_knots(),shiftedknots,2)
            
            
            ax1.plot(wlnew,R,c=my_cmap(l))
            ax1.plot(wlnew,R_final,color=my_cmap(l))
            #ax1.plot(interRa.get_knots(),interR(interRa.get_knots()),'o',c=my_cmap(l))
            diff,rmse,rrmse = diff_stats(R,R_final)
            Fdiff = np.multiply(whiterefnew,diff)
            F_diffs.append(Fdiff)
            F_meandiff = np.sqrt(np.mean(np.square(Fdiff)))
            F_meandiffs.append(F_meandiff)
            #print(rmse)
            rmses.append(rmse)
            rrmses.append(rrmse)

            ax2.plot(wlnew,Fdiff,color=my_cmap(l))

            nlevel = 20
            jmin = -0.5
            #jmin = 0.5
            jmax = 8.0
            scales = np.logspace(jmin,jmax,num=nlevel,base=2)
            f = pywt.scale2frequency('gaus2', scales)
            wlscales = 1/f*0.17
            decompres = []
            resminus = []
            resplus = []
            resapps = []
            for i in range(len(scales)):
                res = residual(atsensornonoise-np.multiply(R_final,whiterefnew),i,scales)
                resapp = residual(atsensornonoise-np.multiply(apparentrefl,whiterefnew),i,scales)
                resup = residual(atsensornonoise-np.multiply(R_final+0.01,whiterefnew),i,scales)
                resdown = residual(atsensornonoise-np.multiply(R_final-0.01,whiterefnew),i,scales)
                decompres.append(res)
                resminus.append(resdown)
                resplus.append(resup)
                resapps.append(resapp)

                
            #resax.plot(scales,decompres,color=my_cmap(l))
            decompres, resminus, resplus = np.array(decompres), np.array(resminus), np.array(resplus)
            resax.plot(scales,resminus-decompres,':',color=my_cmap(l),label='CAB {:d}, LAI {:d}'.format(cab,lai))
            #resax.plot(scales,resplus-decompres,'--',color=my_cmap(l))
            #resax.plot(scales,resapps,'-.',color=my_cmap(l))
            resax.set_xscale('log',base=2)




            l += 1
print(np.mean(rmses))
print(np.mean(F_meandiffs))
ax2.plot(wlnew,np.mean(F_diffs,axis=0),color='red')
ax2.fill_between(wlnew, np.mean(F_diffs,axis=0)-np.std(F_diffs,axis=0), np.mean(F_diffs,axis=0)+np.std(F_diffs,axis=0), alpha = 0.5,color='red')
resax.legend()
plt.show()