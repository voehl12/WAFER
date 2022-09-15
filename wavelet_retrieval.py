import wave
import numpy as np
from scipy import interpolate, signal
from r_opts import rspline,rtanh,rpoly
from ih import prepare_input
import matplotlib.pyplot as plt
import scipy
from utils import wavelets,plotting, funcs
from SFM import SFM,SFM_BSpline
from scipy import optimize
from matplotlib import rc
#plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

def gauss(N,sigma,A,b):
    x = np.arange(-N//2,N//2)
    x = x-b
    y = -A*np.exp(-x*x/2/sigma**2)
    return y
noise = 1
wlorig, signalorig, whitereferenceorig, reflorig, Forig = prepare_input.synthetic(5,2,'002',completedir='../cwavelets/libradtranscope/floxseries_ae_oen/brdf/')
wlbrdf, signalbrdf, whitereferenceorig, reflorig, Forig = prepare_input.synthetic(5,2,'002',completedir='../cwavelets/libradtranscope/floxseries_ae_oen/brdf/')

plt.figure()
plt.plot(wlorig,signalorig/whitereferenceorig)
plt.show()

#wlorig, signalorig, whitereferenceorig,ifldaref,merrororig,time,downerrors = prepare_input.flox('10:30:00',wlmin=670)
wl, upsignal, whitereference, refl, F = prepare_input.synthetic(5,2,'002',wlmin=745,wlmax=755,completedir='../cwavelets/libradtranscope/floxseries_ae_oen/brdf/')

#wl, upsignal, whitereference,ifldaref,merror,time,downerrors = prepare_input.flox('10:30:00',wlmin=745,wlmax=755)
sunspectrum,gaussf = prepare_input.match_solspec(wl,0.3)
""" peakss, _ = signal.find_peaks(-whitereference)
print(len(wl))
fwhms = []
plt.figure()
for i,n in enumerate(peakss[3:-3]):
    sigma,gausscov = optimize.curve_fit(gauss,5,whitereference[n-2:n+3]-whitereference[n-2],p0=[2,5,1])
    fwhm = sigma[0]*2.355*(wl[1]-wl[0])
    fwhms.append(fwhm)
    plt.plot(wl[n-2:n+3],gauss(5,*sigma))
plt.figure()
plt.plot(wl[peakss[3:-3]],fwhms,'.')
plt.show()
#find fwhm for each peak and apply this structure to simulated spectra

plt.figure()
plt.plot(wl,whitereference-whitereference[0])
plt.plot(wl,gauss(len(wl),*fwhm))
plt.show() """
refnoise, whitereference = prepare_input.add_noise(whitereference,1000,noise,N=7)
var = whitereference/1000
noisedecomp = wavelets.create_decomp_p(refnoise,np.logspace(-3,15,64,base=2))
noisescales = np.logspace(-3,15,64,base=2)
totnoise = [np.mean(np.square(noisedecomp[i])) for i in range(len(noisescales))]

sensornoise, upsignal = prepare_input.add_noise(upsignal,1000,noise,N=7)
appref = np.divide(upsignal,whitereference)
signal_m = upsignal-np.mean(upsignal)
plt.figure()
#plt.plot(wl,whitereference)
plt.plot(wlorig,whitereferenceorig)
#plt.fill_between(wlorig,whitereferenceorig-np.multiply(merrororig,whitereferenceorig),whitereferenceorig+np.multiply(merrororig,whitereferenceorig),color='tab:red',alpha=0.5)
whitereference_sm = whitereference.copy()
#whitereference = prepare_input.deconvolve(wl,whitereference)
#plt.plot(wl,whitereference)
plt.show()

plt.figure()
plt.plot(wl,upsignal)
plt.plot(wl,whitereference)
signal_sm = upsignal.copy()
#signal = prepare_input.deconvolve(wl,signal)
plt.plot(wl,upsignal)
wname = '../../Code/cwavelets/libradtranscope/series/wl_array'
wlfine = []
with open(wname,'r') as wf:
    for line in wf:
        line = line.split()
        wlfine.append(float(line[0]))
fname = '../../Code/cwavelets/libradtranscope/floxseries_ae_oen/reflectance/radcomplete_{}_{:d}_{:d}_ae_noconv.dat'.format('002',20,3)
noconv = []
with open(fname,'r') as cf:
    for line in cf:
        line = line.split()
        noconv.append(float(line[0]))
noconv = np.array(noconv)

wl_inds = np.ones(len(wl),dtype=int)
for i in range(len(wl)):
    wl_inds[i] = np.argmin(np.fabs(wl[i]-wlfine))
print(wl_inds)
noconv_downsample = noconv[wl_inds]
plt.plot(wl,noconv_downsample)
plt.show()




sunappref = np.divide(upsignal,sunspectrum)
splineorder = 2


minpeak = np.argmin(np.fabs(wl-758))
maxpeak = np.argmin(np.fabs(wl-769))
nopeak_appref = []
nopeak_wl = []
for i in range(len(wl)):
    if i < minpeak or i > maxpeak:
        nopeak_appref.append(appref[i])
        nopeak_wl.append(wl[i])
plt.figure()
plt.plot(nopeak_wl,nopeak_appref)
plt.show()
interRa_smooth = interpolate.UnivariateSpline(nopeak_wl,nopeak_appref,s=0.0001,k=splineorder)
init_R = interRa_smooth(wl) 
p_init = np.polyfit(nopeak_wl,nopeak_appref,2)
interp = np.poly1d(p_init)
poly_R_init = interp(wl)
jmin = 1.2
jmax = 2.0
# no need to go more than 4 (~4nm) as above, no fine strucutre is resolved at all. Also no need to go below -1 
nlevels = 100

scales = np.logspace(jmin,jmax,num=nlevels,base=2)

decomp = wavelets.create_decomp_p(upsignal,scales)
refdecomp = wavelets.create_decomp_p(whitereference,scales)
wlscales = wavelets.get_wlscales(scales)*(wl[1]-wl[0])
plotting.plot_powerspectrum(wl,refdecomp,scales,wlscales,upsignal)

#init_R = rtanh.optimize_tanh(wl,whitereference,signal,scales,w0=wl[0],w1=wl[-1])

plt.figure()
plt.plot(wl,init_R)
plt.show()
#plt.plot(wl,refl)


interR, smoothing = rspline.adjust_smoothing(wl,init_R,0.5,2,splineorder)
plt.plot(wl,poly_R_init)
plt.plot(wl,appref)
plt.show()

splinecoeffs = interR._data[9]
knots = interR._data[8]
nz_knots = knots[np.nonzero(knots)]

weights = rpoly.determine_weights(wl,upsignal,scales)
print(knots)
""" plt.figure()
plt.pcolor(nz_knots,wlscales,weights[:,:len(nz_knots)],cmap='viridis',shading='auto')
plt.colorbar()
plt.yscale('log',base=10.0)
plt.show() """

sfmmin = np.argmin(np.fabs(wlorig-670))
sfmmax = np.argmin(np.fabs(wlorig-780))
sfmWL = wlorig[sfmmin:sfmmax]
x,Fsfm,Rsfm,resnorm, exitflag, nfevas,sfmres = SFM.FLOX_SpecFit_6C(sfmWL,whitereferenceorig[sfmmin:sfmmax],signalorig[sfmmin:sfmmax],[1,1],1.,wl,alg='trf')


coeffs = rpoly.optimize_coeffs(wl,whitereference,upsignal,p_init,scales,lbl=1)

coeffs = np.array(coeffs)
polyrefls = []
cmap = plotting.get_colormap(len(coeffs))
plt.figure()
for i,polycoef in enumerate(coeffs):
    interp = np.poly1d(polycoef)
    polyrefls.append(interp(wl))
    plt.plot(wl,interp(wl),color=cmap(i))



polyweights = weights
polyrefls = np.array(polyrefls)
polyR = np.average(polyrefls,weights=polyweights,axis=0)



""" B = rspline.setup_B(wl,knots,splineorder)
final_R = np.zeros(len(wl))
allcoeffs = []

plt.figure()
if coeffs.ndim > 1:
    count = 0
    for i in range(len(coeffs)):

        ncoeffs = np.zeros(len(B))
        ncoeffs[:len(coeffs[0])] = np.array(coeffs[i])
        print(coeffs[i])
        level_R = rspline.bspleval(ncoeffs,B)
        level_R[-1] = level_R[-2]
        allcoeffs.append(ncoeffs)
        
        final_R += level_R
        if np.mean(weights[i]) > np.median(np.mean(weights,axis=1)):
            plt.plot(wl[:-1],level_R[:-1],color=cmap(i),label=i)
        #plt.plot(wl[:-1],np.divide((level_R[:-1]-refl[:-1])*whitereference[:-1],F[:-1]))
        count += 1
    final_R /= count
    allcoeffs = np.array(allcoeffs)
    meancoeffs = np.average(allcoeffs,weights=weights,axis=0)
    stdcoeffs = funcs.weighted_std(allcoeffs, weights,axis=0)
    print(meancoeffs)
    finalR_meancoeffs = rspline.bspleval(meancoeffs,B)
    finalR_meancoeffs[-1] = finalR_meancoeffs[-2]
    finalR_std = rspline.bspleval(stdcoeffs,B)
else:
    ncoeffs = np.zeros(len(B))
    ncoeffs[:len(coeffs)] = np.array(coeffs)
    final_R = rspline.bspleval(ncoeffs,B)
    plt.plot(wl[:-1],final_R[:-1],label='derived R')
    #plt.plot(wl[:-1],final_R[:-1]-refl[:-1])
 """



plt.plot(wl,(upsignal-F)/whitereference,color='tab:orange',label='Expected Reflectance')
#plt.plot(knots,interR(knots),'o',color='tab:blue')
#plt.plot(wl,final_R,color='tab:red',linewidth=2.0)
#plt.plot(wl,finalR_meancoeffs,color='tab:red',linewidth=2.0,label=r'Weighted Average')
plt.plot(wl,polyR,color='tab:red',label='Wavelet Reflectance')
plt.plot(wl,appref,color='tab:blue',label=r'Apparent Reflectance')
#plt.plot(wl,upsignal/sunspectrum,label='Sun Apparent Reflectance')
plt.plot(wl,Rsfm,label=r'Spectral Fitting Method',color='limegreen')
plt.xlim(670,780)
plt.ylim(0,0.5)
#plt.plot(wl,np.fabs(final_R-refl)*whitereference/F)
plt.legend()
plt.savefig('example_finalR_data.pdf')
plt.figure()
""" for i in range(len(meancoeffs)):
    plt.plot(wl, meancoeffs[i]*B[i,splineorder,:]) """
plt.title('B-spline basis functions')
F_der = upsignal-polyR*whitereference
#F_err = np.sqrt((whitereference_sm*finalR_std)**2)
Fdecomp = wavelets.create_decomp_p(F_der,np.logspace(-2,8.5,num=128,base=2))
#Fdecomp_inp = wavelets.create_decomp_p(F,np.logspace(-2,8.5,num=128,base=2))
#Fwlscales = wavelets.get_wlscales(np.logspace(-2,8.5,num=128,base=2))*(wl[1]-wl[0])
#plotting.plot_powerspectrum(wl,Fdecomp_inp,np.logspace(-2,8.5,num=128,base=2),Fwlscales,F)
Frec = wavelets.icwavelet(Fdecomp,np.logspace(-2,14.5,num=128,base=2))
wvl_m = []

with open('../../flox-specfit/sfm_compareimpl_matlab_oenComp','r') as mat:
    for line in mat:
        line = line.split()
        try:
            float(line[0])
        except:
            for w in line[2:]:
                wvl_m.append(float(w))
            continue
        if int(line[0]) == 5 and int(line[1]) == 2:
            F_m = [float(val) for val in line[2:]]
        
        


plt.figure()
plt.plot(wl,F,label=r'Input',color='forestgreen')
plt.plot(wl,F_der,'--',label=r'Wavelet derived',color='tab:red',alpha=0.5)
#plt.plot(wl,Frec,label=r'Large scale reconstruction',color='tab:red')
#plt.fill_between(wl,Frec-F_err,Frec+F_err,alpha=0.5,color='forestgreen')
plt.plot(wl,Fsfm,label=r'Spectral Fitting Method',color='tab:blue')

#plt.fill_between(wl,Fsfm-merror*Fsfm,Fsfm+merror*Fsfm,alpha=0.5)
#plt.plot(760,ifldaref,'o',color='tab:blue')
#plt.plot(wvl_m,F_m,label='Matlab')
plt.legend()
plt.ylim(0,6)
#plt.xlim(670,800)
plt.savefig('example_derivedF_scope.pdf')
plt.figure()
plt.plot(sfmres)
plt.show()

