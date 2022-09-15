import wave
import numpy as np
from scipy import interpolate, signal
from r_opts import rpoly
from ih import prepare_input
import matplotlib.pyplot as plt
import scipy
from utils import wavelets,plotting,funcs,results
from SFM import SFM,SFM_BSpline
from scipy import optimize
import tikzplotlib
from matplotlib import rc
#plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})


ranges = [681,695,687]
pg = 2075*384+152#461714
pw = 2036*384+117#831696

windowmin = ranges[0]
windowmax = ranges[1]
eval_wl = ranges[2]

hyplant_res = results.retrieval_res('Hyplant','{:d}_{:d}'.format(pg,pw),windowmin,windowmax,eval_wl,'hyplant_res')


wlorig, upsignalorig, referenceorig = prepare_input.hyplant(pg,pw,wlmin=660)
wl, upsignal, reference = prepare_input.hyplant(pg,pw,wlmin=windowmin,wlmax=windowmax)

hyplant_res.init_wl(wl)
    
rangefig, rangeax = plt.subplots()
rangeax.plot(wlorig,upsignalorig)
rangeax.axvspan(windowmin, windowmax, color='green', alpha=0.5)
rangeax.set_xlabel(r'Wavelength [nm]')
rangeax.set_ylabel(r'Radiance [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
figname1 = 'HP_examplerange_{:d}{:d}_oens_{:d}_{:d}.pdf'.format(windowmin,windowmax,pg,pw)
figname1tex = 'HP_examplerange_{:d}{:d}_oens_{:d}_{:d}.tex'.format(windowmin,windowmax,pg,pw)
tikzplotlib.save(figure=rangefig,filepath=figname1tex)
rangefig.savefig(figname1)



polyorder = 2

diurnal = []
diurnalsfm = []
diurnalR = []
diurnalRsfm = []
meanerrors = []
diurnalRerrors = []
diurnalFerrors = []
diurnalappRerrors = []
testjmin = -2.0
testjmax = 1.0
testnlevels = 2048

resfig,(resax1,resax2) = plt.subplots(1,2,figsize=(10,6))
scatfig, scatax = plt.subplots()
#datfig, datax = plt.subplots()
Fws = []
Fsfms = []
sfm_residuals = []


appref = np.divide(upsignal,reference)
nopeak_wl, nopeak_appref = prepare_input.rm_peak(wl,appref) 

if len(nopeak_appref) < len(wl):
    pfit = np.polyfit(nopeak_wl,nopeak_appref,2)
    pinterp = np.poly1d(pfit)
    nopeak_appref = pinterp(wl)
    nopeak_wl = wl



p_init = np.polyfit(nopeak_wl,nopeak_appref,polyorder)
interp = np.poly1d(p_init)
poly_R_init = interp(wl)

newdecomp = wavelets.decomp(testjmin,testjmax,testnlevels)
newdecomp.adjust_levels(upsignal)
print(newdecomp.jmin,newdecomp.jmax,wavelets.get_wlscales(newdecomp.scales*(wl[1]-wl[0])))
    


newdecomp.calc_weights(upsignal)
weights = newdecomp.weights

sfmmin = np.argmin(np.fabs(wlorig-670))
sfmmax = np.argmin(np.fabs(wlorig-780))
sfmWL = wlorig[sfmmin:sfmmax]
x,Fsfm,Rsfm,resnorm, exitflag, nfevas,sfmres = SFM.FLOX_SpecFit_6C(sfmWL,referenceorig[sfmmin:sfmmax],upsignalorig[sfmmin:sfmmax],[1,1],1.,wl,alg='trf')
hyplant_res.Fsfm[0] = Fsfm
sfm_residuals.append(sfmres)
sunreference,_ = prepare_input.match_solspec(wl,0.3)

coeffs = rpoly.optimize_coeffs(wl,reference,upsignal,p_init,newdecomp)
polyrefls = []
cmap = plotting.get_colormap(len(coeffs))
plt.figure()
for i,polycoef in enumerate(coeffs[0]):
    
    interp = np.poly1d(polycoef)
    polyrefls.append(interp(wl))
    plt.plot(wl,interp(wl),color=cmap(i))




polyrefls = np.array(polyrefls)
polyR = np.average(polyrefls,weights=weights,axis=0)
hyplant_res.R = polyR
R_err = funcs.weighted_std(polyrefls,weights=weights,axis=0)




plt.plot(wl,polyR,color='tab:red',label=r'Wavelet Reflectance')
plt.plot(wl,appref,color='tab:blue',label=r'Apparent Reflectance')

plt.plot(wl,Rsfm,label=r'Spectral Fitting Method',color='limegreen')
plt.xlim(windowmin,windowmax)
#plt.ylim(0.3,0.6)

plt.legend()



F_der = upsignal-polyR*reference
hyplant_res.F[0] = F_der


resax1.plot(wl,polyR,color='forestgreen',linewidth=0.8,label='Reflectance')
resax1.plot(wl,appref,'--',color='forestgreen',linewidth=0.8,label='Apparent Reflectance')
resax2.plot(wl,F_der,color='tab:red',linewidth=0.8)

resax2.plot(wl,Fsfm,'--',color='tab:red',linewidth=0.8)
#datax.plot(wlorig,signalorig,color=cm(t),label=plottimes[t],linewidth=0.8)
#datax.plot(wlorig,whitereferenceorig,color=cm(t),linewidth=0.8)
    


F_param = np.polyfit(wl,F_der,2)
Finterp = np.poly1d(F_param)
F_smooth = Finterp(wl)
hyplant_res.evaluate_sif(hyplant_res.F)
diurnal.append(hyplant_res.F[1][7])
hyplant_res.evaluate_sif(hyplant_res.Fsfm)
diurnalsfm.append(hyplant_res.Fsfm[1][7])

Fws.append(F_der)
Fsfms.append(Fsfm)

diurnalR.append(np.median(polyR))
diurnalRsfm.append(np.median(Rsfm))
diurnalRerrors.append(np.median(R_err))




plt.show()

