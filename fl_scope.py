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

# the reflectance is imprinted in the absolute depth of the Fraunhofer lines and therefore the wavelet coefficients. The Fluorescence is not as it does not contribute to the small scale wavelet decomposition.
sifeffec = ['002']
CAB = []
CAB = [int(5),int(10)]
for i in range(2,8):
    CAB.append(i*10)

LAI = [1,2,3,4,5,6,7]
n = [0,1]


polyorder = 1

cm = plotting.get_colormap(len(CAB)*len(LAI))

compfig,compax = plt.subplots()
testjmin = -2
testjmax = 4
testnlevels = 2048
N = 0
for noise in n:
    i = 0
    resfig,resax = plt.subplots()
    maes = []
    inputs = []
    diurnal = []
    diurnalsfm = []
    errs_750 = []
    for fe in sifeffec:
        for cab in CAB:
            for lai in LAI:
                wl, upsignal, whitereference, refl, F,noF = prepare_input.synthetic(cab,lai,fe,wlmin=745,wlmax=755,completedir='../cwavelets/libradtranscope/floxseries_ae_oen/brdf/')
                sfmwl, sfmsignal, sfmref, sfmrefl, sfmF,sfmnoF = prepare_input.synthetic(cab,lai,fe,wlmin=670,wlmax=780,completedir='../cwavelets/libradtranscope/floxseries_ae_oen/brdf/')
                inpR_all = sfmnoF/sfmref
                
                inpR = noF/whitereference
                
                refnoise, whitereference = prepare_input.add_noise(whitereference,1000,noise)
                sensornoise, upsignal = prepare_input.add_noise(upsignal,1000,noise)
                sfmnoise, sfmref =  prepare_input.add_noise(sfmref,1000,noise)
                sfmnoise, sfmsignal =  prepare_input.add_noise(sfmsignal,1000,noise)

                x,Fsfm,Rsfm,resnorm, exitflag, nfevas,sfmres = SFM.FLOX_SpecFit_6C(sfmwl,sfmref,sfmsignal,[1,1],1.,wl,alg='trf')

                appref = np.divide(upsignal,whitereference)
        
                nopeak_wl, nopeak_appref = prepare_input.rm_peak(wl,appref) 
    
                p_init = np.polyfit(nopeak_wl,nopeak_appref,polyorder)
            
                p_init[-1] = p_init[-1] - 0.3
                interp = np.poly1d(p_init)
                poly_R_init = interp(wl)
                

                
                
                    
                    
                if i == 0:

                    newdecomp = wavelets.decomp(testjmin,testjmax,testnlevels)
                    newdecomp.adjust_levels(upsignal)
                    print(newdecomp.jmin,newdecomp.jmax)
            

    
            
    
                weights = wavelets.determine_weights(upsignal,newdecomp.scales)
                print(weights)

                coeffs,ress = rpoly.optimize_coeffs(wl,whitereference,upsignal,p_init,newdecomp.scales,lbl=1)
                """ plt.figure()
                plt.plot(ress[0],label='Initial Residual')
                plt.plot(ress[1],label='Final Residual')
                plt.legend()
                plt.show() """
                coeffs = np.array(coeffs)
                polyrefls = []
                cmap = plotting.get_colormap(len(coeffs))
                #plt.figure()
                for j,polycoef in enumerate(coeffs):
                    interp = np.poly1d(polycoef)
                    polyrefls.append(interp(wl))
                    #plt.plot(wl,interp(wl),color=cmap(i))

        
        
                polyrefls = np.array(polyrefls)
                polyR = np.ma.average(polyrefls,weights=weights,axis=0)
                R_std = funcs.weighted_std(polyrefls,weights=weights,axis=0)
                plt.figure()
                plt.plot(wl,inpR,color='forestgreen',label=r'Input')
                plt.plot(wl,appref,color='tab:blue',label=r'Apparent Reflectance')
                plt.plot(wl,Rsfm,label=r'Spectral Fitting Method',color='limegreen')
                plt.plot(wl,polyR,color='tab:red',label=r'Wavelet Reflectance')
                
                #plt.xlim(670,780)
                #plt.ylim(0.3,0.6)
                
                plt.legend()
                plt.xlabel(r'Wavelength [nm]')
                plt.ylabel(r'Reflectance')
                plt.savefig('FLrefl_lai{:d}_cab{:d}_noise{:d}.pdf'.format(lai,cab,noise))



                F_der = upsignal-polyR*whitereference
                F_err = np.sqrt(np.square(sensornoise) + np.square(np.multiply(whitereference,R_std)) + np.square(np.multiply(polyR,refnoise)))
                Ferr_750 = F_err[np.argmin(np.fabs(wl-750))]
                errs_750.append(Ferr_750)

                F_param = np.polyfit(wl,F_der,1)
                Finterp = np.poly1d(F_param)
                F_smooth = Finterp(wl)
                maes.append(np.mean(np.divide(np.fabs(F_smooth-F),F)))
                Ftotal,F687,F760,Fr,wlFr,Ffr,wlFfr = funcs.evaluate_sif(wl,F_der)
                diurnal.append(F760)
                Ftotal,F687,F760,Fr,wlFr,Ffr,wlFfr = funcs.evaluate_sif(wl,Fsfm)
                diurnalsfm.append(F760)
                Ftotal,F687,F760,Fr,wlFr,Ffr,wlFfr = funcs.evaluate_sif(wl,F)
                inputs.append(F760)
                
                
                if i == 0:

                    resax.plot(wl,F_der,'--',label=r'Wavelet derived',color=cm(i),alpha=0.5)
                    resax.plot(wl,F_smooth,label=r'Polynomial',color=cm(i))
                    resax.plot(wl,F,linestyle='dotted',label=r'Input',color=cm(i))
                else:
                    resax.plot(wl,F_der,'--',color=cm(i),alpha=0.5)
                    resax.plot(wl,F_smooth,color=cm(i))
                    resax.plot(wl,F,linestyle='dotted',color=cm(i))
                

                #plt.plot(wl,Fsfm,label=r'Spectral Fitting Method',color='tab:blue')
                
                
                i += 1
    if noise == 0:

        #resax.plot(maes,'.',label='FL, no noise')
        r2_w = funcs.calc_rsquared(np.array(diurnal),np.array(inputs))
        r2_s = funcs.calc_rsquared(np.array(diurnalsfm),np.array(inputs))
        compax.errorbar(inputs,diurnal,errs_750,fmt='.',label=r'no noise, $R^2 = {:.2f}$'.format(r2_w),color='tab:blue')
        compax.plot(inputs,diurnalsfm,'*',label=r'SFM, no noise, $R^2 = {:.2f}$'.format(r2_s),color='tab:blue')
        
        one = np.linspace(0,np.max(diurnal),100)    
        w_coeffs,w_cov = optimize.curve_fit(funcs.fitfct_linear,inputs,diurnal,sigma=errs_750)
        compax.plot(one,funcs.fitfct_linear(one,*w_coeffs),color='tab:blue',linewidth=0.4)
        s_coeffs,s_cov = optimize.curve_fit(funcs.fitfct_linear,inputs,diurnalsfm)
        compax.plot(one,funcs.fitfct_linear(one,*s_coeffs),color='tab:blue',linewidth=0.4)
        
    else:
        #resax.plot(maes,'.',label='FL, noise')  
        r2_w_n = funcs.calc_rsquared(np.array(diurnal),np.array(inputs))
        r2_s_n = funcs.calc_rsquared(np.array(diurnalsfm),np.array(inputs))
        compax.errorbar(inputs,diurnal,errs_750,fmt='.',label=r'noise, $R^2 = {:.2f}$'.format(r2_w_n),color='tab:red')
        compax.plot(inputs,diurnalsfm,'*',label=r'SFM, noise, $R^2 = {:.2f}$'.format(r2_s_n),color='tab:red')  

        one = np.linspace(0,np.max(diurnal),100)    
        w_coeffs,w_cov = optimize.curve_fit(funcs.fitfct_linear,inputs,diurnal,sigma=errs_750)
        compax.plot(one,funcs.fitfct_linear(one,*w_coeffs),color='tab:red',linewidth=0.4)
        s_coeffs,s_cov = optimize.curve_fit(funcs.fitfct_linear,inputs,diurnalsfm)
        compax.plot(one,funcs.fitfct_linear(one,*s_coeffs),color='tab:red',linewidth=0.4)
        
    
    resax.legend()
    resfig.savefig('FL_fl_'+str(noise)+'.pdf')
print(r2_w,r2_s)
print(r2_w_n,r2_s_n)  
""" resax.set_xlabel('Simulation Number')
resax.set_ylabel('Relative mean error') """

one = np.linspace(0,np.max(diurnal),100)            
compax.set_xlabel(r'Fluorescence Input at $750$~nm [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')   
compax.set_ylabel(r'Fluorescence Retrieved at $750$~nm [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')  
compax.legend()
compax.plot(one,one,'--',color='k')
compfig.savefig('FL_750.pdf')
plt.show()


