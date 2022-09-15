import numpy as np
from scipy import interpolate, signal
from r_opts import rspline,rtanh,rpoly
from ih import prepare_input
import matplotlib.pyplot as plt
import scipy
from utils import wavelets,plotting, funcs
from SFM import SFM,SFM_BSpline
from scipy import optimize
import tikzplotlib
from matplotlib import rc
#plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})


################### retrieval parameters ###############################

# sifeffec: fluorescence yield (0.02 is SCOPE standard value)
sifeffec = ['002']

# chlorophyll contents and leaf area indices:
CAB = []
CAB = [int(5),int(10)]
for i in range(2,8):
    CAB.append(i*10)

LAI = [1,2,3,4,5,6,7]

# noise: n=1, no noise added: n=0
n = [0,1]

# order of polynomial for reflectance:
polyorder = 2

# retrieval window and evaluation wavelength for analysis
mineval,maxeval,eval_wl = 754,773,760
#mineval,maxeval,eval_wl = 745,755,750

# initial decomposition scales (=[2^jmin,...,2^jmax]) from which optimal scales are determined:
jmin = -2.5
jmax = 3
nlevels = 2048

########################################################################

cm = plotting.get_colormap(len(CAB)*len(LAI))

compfig,compax = plt.subplots()

N = 0
#exfig,(exax1,exax2) = plt.subplots(1,2)
for noise in n:
    Fgrid = np.zeros((len(CAB),len(LAI)))
    i = 0
    resfig,resax = plt.subplots()
    maes = []
    inputs = []
    diurnal = []
    diurnalsfm = []
    errs_750 = []
    for fe in sifeffec:
        for c,cab in enumerate(CAB):
            for l,lai in enumerate(LAI):

                ############### data preparation #######################################
                
                wl, upsignal, whitereference, refl, F,noF = prepare_input.synthetic(cab,lai,fe,wlmin=mineval,wlmax=maxeval,completedir='../cwavelets/libradtranscope/floxseries_ae_oen/brdf/')
                sfmwl, sfmsignal, sfmref, sfmrefl, sfmF,sfmnoF = prepare_input.synthetic(cab,lai,fe,wlmin=670,wlmax=780,completedir='../cwavelets/libradtranscope/floxseries_ae_oen/brdf/')
                """ exax1.plot(sfmwl,sfmsignal,color=cm(i),linewidth=0.3)
                exax1.plot(sfmwl,sfmref,color=cm(i),linewidth=0.3)
                exax2.plot(sfmwl,sfmF,color=cm(i),linewidth=0.3,label=r'$C_{{ab}} = {:d} \mu \textrm{{g}}/ \textrm{{cm}}^2$, $\textrm{{LAI}} = {:d} \textrm{{m}}^2/\textrm{{m}}^2$'.format(cab,lai)) """

                inpR_all = sfmnoF/sfmref
                
                inpR = noF/whitereference
                
                refnoise, whitereference = prepare_input.add_noise(whitereference,1000,noise)
                sensornoise, upsignal = prepare_input.add_noise(upsignal,1000,noise)
                sfmnoise, sfmref =  prepare_input.add_noise(sfmref,1000,noise)
                sfmnoise, sfmsignal =  prepare_input.add_noise(sfmsignal,1000,noise)

                ########################################################################

                ######################### SFM retrieval ################################

                x,Fsfm,Rsfm,resnorm, exitflag, nfevas,sfmres = SFM.FLOX_SpecFit_6C(sfmwl,sfmref,sfmsignal,[1,1],1.,wl,alg='trf')

                ########################################################################

                ############### initial guess and reflectance optimization #############

                appref = np.divide(upsignal,whitereference)
                nopeak_wl, nopeak_appref = prepare_input.rm_peak(wl,appref) 
    
                p_init = np.polyfit(nopeak_wl,nopeak_appref,polyorder)
              
            
                #p_init[-1] = p_init[-1] - 0.3
                interp = np.poly1d(p_init)
                poly_R_init = interp(wl)
                    
                if i == 0:
                    # decomposition levels only adjusted for first dataset, to save time, others are assumed to be similar enough
                    newdecomp = wavelets.decomp(jmin,jmax,nlevels)
                    newdecomp.adjust_levels(upsignal)
                    newdecomp.create_comps(upsignal)
                    
                    ######################## plotting ######################################################
                    plotting.plot_powerspectrum(wl,newdecomp.comps,newdecomp.scales,wavelets.get_wlscales(newdecomp.scales*(wl[1]-wl[0])),'upsignal_decomp')
                    decfig,decax = plt.subplots()
                    decax.plot(wl,newdecomp.comps[5],label=r'$\hat{s}$',color='forestgreen')
                    newdecomp.create_comps(whitereference)
                    plotting.plot_powerspectrum(wl,newdecomp.comps,newdecomp.scales,wavelets.get_wlscales(newdecomp.scales*(wl[1]-wl[0])),'whitereference_decomp')
                    decax.plot(wl,newdecomp.comps[5],label=r'$\hat{s_0}$',color='tab:orange')
                    ax2 = decax.twinx()
                    ax2.plot(wl,upsignal,color='forestgreen',linewidth=0.8,linestyle='dashed',label=r'$s$')
                    ax2.plot(wl,whitereference,color='tab:orange',linewidth=0.8,linestyle='dashed',label=r'$s_0$')
                    ax2.set_ylabel(r'Radiance [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
                    ax2.legend(loc='upper right')
                    decax.legend(loc='upper left')
                    decax.set_xlim(745.2,754.5)
                    decax.set_ylim(-4,4)
                    decax.set_xlabel(r'Wavelength [nm]')
                    decax.set_ylabel(r'Coefficient Strength')
                    tikzplotlib.save(figure=decfig,filepath='decomp_onelevel.tex')
                    plt.show()
                    ##########################################################################################
            

                coeffs,ress = rpoly.optimize_coeffs(wl,whitereference,upsignal,p_init,newdecomp)

                ########################################################################

                ############# weighting and final reflectance ##########################
                weights = newdecomp.calc_weights(upsignal)
                
                coeffs = np.array(coeffs)
                polyrefls = []
                #cmap = plotting.get_colormap(len(coeffs))
                #plt.figure()
                for j,polycoef in enumerate(coeffs):
                    interp = np.poly1d(polycoef)
                    polyrefls.append(interp(wl))
                    #plt.plot(wl,interp(wl),color=cmap(i))

                polyrefls = np.array(polyrefls)
                polyR = np.ma.average(polyrefls,weights=weights,axis=0)
                R_std = funcs.weighted_std(polyrefls,weights=weights,axis=0)

                """ plt.figure()
                plt.plot(ress[0],label='Initial Residual')
                plt.plot(ress[1],label='Final Residual')
                plt.legend()
                plt.show() """

                ########################################################################

                F_der = upsignal-polyR*whitereference
                F_param = np.polyfit(wl,F_der,1)
                Finterp = np.poly1d(F_param)
                F_smooth = Finterp(wl)

                
                ################## errors and statistics #################################

                F_err = np.sqrt(np.square(sensornoise) + np.square(np.multiply(whitereference,R_std)) + np.square(np.multiply(polyR,refnoise)))
                Ferr_750 = F_err[np.argmin(np.fabs(wl-760))]
                errs_750.append(Ferr_750)

                
                maes.append(np.mean(np.divide(np.fabs(F_smooth-F),F)))
                Ftotal,F687,F760,Fr,wlFr,Ffr,wlFfr,specialder = funcs.evaluate_sif(wl,F_der,eval_wl)
                diurnal.append(specialder)
                Ftotal,F687,F760,Fr,wlFr,Ffr,wlFfr,specialsfm = funcs.evaluate_sif(wl,Fsfm,eval_wl)
                diurnalsfm.append(specialsfm)
                Ftotal,F687,F760,Fr,wlFr,Ffr,wlFfr,specialin = funcs.evaluate_sif(wl,F,eval_wl)
                inputs.append(specialin)
                Fgrid[c,l] = np.sqrt(np.mean(np.square(F_der-F)))
                
                
                ################## plotting #############################################
                if i == 0:

                    resax.plot(wl,F_der,'--',label=r'Wavelet derived',color=cm(i),alpha=0.5)
                    resax.plot(wl,F_smooth,label=r'Polynomial',color=cm(i))
                    resax.plot(wl,F,linestyle='dotted',label=r'Input',color=cm(i))
                else:
                    resax.plot(wl,F_der,'--',color=cm(i),alpha=0.5)
                    resax.plot(wl,F_smooth,color=cm(i))
                    resax.plot(wl,F,linestyle='dotted',color=cm(i))
                ##########################################################################

                i += 1
    
    
    ################## more plotting and statistics  #################################
    meshname = 'scope_rsme_{:d}{:d}_{}.pdf'.format(mineval,maxeval,noise)
    if noise == 0:
        meshtext = r'{:d}-{:d} nm'.format(mineval,maxeval)
        #resax.plot(maes,'.',label='FL, no noise')
        r2_w = funcs.calc_rsquared(np.array(diurnal),np.array(inputs))
        r2_s = funcs.calc_rsquared(np.array(diurnalsfm),np.array(inputs))
        rsme_w = funcs.calc_rmse(np.array(diurnal),np.array(inputs))
        rsme_s = funcs.calc_rmse(np.array(diurnalsfm),np.array(inputs))
        compax.errorbar(inputs,diurnal,errs_750,fmt='.',label=r'no noise, $R^2 = {:.2f}$'.format(r2_w),color='tab:blue')
        compax.plot(inputs,diurnalsfm,'*',label=r'SFM, no noise, $R^2 = {:.2f}$'.format(r2_s),color='tab:blue')
        allmax = np.maximum(diurnal,inputs)
        one = np.linspace(0,np.max(allmax),100)    
        w_coeffs,w_cov = optimize.curve_fit(funcs.fitfct_linear,inputs,diurnal)#,sigma=errs_750)
        print(w_coeffs,r2_w,rsme_w)
        compax.plot(one,funcs.fitfct_linear(one,*w_coeffs),color='tab:blue',linewidth=0.4)
        s_coeffs,s_cov = optimize.curve_fit(funcs.fitfct_linear,inputs,diurnalsfm)
        print(s_coeffs,r2_s,rsme_s)
        compax.plot(one,funcs.fitfct_linear(one,*s_coeffs),color='tab:blue',linewidth=0.4)
     
        
    else:
        meshtext = r'{:d}-{:d} nm, with noise'.format(mineval,maxeval)
        #resax.plot(maes,'.',label='FL, noise')  
        r2_w_n = funcs.calc_rsquared(np.array(diurnal),np.array(inputs))
        r2_s_n = funcs.calc_rsquared(np.array(diurnalsfm),np.array(inputs))
        rsme_w_n = funcs.calc_rmse(np.array(diurnal),np.array(inputs))
        rsme_s_n = funcs.calc_rmse(np.array(diurnalsfm),np.array(inputs))
        compax.errorbar(inputs,diurnal,errs_750,fmt='.',label=r'noise, $R^2 = {:.2f}$'.format(r2_w_n),color='tab:red')
        compax.plot(inputs,diurnalsfm,'*',label=r'SFM, noise, $R^2 = {:.2f}$'.format(r2_s_n),color='tab:red')  
        allmax = np.maximum(diurnal,inputs)
        one = np.linspace(0,np.max(allmax),100)    
        w_coeffs,w_cov = optimize.curve_fit(funcs.fitfct_linear,inputs,diurnal,sigma=errs_750)
        print(w_coeffs,r2_w_n,rsme_w_n)
        compax.plot(one,funcs.fitfct_linear(one,*w_coeffs),color='tab:red',linewidth=0.4)
        s_coeffs,s_cov = optimize.curve_fit(funcs.fitfct_linear,inputs,diurnalsfm)
        print(s_coeffs,r2_s_n,rsme_s_n)
        compax.plot(one,funcs.fitfct_linear(one,*s_coeffs),color='tab:red',linewidth=0.4)
        
    plotting.plot_3d(LAI,CAB,Fgrid,meshname,meshtext)
    resax.legend()
    resfig.savefig('spectral_{:d}_{:d}_poly2_looseboundary_new.pdf'.format(eval_wl,noise))
    #exax2.legend(loc='upper left')
    #tikzplotlib.save(figure=exfig,filepath='example_specs.tex')
 

one = np.linspace(0,np.max(allmax),100)            
compax.set_xlabel(r'$F_{{{:d}}}$ Input [mW nm$^{{-1}}$ m$^{{-2}}$ ster$^{{-1}}$]'.format(eval_wl))   
compax.set_ylabel(r'$F_{{{:d}}}$ Retrieved [mW nm$^{{-1}}$ m$^{{-2}}$ ster$^{{-1}}$]'.format(eval_wl))  
compax.legend()
compax.plot(one,one,'--',color='k')
compfig.savefig('FL_{:d}_poly2_looseboundary_new.pdf'.format(eval_wl))
tikzplotlib.save(figure=compfig,filepath='FL_{:d}_poly2_looseboundary_new.tex'.format(eval_wl))




