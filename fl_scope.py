import numpy as np
from scipy import interpolate, signal
from r_opts import rspline,rpoly
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
rng = np.random.default_rng(seed=42)
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
    scope_res = results.retrieval_res('SCOPE','Cab_LAI',mineval,maxeval,eval_wl,'scope_res_sfmwind')
    
    for fe in sifeffec:
        for c,cab in enumerate(CAB):
            for l,lai in enumerate(LAI):

                ############### data preparation #######################################
                
                wl, upsignal, whitereference, refl, F,noF = prepare_input.synthetic(cab,lai,fe,wlmin=mineval,wlmax=maxeval,completedir='../cwavelets/libradtranscope/floxseries_ae_oen/brdf/')
                sfmwl, sfmsignal, sfmref, sfmrefl, sfmF,sfmnoF = prepare_input.synthetic(cab,lai,fe,wlmin=mineval,wlmax=maxeval,completedir='../cwavelets/libradtranscope/floxseries_ae_oen/brdf/')
                scope_res.init_wl(wl)
                if N == 0:
                    scope_res.initiate_ts_tofile()
                scope_res.Finp.spec = F
                
                
                """ exax1.plot(sfmwl,sfmsignal,color=cm(i),linewidth=0.3)
                exax1.plot(sfmwl,sfmref,color=cm(i),linewidth=0.3)
                exax2.plot(sfmwl,sfmF,color=cm(i),linewidth=0.3,label=r'$C_{{ab}} = {:d} \mu \textrm{{g}}/ \textrm{{cm}}^2$, $\textrm{{LAI}} = {:d} \textrm{{m}}^2/\textrm{{m}}^2$'.format(cab,lai)) """

                inpR_all = sfmnoF/sfmref
                
                inpR = noF/whitereference
                
                refnoise, whitereference = prepare_input.add_noise(whitereference,1000,noise,rng)
                sensornoise, upsignal = prepare_input.add_noise(upsignal,1000,noise,rng)
                sfmnoise, sfmref =  prepare_input.add_noise(sfmref,1000,noise,rng)
                sfmnoise, sfmsignal =  prepare_input.add_noise(sfmsignal,1000,noise,rng)

                ########################################################################

                ######################### SFM retrieval ################################

                x,Fsfm,Rsfm,resnorm, exitflag, nfevas,sfmres = SFM.FLOX_SpecFit_6C(sfmwl,sfmref,sfmsignal,[1,1],1.,wl,alg='trf')
                scope_res.Fsfm.spec = Fsfm

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
                    figF,axF = plt.subplots(1,2,figsize=(9/1.3,3.9/1.5))
                    figF,axF,pF2 = plotting.plot_powerspectrum(wl,newdecomp.comps,newdecomp.scales,wavelets.get_wlscales(newdecomp.scales*(wl[1]-wl[0])),(figF,axF),1,widths=True)
                    ######################## plotting ######################################################
                    decfig,decax = plt.subplots()
                    decax.plot(wl,newdecomp.comps[5],label=r'$\hat{s}$',color='forestgreen')
                    newdecomp.create_comps(whitereference)
                    figF,axF,pF1 = plotting.plot_powerspectrum(wl,newdecomp.comps,newdecomp.scales,wavelets.get_wlscales(newdecomp.scales*(wl[1]-wl[0])),(figF,axF),0,levels=True,colorbar=True)
                   
                    
                    
                    figF.subplots_adjust(wspace=0.3)
                    cbar_ax = figF.add_axes([0.5, 0.15, 0.015, 0.7])
                    cbar = figF.colorbar(pF1, cax=cbar_ax)
                    cbar.ax.tick_params(labelsize=8) 
                    figF.savefig('decomps_ex.pdf',dpi=300,bbox_inches="tight")
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
                polyR, R_std = funcs.weighted_std(polyrefls,weights=weights,axis=0)
                scope_res.R = polyR
      

                ########################################################################

                F_der = upsignal-polyR*whitereference
                scope_res.F.spec = F_der
                F_param = np.polyfit(wl,F_der,1)
                Finterp = np.poly1d(F_param)
                F_smooth = Finterp(wl)
                scope_res.write_ts_tofile('{:d}_{:d}_rerun'.format(cab,lai))

                
                ################## errors and statistics #################################

                F_err = np.sqrt(np.square(sensornoise) + np.square(np.multiply(whitereference,R_std)) + np.square(np.multiply(polyR,refnoise)))
                Ferr_750 = F_err[np.argmin(np.fabs(wl-760))]
                errs_750.append(Ferr_750)

                
                maes.append(np.mean(np.divide(np.fabs(F_der-F),F)))
                scope_res.F.evaluate_sif()
                diurnal.append(scope_res.F.spec_val)
                scope_res.Fsfm.evaluate_sif()
                diurnalsfm.append(scope_res.Fsfm.spec_val)
                scope_res.Finp.evaluate_sif()
                inputs.append(scope_res.Finp.spec_val)
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
                N += 1
    
    
    ################## more plotting and statistics  #################################
    meshname = 'scope_rsme_{:d}{:d}_{}_rerun.pdf'.format(mineval,maxeval,noise)
    if noise == 0:
        meshtext = r'{:d}-{:d} nm'.format(mineval,maxeval)
        #resax.plot(maes,'.',label='FL, no noise')
        r2_w = funcs.calc_rsquared(np.array(diurnal),np.array(inputs))
        r2_s = funcs.calc_rsquared(np.array(diurnalsfm),np.array(inputs))
        rsme_w = funcs.calc_rmse(np.array(diurnal),np.array(inputs))
        rsme_s = funcs.calc_rmse(np.array(diurnalsfm),np.array(inputs))
        rrsme_w = funcs.calc_rrmse(np.array(diurnal),np.array(inputs))
        rrsme_s = funcs.calc_rrmse(np.array(diurnalsfm),np.array(inputs))

        compax.errorbar(inputs,diurnal,errs_750,fmt='.',label=r'no noise, $R^2 = {:.2f}$'.format(r2_w),color='tab:blue')
        compax.plot(inputs,diurnalsfm,'*',label=r'SFM, no noise, $R^2 = {:.2f}$'.format(r2_s),color='tab:blue')
        allmax = np.maximum(diurnal,inputs)
        one = np.linspace(0,np.max(allmax),100)    
        w_coeffs,w_cov = optimize.curve_fit(funcs.fitfct_linear,inputs,diurnal)#,sigma=errs_750)
        print(w_coeffs,r2_w,rsme_w,rrsme_w)
        compax.plot(one,funcs.fitfct_linear(one,*w_coeffs),color='tab:blue',linewidth=0.4)
        s_coeffs,s_cov = optimize.curve_fit(funcs.fitfct_linear,inputs,diurnalsfm)
        print(s_coeffs,r2_s,rsme_s,rrsme_s)
        compax.plot(one,funcs.fitfct_linear(one,*s_coeffs),color='tab:blue',linewidth=0.4)
        # r^2 with respect to the regression line (meaningless in this scenario, but routinely done)
        r2_w_reg = funcs.calc_rsquared(np.array(diurnal),funcs.fitfct_linear(np.array(inputs),*w_coeffs))
        r2_s_reg = funcs.calc_rsquared(np.array(diurnalsfm),funcs.fitfct_linear(np.array(inputs),*s_coeffs))
        print(r2_w_reg,r2_s_reg)
        print(np.mean(np.array(diurnal-np.array(inputs))),np.mean(np.array(diurnalsfm-np.array(inputs))))
        
     
        
    else:
        meshtext = r'{:d}-{:d} nm, with noise'.format(mineval,maxeval)
        #resax.plot(maes,'.',label='FL, noise')  
        r2_w_n = funcs.calc_rsquared(np.array(diurnal),np.array(inputs))
        r2_s_n = funcs.calc_rsquared(np.array(diurnalsfm),np.array(inputs))
        rsme_w_n = funcs.calc_rmse(np.array(diurnal),np.array(inputs))
        rsme_s_n = funcs.calc_rmse(np.array(diurnalsfm),np.array(inputs))
        rrsme_w_n = funcs.calc_rrmse(np.array(diurnal),np.array(inputs))
        rrsme_s_n = funcs.calc_rrmse(np.array(diurnalsfm),np.array(inputs))
        compax.errorbar(inputs,diurnal,errs_750,fmt='.',label=r'noise, $R^2 = {:.2f}$'.format(r2_w_n),color='tab:red')
        compax.plot(inputs,diurnalsfm,'*',label=r'SFM, noise, $R^2 = {:.2f}$'.format(r2_s_n),color='tab:red')  
        allmax = np.maximum(diurnal,inputs)
        one = np.linspace(0,np.max(allmax),100)    
        w_coeffs,w_cov = optimize.curve_fit(funcs.fitfct_linear,inputs,diurnal,sigma=errs_750)
        print(w_coeffs,r2_w_n,rsme_w_n,rrsme_w_n)
        compax.plot(one,funcs.fitfct_linear(one,*w_coeffs),color='tab:red',linewidth=0.4)
        s_coeffs,s_cov = optimize.curve_fit(funcs.fitfct_linear,inputs,diurnalsfm)
        print(s_coeffs,r2_s_n,rsme_s_n,rrsme_s_n)
        compax.plot(one,funcs.fitfct_linear(one,*s_coeffs),color='tab:red',linewidth=0.4)
        r2_w_reg_n = funcs.calc_rsquared(np.array(diurnal),funcs.fitfct_linear(np.array(inputs),*w_coeffs))
        r2_s_reg_n = funcs.calc_rsquared(np.array(diurnalsfm),funcs.fitfct_linear(np.array(inputs),*s_coeffs))
        print(r2_w_reg_n,r2_s_reg_n)
        print(np.mean(np.array(diurnal-np.array(inputs))),np.mean(np.array(diurnalsfm-np.array(inputs))))
        
    plotting.plot_3d(LAI,CAB,Fgrid,meshname,meshtext)
    resax.legend()
    resfig.savefig('spectral_{:d}_{:d}_poly2_looseboundary_rerun.pdf'.format(eval_wl,noise))
    #exax2.legend(loc='upper left')
    #tikzplotlib.save(figure=exfig,filepath='example_specs.tex')
 

one = np.linspace(0,np.max(allmax),100)            
compax.set_xlabel(r'$F_{{{:d}}}$ Input [mW nm$^{{-1}}$ m$^{{-2}}$ ster$^{{-1}}$]'.format(eval_wl))   
compax.set_ylabel(r'$F_{{{:d}}}$ Retrieved [mW nm$^{{-1}}$ m$^{{-2}}$ ster$^{{-1}}$]'.format(eval_wl))  
compax.legend()
compax.plot(one,one,'--',color='k')
compfig.savefig('O2_{:d}_poly2_looseboundary_sfmwin_rerun.pdf'.format(eval_wl))
tikzplotlib.save(figure=compfig,filepath='O2_{:d}_poly2_looseboundary_sfmwin_rerun.tex'.format(eval_wl))




