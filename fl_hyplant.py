import wave
import numpy as np
from scipy import interpolate, signal
from r_opts import rpoly
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


ranges = [[660,679,670],[681,695,687],[700,720,710],[725,740,735],[745,755,750],[754,773,760],[770,800,790]]
ranges = [[725,740,735]]
#ranges = [[684,702,687],[745,757,750],[753,775,760]]
#ranges = [[684,702,687],[753,775,760]]
pg = 2075*384+152#461714
pw = 2036*384+117#831696
for range1 in ranges:
    print(range1)
    windowmin = range1[0]
    windowmax = range1[1]
    eval_wl = range1[2]
    wlorig, upsignalorig, referenceorig = prepare_input.hyplant(pg,pw,wlmin=660)
    wl, upsignal, reference = prepare_input.hyplant(pg,pw,wlmin=windowmin,wlmax=windowmax)
    
        
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
            

    
    
        weights = wavelets.determine_weights(upsignal,newdecomp.scales)

        sfmmin = np.argmin(np.fabs(wlorig-670))
        sfmmax = np.argmin(np.fabs(wlorig-780))
        sfmWL = wlorig[sfmmin:sfmmax]
        x,Fsfm,Rsfm,resnorm, exitflag, nfevas,sfmres = SFM.FLOX_SpecFit_6C(sfmWL,referenceorig[sfmmin:sfmmax],upsignalorig[sfmmin:sfmmax],[1,1],1.,wl,alg='trf')
        
        sfm_residuals.append(sfmres)
        sunreference,_ = prepare_input.match_solspec(wl,0.3)
    
        coeffs = rpoly.optimize_coeffs(wl,reference,upsignal,p_init,newdecomp.scales,lbl=1)
        polyrefls = []
        cmap = plotting.get_colormap(len(coeffs))
        plt.figure()
        for i,polycoef in enumerate(coeffs[0]):
            
            interp = np.poly1d(polycoef)
            polyrefls.append(interp(wl))
            plt.plot(wl,interp(wl),color=cmap(i))
        

        
        
        polyrefls = np.array(polyrefls)
        polyR = np.average(polyrefls,weights=weights,axis=0)
        R_err = funcs.weighted_std(polyrefls,weights=weights,axis=0)

        
        
        
        plt.plot(wl,polyR,color='tab:red',label=r'Wavelet Reflectance')
        plt.plot(wl,appref,color='tab:blue',label=r'Apparent Reflectance')

        plt.plot(wl,Rsfm,label=r'Spectral Fitting Method',color='limegreen')
        plt.xlim(windowmin,windowmax)
        #plt.ylim(0.3,0.6)
        
        plt.legend()



        F_der = upsignal-polyR*reference
        

        
        resax1.plot(wl,polyR,color='forestgreen',linewidth=0.8,label='Reflectance')
        resax1.plot(wl,appref,'--',color='forestgreen',linewidth=0.8,label='Apparent Reflectance')
        resax2.plot(wl,F_der,color='tab:red',linewidth=0.8)
        
        resax2.plot(wl,Fsfm,'--',color='tab:red',linewidth=0.8)
        #datax.plot(wlorig,signalorig,color=cm(t),label=plottimes[t],linewidth=0.8)
        #datax.plot(wlorig,whitereferenceorig,color=cm(t),linewidth=0.8)
            


        F_param = np.polyfit(wl,F_der,2)
        Finterp = np.poly1d(F_param)
        F_smooth = Finterp(wl)
        Ftotal,F687,F760,Fr,wlFr,Ffr,wlFfr,special = funcs.evaluate_sif(wl,F_der,eval_wl)
        diurnal.append(special)
        Ftotal,F687,F760,Fr,wlFr,Ffr,wlFfr,special = funcs.evaluate_sif(wl,Fsfm,eval_wl)
        diurnalsfm.append(special)
        
        
        Fws.append(F_der)
        Fsfms.append(Fsfm)
        
        diurnalR.append(np.median(polyR))
        diurnalRsfm.append(np.median(Rsfm))
        diurnalRerrors.append(np.median(R_err))
        
    Fws = np.array(Fws)
    Fsfms = np.array(Fsfms)
    """ plt.figure()
    plt.plot(diurnal,np.divide(np.array(diurnal)-np.array(diurnalsfm),diurnal),'.',label='sfm')
    plt.plot(diurnal,np.divide(np.array(diurnal)-np.array(ifldb_ref),diurnal),'.',label='ifld')
    plt.legend() """

    """ reldevs_sfm = np.divide(np.array(diurnal)-np.array(diurnalsfm),diurnal) 
    num_reldevs = [reldevs_sfm[i] for i in range(len(reldevs_sfm)) if np.fabs(reldevs_sfm[i]) <= 0.1]
    frac_less01_sfm = len(num_reldevs)/len(reldevs_sfm)
    reldevs_ifld = np.divide(np.array(diurnal)-np.array(ifldb_ref),diurnal) 
    num_reldevs = [reldevs_ifld[i] for i in range(len(reldevs_ifld)) if np.fabs(reldevs_ifld[i]) <= 0.1]
    frac_less01_ifld = len(num_reldevs)/len(reldevs_ifld)

    print(frac_less01_sfm,frac_less01_ifld)


    plt.figure()
    plt.plot(diurnal,np.array(diurnal)-np.array(diurnalsfm),'.',label='sfm, absolute')
    plt.plot(diurnal,np.array(diurnal)-np.array(ifldb_ref),'.',label='ifld, absolute')
    plt.legend()

    absdevs_sfm = np.array(diurnal)-np.array(diurnalsfm)
    num_absdevs = [absdevs_sfm[i] for i in range(len(absdevs_sfm)) if np.fabs(absdevs_sfm[i]) <= 0.1]
    frac_less01_sfm = len(num_absdevs)/len(absdevs_sfm)
    absdevs_ifld = np.array(diurnal)-np.array(ifldb_ref)
    num_absdevs = [absdevs_ifld[i] for i in range(len(absdevs_ifld)) if np.fabs(absdevs_ifld[i]) <= 0.1]
    frac_less01_ifld = len(num_absdevs)/len(absdevs_ifld)

    print(frac_less01_sfm,frac_less01_ifld)
    sfm_residuals = np.array(sfm_residuals)
    mean_residual = np.mean(sfm_residuals,axis=0)
    sfmfig,sfmax = plt.subplots()
    sfmax.plot(sfmWL,mean_residual,label='Mean SFM residual')
    sfmax.plot(wlorig,(whitereferenceorig-np.mean(whitereferenceorig))/np.max(whitereferenceorig),label='Reference Spectrum')
    sfmax.legend()

    rmse_comp = [funcs.calc_rmse(Fws[:,i],Fsfms[:,i]) for i in range(len(Fws[0]))]
    rmsefig,rmseax = plt.subplots()
    rmseax.plot(wl,rmse_comp)
    rmseax.set_xlabel(r'$\lambda$ [nm]')
    rmseax.set_ylabel(r'$\mathrm{RMSD}_F$ [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
    figname5 = 'rmse_oens_{:d}{:d}_{}.pdf'.format(windowmin,windowmax,day)
    figname5tex = 'rmse_oens_{:d}{:d}_{}.tex'.format(windowmin,windowmax,day)
    rmsefig.savefig(figname5)
    tikzplotlib.save(figure=rmsefig,filepath=figname5tex)
    ones = np.linspace(0,np.max(diurnal),100)
    datax.set_xlabel(r'$\lambda$ [nm]')
    datax.set_ylabel(r'Radiance [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
    datax.legend()
    datfig.savefig('flox_examples_{}.pdf'.format(day)) """
    resax1.legend(loc='upper right')
    resax1.set_xlabel(r'$\lambda$ [nm]')
    resax2.set_xlabel(r'$\lambda$ [nm]')
    resax1.set_ylabel(r'Reflectance')
    resax2.set_ylabel(r'Fluorescence [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
    figname2 = 'HPrefl_oens_{:d}{:d}_{}_{}.pdf'.format(windowmin,windowmax,pg,pw)
    figname2tex = 'HPrefl_oens_{:d}{:d}_{}_{}.tex'.format(windowmin,windowmax,pg,pw)
    #tikzplotlib.clean_figure()
    tikzplotlib.save(figure=resfig,filepath=figname2tex)
    resfig.savefig(figname2)
    
    
    """ rsme_sfm = funcs.calc_rmse(np.array(diurnal),np.array(diurnalsfm))
    scatax.scatter(diurnal,diurnalsfm,label=r'SFM, RMSE = {:.2f}'.format(rsme_sfm),color='tab:red',s=20)
    scatax.plot(ones,ones,linestyle='dotted',color='black')
    
    difig,diax = plt.subplots(figsize=(8,7))
    ax2 = diax.twinx()
    diax.plot(plottimes,diurnal,label=r'$F_W$',color='tab:red',linewidth=0.8)
    diax.plot(plottimes,diurnalsfm,'--',label=r'$F_{{SFM}}$',color='tab:red',linewidth=0.8)
    if eval_wl == 687:
        diax.plot(plottimes,ifldb_ref,linestyle='dotted',label=r'$F_{{iFLD}}$',color='tab:red',linewidth=0.8)
        rsme_fldb = funcs.calc_rmse(np.array(diurnal),np.array(ifldb_ref))
        print(rsme_fldb)
        scatax.scatter(diurnal,ifldb_ref,label=r'iFLD, RMSE = {:.2f}'.format(rsme_fldb),color='tab:orange',s=20)
        
        
        #diax.fill_between(plottimes, np.array(ifldb_ref)-np.array(ifldb_errors), np.array(ifldb_ref)+np.array(ifldb_errors),color='tab:red',alpha=0.6)
    if eval_wl == 760:
        diax.plot(plottimes,iflda_ref,linestyle='dotted',label=r'$F_{{iFLD}}$',color='tab:red',linewidth=0.8)
        rsme_flda = funcs.calc_rmse(np.array(diurnal),np.array(iflda_ref))
        print(rsme_flda)
        scatax.scatter(diurnal,iflda_ref,label=r'iFLD, RMSE = {:.2f}'.format(rsme_flda),color='tab:orange',s=20)
        
        #diax.fill_between(plottimes, np.array(iflda_ref)-np.array(iflda_errors), np.array(iflda_ref)+np.array(iflda_errors),color='tab:red',alpha=0.6)
    scatax.set_ylabel(r'$F_{{{:d}}}$ [mW nm$^{{-1}}$ m$^{{-2}}$ ster$^{{-1}}$]'.format(eval_wl))
    scatax.set_xlabel(r'$F_{{{:d}, \textrm{{Wavelets}}}}$ [mW nm$^{{-1}}$ m$^{{-2}}$ ster$^{{-1}}$]'.format(eval_wl))
    scatax.legend()
    diax.fill_between(plottimes, np.array(diurnal)-np.array(diurnalFerrors), np.array(diurnal)+np.array(diurnalFerrors),color='tab:red',alpha=0.6)
    ax2.plot(plottimes,diurnalR,label=r'$R_W$',color='forestgreen',linewidth=0.8)
    ax2.fill_between(plottimes, np.array(diurnalR)-np.array(diurnalRerrors), np.array(diurnalR)+np.array(diurnalRerrors),color='forestgreen',alpha=0.6)
    #ax2.fill_between(plottimes, np.array(diurnalR)-np.array(diurnalappRerrors), np.array(diurnalR)+np.array(diurnalappRerrors),color='limegreen',alpha=0.6,linestyle='-.')
    ax2.plot(plottimes,diurnalRsfm,'--',label=r'$R_{{SFM}}$',color='forestgreen',linewidth=0.8)
    diax.set_xticks(plottimes[::20])
    diax.set_xticklabels(plottimes[::20],rotation=45)
    ax2.set_xticks(plottimes[::20])
    ax2.set_xticklabels(plottimes[::20],rotation=45)
    ax2.set_ylabel(r'Reflectance')
    diax.set_ylabel(r'$F_{{{:d}}}$ [mW nm$^{{-1}}$ m$^{{-2}}$ ster$^{{-1}}$]'.format(eval_wl))
    diax.set_xlabel(r'Time (UTC)')
    diax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    figname3 = 'diurnal_{:d}{:d}_{:d}_oens_{}_{}poly_lb.pdf'.format(windowmin,windowmax,eval_wl,day,polyorder)
    figname3tex = 'diurnal_{:d}{:d}_{:d}_oens_{}_{}poly_lb.tex'.format(windowmin,windowmax,eval_wl,day,polyorder)
    #tikzplotlib.clean_figure()
    tikzplotlib.save(figure=difig,filepath=figname3tex)
    difig.savefig(figname3)
    figname4 = 'scatter_{:d}_oens_{}.pdf'.format(eval_wl,day)
    figname4tex = 'scatter_{:d}_oens_{}.tex'.format(eval_wl,day)
    tikzplotlib.save(figure=scatfig,filepath=figname4tex)
    scatfig.savefig(figname4) """
    
    
    plt.show()

