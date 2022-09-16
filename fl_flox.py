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


ranges = [[660,679,670],[681,695,687],[700,720,710],[725,740,735],[745,755,750],[754,773,760],[770,800,790]]
ranges = [[681,695,687]]
#ranges = [[684,702,687],[745,757,750],[753,775,760]]
#ranges = [[684,702,687],[753,775,760]]
day = '2021-04-23'
for range1 in ranges:
    print(range1)
    windowmin = range1[0]
    windowmax = range1[1]
    eval_wl = range1[2]
    times, wlorig, upseriesorig, downseriesorig, uperrorsorig, downerrorsorig, iflda_ref, iflda_errors, ifldb_ref, ifldb_errors = prepare_input.flox_allday(day,wlmin=660)
    times, wl, upseries, downseries, uperrors, downerrors, iflda_ref, iflda_errors, ifldb_ref, ifldb_errors = prepare_input.flox_allday(day,wlmin=windowmin,wlmax=windowmax)
    plottimes = times.dt.strftime('%H:%M').data

    ts_res = results.retrieval_res('Oensingen',day,windowmin,windowmax,eval_wl,'timeseries_poly2_R_F')
    ts_res.initiate_ts_tofile()
    ts_res.init_wl(wl)

        
    rangefig, rangeax = plt.subplots()
    rangeax.plot(wlorig,upseriesorig[10])
    rangeax.axvspan(windowmin, windowmax, color='green', alpha=0.5)
    rangeax.set_xlabel(r'Wavelength [nm]')
    rangeax.set_ylabel(r'Radiance [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
    figname1 = 'FloX_examplerange_{:d}{:d}_oens_{}.pdf'.format(windowmin,windowmax,day)
    figname1tex = 'FloX_examplerange_{:d}{:d}_oens_{}.tex'.format(windowmin,windowmax,day)
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
    testjmin = -2.7
    testjmax = 4
    testnlevels = 2048
    cm = plotting.get_colormap(len(times))
    resfig,(resax1,resax2) = plt.subplots(1,2,figsize=(10,6))
    scatfig, scatax = plt.subplots()
    datfig, datax = plt.subplots()
    Fws = []
    Fsfms = []
    sfm_residuals = []
    for t,time in enumerate(times):
        print(t,time.data)
        
        uperror = uperrors[t]
        downerror = downerrors[t]
        meanerrors.append(np.mean(uperror))
        signalorig = upseriesorig[t]
        whitereferenceorig = downseriesorig[t]
        upsignal = upseries[t]
    
        whitereference = downseries[t]
        
        iflda = iflda_ref[t]
        appref = np.divide(upsignal,whitereference)


        
        nopeak_wl, nopeak_appref = prepare_input.rm_peak(wl,appref) 

        if len(nopeak_appref) < len(wl):
            pfit = np.polyfit(nopeak_wl,nopeak_appref,2)
            pinterp = np.poly1d(pfit)
            nopeak_appref = pinterp(wl)
            nopeak_wl = wl
    
        

        p_init = np.polyfit(nopeak_wl,nopeak_appref,polyorder)
        interp = np.poly1d(p_init)
        poly_R_init = interp(wl)

    
        
                
            
        if t == 0:

            newdecomp = wavelets.decomp(testjmin,testjmax,testnlevels)
            newdecomp.adjust_levels(upsignal)
            print(newdecomp.jmin,newdecomp.jmax,wavelets.get_wlscales(newdecomp.scales*(wl[1]-wl[0])))
            

    
    
        newdecomp.calc_weights(upsignal)
        weights = newdecomp.weights

        sfmmin = np.argmin(np.fabs(wlorig-670))
        sfmmax = np.argmin(np.fabs(wlorig-780))
        sfmWL = wlorig[sfmmin:sfmmax]
        x,Fsfm,Rsfm,resnorm, exitflag, nfevas,sfmres = SFM.FLOX_SpecFit_6C(sfmWL,whitereferenceorig[sfmmin:sfmmax],signalorig[sfmmin:sfmmax],[1,1],1.,wl,alg='trf')
        ts_res.Fsfm[0] = Fsfm
        sfm_residuals.append(sfmres)
        sunreference,_ = prepare_input.match_solspec(wl,0.3)
    
        coeffs = rpoly.optimize_coeffs(wl,whitereference,upsignal,p_init,newdecomp)
        #coeffs = np.array(coeffs)
        polyrefls = []
        cmap = plotting.get_colormap(len(coeffs))
        #plt.figure()
        for i,polycoef in enumerate(coeffs[0]):
            
            interp = np.poly1d(polycoef)
            polyrefls.append(interp(wl))
            #plt.plot(wl,interp(wl),color=cmap(i))
        

        
        
        polyrefls = np.array(polyrefls)
        polyR, R_err = funcs.weighted_std(polyrefls,weights=weights,axis=0)
        ts_res.R = polyR
        

        appR_err = appref*np.sqrt(np.square(np.divide(uperror,upsignal)) + np.square(np.divide(downerror,whitereference)))
        #plt.plot(wl,polyR)
        
        """ plt.plot(wl,polyR,color='tab:red',label=r'Wavelet Reflectance')
        plt.plot(wl,appref,color='tab:blue',label=r'Apparent Reflectance')

        plt.plot(wl,Rsfm,label=r'Spectral Fitting Method',color='limegreen')
        plt.xlim(670,780)
        plt.ylim(0.3,0.6)
        
        plt.legend() """



        F_der = upsignal-polyR*whitereference
        ts_res.F[0] = F_der
        ts_res.write_ts_tofile(plottimes[t])
        

        F_err = np.sqrt(np.square(uperror) + np.square(np.multiply(whitereference,R_err)) + np.square(np.multiply(polyR,downerror)))
        if t % 23 == 0:
            resax1.plot(wl,polyR,color=cm(t),linewidth=0.8,label=plottimes[t])
            resax1.plot(wl,appref,'--',color=cm(t),linewidth=0.8)
            resax2.plot(wl,F_der,color=cm(t),linewidth=0.8)
            
            resax2.plot(wl,Fsfm,'--',color=cm(t),linewidth=0.8)
            datax.plot(wlorig,signalorig,color=cm(t),label=plottimes[t],linewidth=0.8)
            datax.plot(wlorig,whitereferenceorig,color=cm(t),linewidth=0.8)
            


        F_param = np.polyfit(wl,F_der,2)
        Finterp = np.poly1d(F_param)
        F_smooth = Finterp(wl)

        ts_res.evaluate_sif(ts_res.F)
        diurnal.append(ts_res.F[1][7])
        ts_res.evaluate_sif(ts_res.Fsfm)
        diurnalsfm.append(ts_res.Fsfm[1][7])
        
        
        Fws.append(F_der)
        Fsfms.append(Fsfm)
        
        diurnalR.append(np.median(polyR))
        diurnalRsfm.append(np.median(Rsfm))
        diurnalRerrors.append(np.median(R_err))
        diurnalFerrors.append(np.median(F_err))
        diurnalappRerrors.append(np.median(appR_err))
    
        """ plt.figure()
        plt.plot(wl,F_der,'--',label=r'Wavelet derived',color='tab:red',alpha=0.5)
        plt.plot(wl,F_smooth,label=r'Polynomial',color='tab:red')
        plt.plot(wl,Fsfm,label=r'Spectral Fitting Method',color='tab:blue')
        plt.plot(760,ifldaref,'o',color='tab:blue')
        plt.legend()
        plt.title(time)
        plt.ylim(0,6)
        plt.show()
    
        """
    Fws = np.array(Fws)
    Fsfms = np.array(Fsfms)
    plt.figure()
    plt.plot(diurnal,np.divide(np.array(diurnal)-np.array(diurnalsfm),diurnal),'.',label='sfm')
    plt.plot(diurnal,np.divide(np.array(diurnal)-np.array(ifldb_ref),diurnal),'.',label='ifld')
    plt.legend()

    reldevs_sfm = np.divide(np.array(diurnal)-np.array(diurnalsfm),diurnal) 
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
    datfig.savefig('flox_examples_{}.pdf'.format(day))
    resax1.legend(loc='upper right')
    resax1.set_xlabel(r'$\lambda$ [nm]')
    resax2.set_xlabel(r'$\lambda$ [nm]')
    resax1.set_ylabel(r'Reflectance')
    resax2.set_ylabel(r'Fluorescence [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
    figname2 = 'refls_oens_{:d}{:d}_{}_{}poly_lb.pdf'.format(windowmin,windowmax,day,polyorder)
    figname2tex = 'refls_oens_{:d}{:d}_{}_{}poly_lb.tex'.format(windowmin,windowmax,day,polyorder)
    #tikzplotlib.clean_figure()
    tikzplotlib.save(figure=resfig,filepath=figname2tex)
    resfig.savefig(figname2)
    
    
    rsme_sfm = funcs.calc_rmse(np.array(diurnal),np.array(diurnalsfm))
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
    scatfig.savefig(figname4)
    
    
    plt.show()

