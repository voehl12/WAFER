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

################### retrieval parameters ###############################

ranges = [[660,679,670],[681,695,687],[700,720,710],[725,740,735],[745,755,750],[754,773,760],[770,800,790]]
#ranges = [[681,695,687]]
site = 'Oensingen'
day = '2021-04-23'

jmin = -2.7
jmax = 4
nlevels = 2048
polyorder = 2

########################################################################


for range1 in ranges:
    print(range1)

    ############### data & results preparation #######################################

    windowmin = range1[0]
    windowmax = range1[1]
    eval_wl = range1[2]
    # Input preparation: for FloX data, the path to the input netCDF needs to be specified within the function 
    datapath = 'data/flox/FloX_JB023HT_S20210326_E20210610_C20210615.nc'
    times, wlorig, upseriesorig, downseriesorig, uperrorsorig, downerrorsorig, iflda_ref, iflda_errors, ifldb_ref, ifldb_errors = prepare_input.flox_allday(day,datapath,wlmin=660)
    times, wl, upseries, downseries, uperrors, downerrors, iflda_ref, iflda_errors, ifldb_ref, ifldb_errors = prepare_input.flox_allday(day,datapath,wlmin=windowmin,wlmax=windowmax)
    plottimes = times.dt.strftime('%H:%M').data

    ts_res = results.retrieval_res(site,day,windowmin,windowmax,eval_wl,'timeseries_poly2_R_F')
    ts_res.init_wl(wl)
    ts_res.initiate_ts_tofile()

    diurnal = []
    diurnalsfm = []
    diurnalR = []
    diurnalRsfm = []
    meanerrors = []
    diurnalRerrors = []
    diurnalFerrors = []
    diurnalappRerrors = []
    Fws = []
    Fsfms = []
    sfm_residuals = []
    
    cm = plotting.get_colormap(len(times))
    resfig,(resax1,resax2) = plt.subplots(1,2,figsize=(10,6))
    scatfig, scatax = plt.subplots()
    datfig, datax = plt.subplots()
        
    rangefig, rangeax = plt.subplots()
    rangeax.plot(wlorig,upseriesorig[10])
    rangeax.axvspan(windowmin, windowmax, color='green', alpha=0.5)
    rangeax.set_xlabel(r'Wavelength [nm]')
    rangeax.set_ylabel(r'Radiance [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
    figname1 = 'FloX_examplerange_{:d}{:d}_oens_{}.pdf'.format(windowmin,windowmax,day)
    figname1tex = 'FloX_examplerange_{:d}{:d}_oens_{}.tex'.format(windowmin,windowmax,day)
    tikzplotlib.save(figure=rangefig,filepath=figname1tex)
    rangefig.savefig(figname1)
    
    ########################################################################

        
    for t,time in enumerate(times):
        print(t,time.data)
        ############### data preparation #######################################
        uperror = uperrors[t]
        downerror = downerrors[t]
        meanerrors.append(np.mean(uperror))
        signalorig = upseriesorig[t]
        whitereferenceorig = downseriesorig[t]
        upsignal = upseries[t]
    
        whitereference = downseries[t]
        
        ############### iFLD and SFM #############
        ts_res.iFLD[0] = iflda_ref[t]
        ts_res.iFLD[1] = ifldb_ref[t]

        sfmmin = np.argmin(np.fabs(wlorig-windowmin))
        sfmmax = np.argmin(np.fabs(wlorig-windowmax))
        sfmWL = wlorig[sfmmin:sfmmax]
        x,Fsfm,Rsfm,resnorm, exitflag, nfevas,sfmres = SFM.FLOX_SpecFit_6C(sfmWL,whitereferenceorig[sfmmin:sfmmax],signalorig[sfmmin:sfmmax],[1,1],1.,wl,alg='trf')
        ts_res.Fsfm.spec = Fsfm
        sfm_residuals.append(sfmres)
        ts_res.Fsfm.evaluate_sif()
        diurnalsfm.append(ts_res.Fsfm.spec_val)
        
        ############### initial guess and reflectance optimization #############

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

            newdecomp = wavelets.decomp(jmin,jmax,nlevels)
            newdecomp.adjust_levels(upsignal)
            print(newdecomp.jmin,newdecomp.jmax,wavelets.get_wlscales(newdecomp.scales*(wl[1]-wl[0])))
        
        coeffs = rpoly.optimize_coeffs(wl,whitereference,upsignal,p_init,newdecomp)
        polyrefls = []
        cmap = plotting.get_colormap(len(coeffs))
        #plt.figure()
        for i,polycoef in enumerate(coeffs[0]):
            
            interp = np.poly1d(polycoef)
            polyrefls.append(interp(wl))
            #plt.plot(wl,interp(wl),color=cmap(i))
        #plt.plot(wl,polyR)
        polyrefls = np.array(polyrefls)
        
        ########################################################################
              
        ############# weighting and final reflectance ##########################

        newdecomp.calc_weights(upsignal)
        weights = newdecomp.weights
         
        polyR, R_err = funcs.weighted_std(polyrefls,weights=weights,axis=0)
        ts_res.R = polyR

        ########################################################################
        

        ############# SIF extraction and evaluation #############################

        F_der = upsignal-polyR*whitereference
        ts_res.F.spec = F_der
        ts_res.write_ts_tofile(plottimes[t])
        F_param = np.polyfit(wl,F_der,2)
        Finterp = np.poly1d(F_param)
        F_smooth = Finterp(wl)

        ts_res.F.evaluate_sif()
        diurnal.append(ts_res.F.spec_val)

        ########################################################################
        
        ################## result plotting #################################
        
        if t % 23 == 0:
            resax1.plot(wl,polyR,color=cm(t),linewidth=0.8,label=plottimes[t])
            resax1.plot(wl,appref,'--',color=cm(t),linewidth=0.8)
            resax2.plot(wl,F_der,color=cm(t),linewidth=0.8)
            
            resax2.plot(wl,Fsfm,'--',color=cm(t),linewidth=0.8)
            datax.plot(wlorig,signalorig,color=cm(t),label=plottimes[t],linewidth=0.8)
            datax.plot(wlorig,whitereferenceorig,color=cm(t),linewidth=0.8)
            
        ########################################################################

        ################## errors and statistics #################################
        
        appR_err = appref*np.sqrt(np.square(np.divide(uperror,upsignal)) + np.square(np.divide(downerror,whitereference)))
        F_err = np.sqrt(np.square(uperror) + np.square(np.multiply(whitereference,R_err)) + np.square(np.multiply(polyR,downerror)))
        Fws.append(F_der)
        Fsfms.append(Fsfm)
        
        diurnalR.append(np.median(polyR))
        diurnalRsfm.append(np.median(Rsfm))
        diurnalRerrors.append(np.median(R_err))
        diurnalFerrors.append(np.median(F_err))
        diurnalappRerrors.append(np.median(appR_err))

        ########################################################################
    

    Fws = np.array(Fws)
    Fsfms = np.array(Fsfms)

    ################## more plotting and statistics  #################################
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
    figname5 = 'rmse_oens_{:d}{:d}_{}_sfmwin.pdf'.format(windowmin,windowmax,day)
    figname5tex = 'rmse_oens_{:d}{:d}_{}_sfmwin.tex'.format(windowmin,windowmax,day)
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
    figname2 = 'refls_oens_{:d}{:d}_{}_{}poly_sfmwin.pdf'.format(windowmin,windowmax,day,polyorder)
    figname2tex = 'refls_oens_{:d}{:d}_{}_{}poly_sfmwin.tex'.format(windowmin,windowmax,day,polyorder)
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
    figname3 = 'diurnal_{:d}{:d}_{:d}_oens_{}_{}poly_sfmwin.pdf'.format(windowmin,windowmax,eval_wl,day,polyorder)
    figname3tex = 'diurnal_{:d}{:d}_{:d}_oens_{}_{}poly_sfmwin.tex'.format(windowmin,windowmax,eval_wl,day,polyorder)
    #tikzplotlib.clean_figure()
    tikzplotlib.save(figure=difig,filepath=figname3tex)
    difig.savefig(figname3)
    figname4 = 'scatter_{:d}_oens_{}_sfmwin.pdf'.format(eval_wl,day)
    figname4tex = 'scatter_{:d}_oens_{}_sfmwin.tex'.format(eval_wl,day)
    tikzplotlib.save(figure=scatfig,filepath=figname4tex)
    scatfig.savefig(figname4)
    
    
    

