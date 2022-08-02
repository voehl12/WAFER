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
from matplotlib import rc
#plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})


ranges = [[665,690,684],[684,705,687],[700,730,710],[720,747,735],[740,757,750],[753,780,760],[775,800,790]]

day = '2021-04-23'
for range in ranges:
    print(range)
    windowmin = range[0]
    windowmax = range[1]
    eval_wl = range[2]
    times, wlorig, upseriesorig, downseriesorig, uperrorsorig, downerrorsorig, _ = prepare_input.flox_allday(day,wlmin=670)
    times, wl, upseries, downseries, uperrors, downerrors, iflda_ref = prepare_input.flox_allday(day,wlmin=windowmin,wlmax=windowmax)
    plottimes = times.dt.strftime('%H:%M').data

    filename = 'timeseries_R_F_'+day+'_{:d}{:d}.txt'.format(windowmin,windowmax)
    #filename = 'trash.txt'
    with open(filename,'w') as f:
        print('**Oensingen',file=f)
        print('**range: {:d}-{:d} nm'.format(windowmin,windowmax),file=f)
        print('**day: '+day,file=f)
        print('time',file=f,end=',')
        for w in wl:
            print(float(w),file=f,end=',')
        print(file=f)
        
    plt.figure()
    plt.plot(wlorig,upseriesorig[10])
    plt.axvspan(windowmin, windowmax, color='green', alpha=0.5)
    plt.xlabel(r'Wavelength [nm]')
    plt.ylabel(r'Radiance [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
    figname1 = 'FloX_examplerange_{:d}{:d}_oens_{}.pdf'.format(windowmin,windowmax,day)
    plt.savefig(figname1)


    polyorder = 2
    diurnal = []
    diurnalsfm = []
    diurnalR = []
    diurnalRsfm = []
    meanerrors = []
    diurnalRerrors = []
    diurnalFerrors = []
    diurnalappRerrors = []
    testjmin = -2
    testjmax = 4
    testnlevels = 2048
    cm = plotting.get_colormap(len(times))
    resfig,(resax1,resax2) = plt.subplots(1,2,figsize=(10,6))
    for t,time in enumerate(times):
        print(t,time.data)
        with open(filename,'a') as f:
            print('R',file=f,end=',')
            print(plottimes[t],file=f,end=',')
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
    

        p_init = np.polyfit(nopeak_wl,nopeak_appref,polyorder)
        interp = np.poly1d(p_init)
        poly_R_init = interp(wl)
    
        
                
            
        if t == 0:

            newdecomp = wavelets.decomp(testjmin,testjmax,testnlevels)
            newdecomp.adjust_levels(upsignal)
            print(newdecomp.jmin,newdecomp.jmax)
            

    
    
        weights = wavelets.determine_weights(upsignal,newdecomp.scales)

        sfmmin = np.argmin(np.fabs(wlorig-670))
        sfmmax = np.argmin(np.fabs(wlorig-780))
        sfmWL = wlorig[sfmmin:sfmmax]
        x,Fsfm,Rsfm,resnorm, exitflag, nfevas,sfmres = SFM.FLOX_SpecFit_6C(sfmWL,whitereferenceorig[sfmmin:sfmmax],signalorig[sfmmin:sfmmax],[1,1],1.,wl,alg='trf')

        sunreference,_ = prepare_input.match_solspec(wl,0.3)
    
        coeffs = rpoly.optimize_coeffs(wl,whitereference,upsignal,p_init,newdecomp.scales,lbl=1)
        #coeffs = np.array(coeffs)
        polyrefls = []
        cmap = plotting.get_colormap(len(coeffs))
        #plt.figure()
        for i,polycoef in enumerate(coeffs[0]):
            
            interp = np.poly1d(polycoef)
            polyrefls.append(interp(wl))
            #plt.plot(wl,interp(wl),color=cmap(i))
        

        
        
        polyrefls = np.array(polyrefls)
        polyR = np.average(polyrefls,weights=weights,axis=0)
        R_err = funcs.weighted_std(polyrefls,weights=weights,axis=0)

        appR_err = appref*np.sqrt(np.square(np.divide(uperror,upsignal)) + np.square(np.divide(downerror,whitereference)))
        #plt.plot(wl,polyR)
        
        """ plt.plot(wl,polyR,color='tab:red',label=r'Wavelet Reflectance')
        plt.plot(wl,appref,color='tab:blue',label=r'Apparent Reflectance')

        plt.plot(wl,Rsfm,label=r'Spectral Fitting Method',color='limegreen')
        plt.xlim(670,780)
        plt.ylim(0.3,0.6)
        
        plt.legend() """



        F_der = upsignal-polyR*whitereference
        with open(filename,'a') as f:
            for val in polyR:
                print(val,file=f,end=',')
            print(file=f)
            print('F',file=f,end=',')
            print(plottimes[t],file=f,end=',')
            for val in F_der:
                print(val,file=f,end=',')
            print(file=f)

        F_err = np.sqrt(np.square(uperror) + np.square(np.multiply(whitereference,R_err)) + np.square(np.multiply(polyR,downerror)))
        if t % 10 == 0:
            resax1.plot(wl,polyR,color=cm(t),linewidth=0.8,label=plottimes[t])
            resax1.plot(wl,appref,'--',color=cm(t),linewidth=0.8)
            resax2.plot(wl,F_der,color=cm(t),linewidth=0.8)
            resax2.plot(wl,Fsfm,'--',color=cm(t),linewidth=0.8)

        F_param = np.polyfit(wl,F_der,2)
        Finterp = np.poly1d(F_param)
        F_smooth = Finterp(wl)
        Ftotal,F687,F760,Fr,wlFr,Ffr,wlFfr,special = funcs.evaluate_sif(wl,F_der,eval_wl)
        diurnal.append(special)
        Ftotal,F687,F760,Fr,wlFr,Ffr,wlFfr,special = funcs.evaluate_sif(wl,Fsfm,eval_wl)
        diurnalsfm.append(special)
        
        

        
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
    

    resax1.legend(loc='upper right')
    resax1.set_xlabel(r'$\lambda$ [nm]')
    resax2.set_xlabel(r'$\lambda$ [nm]')
    resax1.set_ylabel(r'Reflectance')
    resax2.set_ylabel(r'Fluorescence [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
    figname2 = 'refls_oens_{:d}{:d}_{}_2poly.pdf'.format(windowmin,windowmax,day)
    resfig.savefig(figname2)
    difig,diax = plt.subplots(figsize=(8,7))
    ax2 = diax.twinx()
    diax.plot(plottimes,diurnal,label=r'$F_W$',color='tab:red',linewidth=0.8)
    diax.plot(plottimes,diurnalsfm,'--',label=r'$F_{{SFM}}$',color='tab:red',linewidth=0.8)
    diax.fill_between(plottimes, np.array(diurnal)-np.array(diurnalFerrors), np.array(diurnal)+np.array(diurnalFerrors),color='tab:red',alpha=0.6)
    ax2.plot(plottimes,diurnalR,label=r'$R_W$',color='forestgreen',linewidth=0.8)
    ax2.fill_between(plottimes, np.array(diurnalR)-np.array(diurnalRerrors), np.array(diurnalR)+np.array(diurnalRerrors),color='forestgreen',alpha=0.6)
    #ax2.fill_between(plottimes, np.array(diurnalR)-np.array(diurnalappRerrors), np.array(diurnalR)+np.array(diurnalappRerrors),color='limegreen',alpha=0.6,linestyle='-.')
    ax2.plot(plottimes,diurnalRsfm,'--',label=r'$R_{{SFM}}$',color='forestgreen',linewidth=0.8)
    diax.set_xticks(plottimes[::20])
    diax.set_xticklabels(plottimes[::20],rotation=45)
    ax2.set_ylabel(r'Reflectance')
    diax.set_ylabel(r'Fluorescence [mW nm$^{-1}$ m$^{-2}$ ster$^{-1}$]')
    diax.set_xlabel(r'Time (UTC)')
    diax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    figname3 = 'diurnal_{:d}{:d}_{:d}_oens_{}_2poly.pdf'.format(windowmin,windowmax,eval_wl,day)
    difig.savefig(figname3)

