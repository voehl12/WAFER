from scipy import interpolate
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import xarray as xr


""" 
functions to prepare data for the wavelet retrieval.

combining functions to prepare arrays of radiances for SIF retrieval depending on the dataset
the setup and whether the functions are useful completely depends on how the data is stored. 

additionally, there are functions to add noise and remove the peaks in the apparent reflectance for the inital guess of the reflectance

"""

def synthetic(cab,lai,feffef,wlmin=0,wlmax=1000,completedir='data/scope/'):
    completename = completedir+'radcomplete_{}_{:d}_{:d}_ae_conv.dat'.format(feffef,cab,lai)
    woFname = completedir+'radwoF_{}_{:d}_{:d}_ae_conv.dat'.format(feffef,cab,lai)
    reflname = completedir+'refl/rho_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    Fname = completedir+'Fcomp_{}_{:d}_{:d}_ae.dat'.format(feffef,cab,lai) 
    whiterefname = completedir+'whiteref_{}_{:d}_{:d}_ae_conv.dat'.format(feffef,cab,lai)
    wlRname = completedir+'wlR'
    wlFname = completedir+'wlF'
    filenames = [completename,whiterefname,reflname,Fname,wlRname,wlFname,woFname]
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

    wl = []
    with open(completedir+'radcomplete_002_5_7_ae_conv.dat','r') as g:
        for line in g:
            line = line.split()
            wl.append(float(line[0]))
    wl = np.array(wl)
    noconvwl = []
    with open(completedir+'wl_array','r') as ncw:
        for line in ncw:
            line = line.split()
            noconvwl.append(float(line[0]))
    noconvwl = np.array(noconvwl)

    

    signal, whitereference, refl, F, wlR, wlF, noF = array_list[0], array_list[1], array_list[2], array_list[3], array_list[4], array_list[5], array_list[6]

    interR = interpolate.interp1d(wlR, refl,kind='cubic')
    refl = interR(wl)

    interF = interpolate.interp1d(noconvwl, F, kind='cubic')
    F = interF(wl)
    if wlmin == 0:
        wlmin = wl[0]
    if wlmax == 1000:
        wlmax = wl[-1]
    startind = np.argmin(np.fabs(wl-wlmin))
    endind = np.argmin(np.fabs(wl-wlmax))

    wl, signal, whitereference, refl, F, noF = wl[startind:endind], signal[startind:endind], whitereference[startind:endind], refl[startind:endind], F[startind:endind],noF[startind:endind]

    return wl, signal, whitereference, refl, F,noF


def hyplant(pixelg,pixelw,wlmin=0,wlmax=1000,path='../../Data/20200625-OEN-1132-1800-L1-S-FLUO_radiance_deconv_i1.dat'):

    def get_wavelengths():
        wlpath = '../../Data/20200625-OEN-1132-1800-L1-S-FLUO_radiance_deconv_i1.hdr'
        wl = []
        with open(wlpath, 'r') as w:
            for line in w:
                linep = line.split()
                if linep[0] == 'samples':
                    width = int(linep[2])
                    print(width)
                elif linep[0] == 'lines':
                    length = int(linep[2])
                    print(length)
                elif linep[0] == 'bands':
                    bands = int(linep[2])
                    print(bands)
                elif len(wl) < 1024:
                    line = line.split(',')
                    try: 
                        sw0 = float(line[0].strip())
                        for val in line:

                            wl.append(float(val.strip()))
                    except:
                        continue
                else:
                    break
             
        return wl,width,length,bands
    
    def call_data(path):
        return np.fromfile(path, dtype=np.uint16)
    
    def find_spectrum(pixel,data,bands,width):
        spectrum = []
        line = np.floor(pixel / width) 
        column = pixel % width
        start_id = int(line * width * bands + column)
        end_id = int(start_id + (bands-1) * width)
        #print((end_id-start_id)/bands)
        ids = np.linspace(start_id,end_id,bands,dtype=int)
        #print(ids)
        for num in ids:
            spectrum.append(data[num])
        return np.array(spectrum)

    data = call_data(path)
    wl,width,length,bands = get_wavelengths()
    wl = np.array(wl)
    startind = np.argmin(np.fabs(wl-wlmin))
    endind = np.argmin(np.fabs(wl-wlmax))
    
    # quick solution to average over multiple pixels, only use this way, when selected pixel is savely in the middle of a large patch with expectedly similar spectra!!
    # better would be to look for correlated spectra automatically, which has been implemented, so please get in touch if this is needed. 
    shift = [-2,-1,0,1,2]
    upwelling_samples = np.array([find_spectrum(pixelg+shift[i],data,bands,width) for i in range(len(shift))])
    upwelling = np.mean(upwelling_samples,axis=0)
    reference_samples = np.array([find_spectrum(pixelw+shift[i],data,bands,width) for i in range(len(shift))])
    reference = np.mean(reference_samples,axis=0)
    
    plt.figure()
    plt.plot(wl,upwelling)
    plt.plot(wl,reference)
    plt.show()

        
    return wl[startind:endind], upwelling[startind:endind]/100, reference[startind:endind]/100


def flox_single(day,time,wlmin=0,wlmax=1000,datapath='data/flox/FloX_JB023HT_S20210326_E20210610_C20210615.nc'):
    day = '2021-04-23'
    timestampbegin = day+' 05:00:00'
    timestampend = day+' 17:00:00'
    timestamp = day+' '+time  
    fluodata = xr.open_dataset(datapath, group="FLUO")
    metadata = xr.open_dataset(datapath,group='METADATA')
    wlfluo = np.array(fluodata["wavelengths"])
    upseries = fluodata["upwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").mean("time").sel(time=timestamp,method='nearest')
    downseries = fluodata["downwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").mean("time").sel(time=timestamp,method='nearest')
    uperrors = fluodata["upwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").std("time").sel(time=timestamp,method='nearest')
    downerrors = fluodata["downwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").std("time").sel(time=timestamp,method='nearest')
    iflda_ref = metadata['SIF_A_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").mean("time").sel(time=timestamp,method='nearest')

    wl = wlfluo
    if wlmin == 0:
        wlmin = wl[0]
    if wlmax == 1000:
        wlmax = wl[-1]
    startind = np.argmin(np.fabs(wl-wlmin))
    endind = np.argmin(np.fabs(wl-wlmax))

    wl, signal, whitereference,error = wl[startind:endind], upseries[startind:endind], downseries[startind:endind], np.sqrt((np.square(np.divide(uperrors,upseries)) + np.square(np.divide(downerrors,downseries))))[startind:endind]
    time = upseries.time

    return wl, signal*1000,whitereference*1000,iflda_ref,error,time,uperrors[startind:endind]

def flox_allday(day,datapath='data/flox/FloX_JB023HT_S20210326_E20210610_C20210615.nc',wlmin=0,wlmax=1000):
    timestampbegin = day+' 04:15:00'
    timestampend = day+' 16:50:00'
    
    fluodata = xr.open_dataset(datapath, group="FLUO")
    metadata = xr.open_dataset(datapath,group='METADATA')
    wl = np.array(fluodata["wavelengths"])
    if wlmin == 0:
        wlmin = wl[0]
    if wlmax == 1000:
        wlmax = wl[-1]
    wlfluo = np.array(fluodata["wavelengths"].sel(wavelengths=slice(wlmin,wlmax)))
    wl = wlfluo
    startind = np.argmin(np.fabs(wl-wlmin))
    endind = np.argmin(np.fabs(wl-wlmax))

    upseries = fluodata["upwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="5Min").mean("time").sel(wavelengths=slice(wlmin,wlmax))
    downseries = fluodata["downwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="5Min").mean("time").sel(wavelengths=slice(wlmin,wlmax))
    uperrors = fluodata["upwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="5Min").std("time").sel(wavelengths=slice(wlmin,wlmax))
    downerrors = fluodata["downwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="5Min").std("time").sel(wavelengths=slice(wlmin,wlmax))
    # if using new netcdf format (current jb_cli), metadata does not need to be called separately and ifld values collected with the following line instead:
    #iflda_ref = fluodata['sif_a_ifld'].sel(time=slice(timestampbegin, timestampend)).resample(time="5Min").mean("time")
    iflda_ref = metadata['SIF_A_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="5Min").mean("time")
    iflda_errors = metadata['SIF_A_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="5Min").std("time")
    ifldb_ref = metadata['SIF_B_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="5Min").mean("time")
    ifldb_errors = metadata['SIF_B_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="5Min").std("time")

    times = upseries.time
    upseries, downseries, uperrors, downerrors = np.transpose(upseries.data), np.transpose(downseries.data), np.transpose(uperrors.data), np.transpose(downerrors.data)

    return times, wl, upseries*1000, downseries*1000, uperrors, downerrors, iflda_ref, iflda_errors, ifldb_ref, ifldb_errors


def add_noise(data,snr,switch,N=10):
    sigmas = []
    if switch == 0:
        sigmas = np.zeros(len(data))
        return sigmas,data
    else:
        noisydata = np.zeros((N,len(data)))
        for j in range(N):

            s0 = 0
            newdata = np.zeros(len(data))
            #SNR = []
            for i,val in enumerate(data):
                sigma = val/snr
                #sigma = 0.3
                #sigma = 1.0
                #snr = val/sigma
                s0 = np.random.normal(0.0, sigma)
                #SNR.append(snr)
                sigmas.append(s0)
                newval = val + s0
                newdata[i] = newval
            noisydata[j] = newdata
        newdata = np.mean(noisydata,axis=0)
        stds = np.std(noisydata,axis=0)
        return stds,newdata


def deconvolve(wl,signal,packlen=8):
    def gauss_fft(N,sigma):
        res = np.ones(N)
        vals = scipy.signal.gaussian(N,1/(sigma))
        res[0] = vals[N//2]
        res[1:N//2] = vals[N//2:-1]
        res[N//2+1:] = vals[1:N//2]
        if N % 2 == 0:
            res[N//2] = vals[0] + vals[-1]
        return res*2*np.sqrt(np.pi/(2/sigma**2))
    pixelfwhm = 0.3/(wl[1]-wl[0])
    sigma = pixelfwhm/2.355
    
    numpack = int(len(signal)/packlen)
    gaussfft = gauss_fft(packlen,sigma)
    my_cmap = cm.get_cmap('viridis',numpack)
    all_deconv = np.zeros(len(signal))
    count = 0
    for i in range(len(signal)-packlen):
        startind = i
        deconv = np.zeros(len(signal))
        signal_short = signal[startind:startind+packlen]
        
        signal_fft = (np.fft.fft(signal_short))
       

        
        packdeconv = np.abs(np.fft.ifft(np.multiply(signal_fft,gaussfft)/(np.multiply(gaussfft,gaussfft)+0.002)))
        deconv[startind:startind+packlen] = packdeconv
        all_deconv += deconv
        count += 1
        
    all_deconv = all_deconv/count
    ratio = np.divide(signal,all_deconv)

    return all_deconv*np.median(ratio)


def spec_respons(wl,rawdata,wlmin,wlmax,Nsamples,fwhm):
    wlfine = np.linspace(wl[0],wl[-1],num=100000)
    finessi = wlfine[1]-wlfine[0]
    intrad = interpolate.interp1d(wl,rawdata,kind='nearest')
    finerad = intrad(wlfine)
    samples = np.linspace(wlmin,wlmax,num=Nsamples)
    pixelfwhm = fwhm/finessi
    sigma = pixelfwhm/2.355
    gauss = scipy.signal.gaussian(int(10*sigma),sigma)
    convolution = scipy.signal.convolve(finerad,gauss,mode='same') / np.sum(gauss)

    sampleinds = [np.argmin(np.fabs(wlfine-samples[i])) for i in range(len(samples))]
    convsignal = convolution[sampleinds]
    
    return samples,convsignal,gauss

def match_solspec(wl,fwhm,path='../../Data/Sun/SUN001kurucz.dat'):
    swl = []
    srad = []
    with open(path,'r') as f:
        for line in f:
            line = line.split()
            try:
                freq = float(line[0])
                if freq == 0.0:
                    continue
            except:
                continue
            
            w = freq
            if w >= 200 and w <= 1000:
                swl.append(w)
                srad.append(float(line[1]) * 1000 / np.pi)
                # converted to mW/(m^2 nm ster)
            else:
                continue

    swl, srad = np.array(swl), np.array(srad)
    swl, sradnew, gaussf= spec_respons(swl,srad,wl[0],wl[-1],len(wl),fwhm)
    return sradnew,gaussf

def rm_peak(wl,appref):
    minpeaka = np.argmin(np.fabs(wl-758))
    maxpeaka = np.argmin(np.fabs(wl-769))
    minpeakb = np.argmin(np.fabs(wl-685))
    maxpeakb = np.argmin(np.fabs(wl-690))
    nopeak_appref = []
    nopeak_wl = []
    for i in range(len(wl)):
        if np.logical_and(i < minpeaka or i > maxpeaka,i < minpeakb or i > maxpeakb):
            nopeak_appref.append(appref[i])
            nopeak_wl.append(wl[i])
    
    return nopeak_wl,nopeak_appref