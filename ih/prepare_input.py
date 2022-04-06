from scipy import interpolate
import numpy as np

def synthetic(cab,lai,feffef):
    completename = '../cwavelets/libradtranscope/floxseries_ae_oen/reflectance/radcomplete_{}_{:d}_{:d}_ae_conv.dat'.format(feffef,cab,lai)
    woFname = '../cwavelets/libradtranscope/floxseries_ae_oen/reflectance/radwoF_{}_{:d}_{:d}_ae_conv.dat'.format(feffef,cab,lai)
    reflname = '../reflectance/szamatch/rho_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    albedoname = '../reflectance/szamatch/albedo_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    scoperef = '../LupSCOPE/szamatch/Lup_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    Fname = '../cwavelets/libradtranscope/floxseries_ae_oen/reflectance/Fcomp_{}_{:d}_{:d}_ae.dat'.format(feffef,cab,lai) #'fluorescence/F_scope_{}_{:d}_{:d}'.format(feffef,cab,lai)
    
    wlRname = '../reflectance/szamatch/wlR'
    wlFname = '../reflectance/szamatch/wlF'
    filenames = [completename,woFname,reflname,albedoname,Fname,wlRname,wlFname,scoperef]
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
    with open('../cwavelets/libradtranscope/floxseries_ae_oen/radcomplete_004_5_7_ae_conv.dat','r') as g:
        for line in g:
            line = line.split()
            wl.append(float(line[0]))
    wl = np.array(wl)
    noconvwl = []
    with open('../cwavelets/libradtranscope/series/wl_array','r') as ncw:
        for line in ncw:
            line = line.split()
            noconvwl.append(float(line[0]))
    noconvwl = np.array(noconvwl)

    refname = '../cwavelets/libradtranscope/floxseries_ae_oen/whiteref_ae_conv.dat'
    whitereference = []
    with open(refname,'r') as wf:
        for k,line in enumerate(wf):
            line = line.split()
            whitereference.append(float(line[0]))
    whitereference = np.array(whitereference)

    
    

    signal, refl, F, wlR, wlF = array_list[0], array_list[2],array_list[4],array_list[5],array_list[6]
    interR = interpolate.interp1d(wlR, refl,kind='cubic')
    refl = interR(wl)

    interF = interpolate.interp1d(noconvwl, F, kind='cubic')
    F = interF(wl)

    return wl, signal, whitereference, refl, F


def hyplant():
    completename = '../cwavelets/Hyplant/spectrum_SIF_high.dat'
    
    
    filenames = [completename]
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
    wl = get_wavelengths('../../Data/hyplant2020_wl.csv')
    wl = np.array(wl)
    refname = '../cwavelets/Hyplant/compare_reference.dat'
    whitereference = []
    with open(refname,'r') as wf:
        for k,line in enumerate(wf):
            line = line.split()
            whitereference.append(float(line[0]))
    whitereference = np.array(whitereference)
    return wl, array_list, whitereference


def flox():
    day = '2021-05-30'
    timestampbegin = day+' 05:00:00'
    timestampend = day+' 17:00:00'
    #datapath = "../FloX_Davos/SDcard/FloX_JB038AD_S20210603_E20211119_C20211208.nc"
    datapath = "../../FloX_Davos/FloX_JB023HT_S20210326_E20210610_C20210615.nc"
    fluodata = xr.open_dataset(datapath, group="FLUO")
    metadata = xr.open_dataset(datapath,group='METADATA')
    wlfluo = np.array(fluodata["wavelengths"])
    upseries = fluodata["upwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").mean("time")
    downseries = fluodata["downwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").mean("time")
    uperrors = fluodata["upwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").std("time")
    downerrors = fluodata["downwelling"].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").std("time")
    iflda_ref = metadata['SIF_A_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").mean("time")
    iflda_err = metadata['SIF_A_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").std("time")
    ifldb_ref = metadata['SIF_B_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").mean("time")
    ifldb_err = metadata['SIF_B_ifld [mW m-2nm-1sr-1]'].sel(time=slice(timestampbegin, timestampend)).resample(time="15Min").std("time")
    int_num = len(upseries)

    return wlfluo, upseries,downseries