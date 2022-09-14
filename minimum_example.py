import numpy as np
from scipy import interpolate, signal
from r_opts import rspline,rtanh,rpoly
from ih import prepare_input
import matplotlib.pyplot as plt
import scipy
from utils import wavelets,plotting, funcs
from SFM import SFM,SFM_BSpline
from scipy import optimize




mineval,maxeval,eval_wl = 754,773,760
polyorder = 2

cab = 20
lai = 3
fe = '002'

testjmin = -2.5
testjmax = 3
testnlevels = 2048


wl, upsignal, whitereference, refl, F,noF = prepare_input.synthetic(cab,lai,fe,wlmin=mineval,wlmax=maxeval,completedir='../cwavelets/libradtranscope/floxseries_ae_oen/brdf/')

refnoise, whitereference = prepare_input.add_noise(whitereference,1000,noise)
sensornoise, upsignal = prepare_input.add_noise(upsignal,1000,noise)

nopeak_wl, nopeak_appref = prepare_input.rm_peak(wl,appref) 

p_init = np.polyfit(nopeak_wl,nopeak_appref,polyorder)
               
interp = np.poly1d(p_init)
poly_R_init = interp(wl)

newdecomp = wavelets.decomp(testjmin,testjmax,testnlevels)
newdecomp.adjust_levels(upsignal)
newdecomp.create_comps(upsignal)

weights = wavelets.determine_weights(upsignal,newdecomp.scales)

coeffs,ress = rpoly.optimize_coeffs(wl,whitereference,upsignal,p_init,newdecomp)

coeffs = np.array(coeffs)
polyrefls = []
           
          
for j,polycoef in enumerate(coeffs):
    interp = np.poly1d(polycoef)
    polyrefls.append(interp(wl))
    

        
        
polyrefls = np.array(polyrefls)
polyR = np.ma.average(polyrefls,weights=weights,axis=0)
R_std = funcs.weighted_std(polyrefls,weights=weights,axis=0)

F_der = upsignal-polyR*whitereference
F_err = np.sqrt(np.square(sensornoise) + np.square(np.multiply(whitereference,R_std)) + np.square(np.multiply(polyR,refnoise)))
                