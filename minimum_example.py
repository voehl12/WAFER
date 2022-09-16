import numpy as np
from scipy import interpolate, signal
from r_opts import rspline,rtanh,rpoly
from ih import prepare_input
import matplotlib.pyplot as plt
import scipy
from utils import wavelets,plotting,funcs
from SFM import SFM,SFM_BSpline
from scipy import optimize
from datetime import datetime



################### retrieval parameters ###############################

directory = '../cwavelets/libradtranscope/floxseries_ae_oen/brdf/'

mineval,maxeval,eval_wl = 754,773,760


cab = 20
lai = 3
fe = '002'

jmin = -2.5
jmax = 3
nlevels = 2048
noise = 0
polyorder = 2

########################################################################

############### data preparation #######################################

wl, upsignal, whitereference, refl, F,noF = prepare_input.synthetic(cab,lai,fe,wlmin=mineval,wlmax=maxeval,completedir=directory)

refnoise, whitereference = prepare_input.add_noise(whitereference,1000,noise)
sensornoise, upsignal = prepare_input.add_noise(upsignal,1000,noise)

########################################################################

############### initial guess and reflectance optimization #############

start = datetime.now()
appref = np.divide(upsignal,whitereference)
nopeak_wl, nopeak_appref = prepare_input.rm_peak(wl,appref) 

p_init = np.polyfit(nopeak_wl,nopeak_appref,polyorder)
               
interp = np.poly1d(p_init)
poly_R_init = interp(wl)

newdecomp = wavelets.decomp(jmin,jmax,nlevels)
newdecomp.adjust_levels(upsignal)
newdecomp.create_comps(upsignal)

coeffs,ress = rpoly.optimize_coeffs(wl,whitereference,upsignal,p_init,newdecomp)

########################################################################

############# weighting and final reflectance ##########################
newdecomp.calc_weights(upsignal)
weights = newdecomp.weights
coeffs = np.array(coeffs)
polyrefls = []
           
          
for j,polycoef in enumerate(coeffs):
    interp = np.poly1d(polycoef)
    polyrefls.append(interp(wl))
    
polyrefls = np.array(polyrefls)
polyR, R_std = funcs.weighted_std(polyrefls,weights=weights,axis=0)


########################################################################


F_der = upsignal-polyR*whitereference

end = datetime.now()
print(end-start)

F_err = np.sqrt(np.square(sensornoise) + np.square(np.multiply(whitereference,R_std)) + np.square(np.multiply(polyR,refnoise)))
                
plt.figure()
plt.plot(wl,F_der)
plt.show()