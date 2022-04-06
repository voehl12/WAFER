import numpy as np
from scipy import interpolate
from r_opts import rspline,rtanh
from ih import prepare_input
import matplotlib.pyplot as plt

noise = 1
wl, signal, whitereference, refl, F = prepare_input.synthetic(60,4,'002')
refnoise, whitereference = prepare_input.add_noise(whitereference,1000,noise)
sensornoise, signal = prepare_input.add_noise(signal,1000,noise)
appref = np.divide(signal,whitereference)

""" minpeak = np.argmin(np.fabs(wl-757))
maxpeak = np.argmin(np.fabs(wl-768))
nopeak_appref = []
nopeak_wl = []
for i in range(len(wl)):
    if i < minpeak or i > maxpeak:
        nopeak_appref.append(appref[i])
        nopeak_wl.append(wl[i])
interRa_smooth = interpolate.UnivariateSpline(nopeak_wl,nopeak_appref,s=0,k=3)
init_R = interRa_smooth(wl) """

jmin = 2
jmax = 4
nlevels = 16
splineorder = 1
scales = np.logspace(jmin,jmax,num=nlevels,base=2)

init_R = rtanh.optimize_tanh(wl,whitereference,signal,scales,w0=wl[0],w1=wl[-1])

plt.figure()
plt.plot(wl,init_R)
plt.plot(wl,refl)


interR, smoothing = rspline.adjust_smoothing(wl,init_R,0.001,20,21,splineorder)
plt.plot(wl,interR(wl))
plt.plot(wl,appref)
plt.show()

splinecoeffs = interR._data[9]
knots = interR._data[8]

coeffs = rspline.optimize_coeffs(wl,whitereference,signal,knots,splinecoeffs,scales,splineorder,lbl=1)
coeffs = np.array(coeffs)
if coeffs.ndim > 1:

    meancoeffs = np.mean(coeffs,axis=0)
else:
    meancoeffs = coeffs

B = rspline.setup_B(wl,knots,splineorder)
plt.figure()
if coeffs.ndim > 1:

    for i in range(len(coeffs)):

        ncoeffs = np.zeros(len(B))
        ncoeffs[:len(meancoeffs)] = np.array(coeffs[i])
        final_R = rspline.bspleval(ncoeffs,B)
        plt.plot(wl[:-1],final_R[:-1],label=i)

else:
    ncoeffs = np.zeros(len(B))
    ncoeffs[:len(meancoeffs)] = np.array(meancoeffs)
    final_R = rspline.bspleval(ncoeffs,B)
    plt.plot(wl[:-1],final_R[:-1],label='derived R')




plt.plot(wl,refl)

plt.plot(wl,appref)
plt.legend()
plt.figure()
for i in range(len(meancoeffs)):
    plt.plot(wl, meancoeffs[i]*B[i,splineorder,:])
plt.title('B-spline basis functions')

plt.show()

