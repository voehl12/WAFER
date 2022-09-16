# SIF retrieval methods (sif_rms)
This repository contains scripts for several SIF (sun induced fluorescence) retrieval methods from different datasets and can be used as a toolbox. 
The focus lies on a newly developed method based on wavelet decompositions of the measured upwelling and reference radiance spectra. 
All scripts are written for Python3 and provided under the GNU General Public License version 3.  

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Dependencies
------------------------

- [Numpy](https://numpy.org/) 
- [Scipy](https://scipy.org/)
- [PyWavelets](https://pywavelets.readthedocs.io/en/latest/index.html)
- [Matplotlib](https://matplotlib.org/) 


Functionality
------------------------


The heart of the newly developed wavelet method is the optimization between upwelling radiance and reference to retrieve a relative reflectance by comparing absolute absorption line depths filtered for and quantified by means of a wavelet transformation. This optimization also happens in the wavelet space. 
The reflectance is imprinted in the absolute depth of the Fraunhofer lines and therefore the wavelet coefficients. The Fluorescence is not as it does not contribute to the small scale wavelet decomposition and merely creates an additional offset and no changes to the absolute line depth of absorption lines. 


Folders
-------------------
* **r_opts**: contains the optimization scripts for different reflectance parametrizations. Possible are polynomial, piece-wise spline or hyperbolic tangent. The most robust, few parameters but still flexible enough is the 2nd order polynomial for a small enough wavelength window and therefore recommendend. 
* **ih**: contains a script with several functions to read and provide different kinds of input data (also to match a highly resolved solar spectrum to the data as a reference) and a script to prepare the data (i.e. add noise, deconvolve, add spectral response function, remove peaks from apparent reflectance). 
* **utils**: contains tools concerning wavelet decompositions (including the important decomposition class which handles selection of decomposition levels as well as calculation of masks and weights) and plotting. The file funcs.py contains useful functions like calculating the weighted standard deviation, other statistical evaluation methods. The file results.py contains a class that handles all results (writes to file and evaluates sif parameters). 
* **SFM**: Python implementation of the SFM (adapted from the original [matlab version](https://gitlab.com/ltda/flox-specfit)). Reference: Cogliati, S., Celesti, M., Cesana, I., Miglietta, F., Genesio, L., Julitta, T., Schuettemeyer, D., Drusch, M., Rascher, U., Jurado, P., Colombo, R. (2019). A Spectral Fitting Algorithm to Retrieve the Fluorescence Spectrum from Canopy Radiance. Remote Sensing, 11(16), 1840. [https://doi.org/10.3390/rs11161840](https://doi.org/10.3390/rs11161840)

Examples
---------
* **minimum_example.py**: Minimum working example to retrieve SIF with the wavelet method from a synthetic dataset containing TOC down- and upwelling radiances.
* **fl_scope.py**:
* **fl_flox.py**:
* **fl_hyplant.py**: 
