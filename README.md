# SIF retrieval methods (sif_rms)
This repository contains scripts for several SIF (sun induced fluorescence) retrieval methods from different datasets and can be used as a toolbox. 
The focus lies on a newly developed method based on wavelet decompositions of the measured upwelling and reference radiance spectra. 
All scripts are written for Python3 and provided under the GNU General Public License version 3 zur Verfügung gestellt:  

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Dependencies
------------------------

- [Numpy](https://numpy.org/) 
- [Scipy](https://scipy.org/)
- [PyWavelets](https://pywavelets.readthedocs.io/en/latest/index.html)
- [Matplotlib](https://matplotlib.org/) 


Functionality
------------------------

The heart of the method is the optimization between upwelling radiance and reference to retrieve a relative reflectance by comparing absolute absorption line depths filtered for and quantified by means of a wavelet transformation. This optimization also happens in the wavelet space. 



Folders
-------------------
* **r_opts**: contains the optimization scripts for different reflectance parametrizations. Possible are polynomial, piece-wise spline or hyperbolic tangent. The most robust, few parameters but still flexible enough is the 2nd order polynomial for a small enough wavelength window and therefore recommendend. 
* **ih**: contains a script with several functions to read and provide different kinds of input data (also to match a highly resolved solar spectrum to the data as a reference) and a script to prepare the data (i.e. add noise, deconvolve, add spectral response function, remove peaks from apparent reflectance). 
* **utils**: contains tools concerning wavelet decompositions (including the important decomposition class which handles selection of decomposition levels as well as calculation of masks and weights) and plotting.
* **SFM**: Python implementation of the SFM (adapted from the original [matlab version](https://gitlab.com/ltda/flox-specfit))

Examples
---------
* **minimum_example.py**: Minimum working example to retrieve SIF with the wavelet method from a synthetic dataset containing TOC down- and upwelling radiances.
* **fl_scope.py**:
* **fl_flox.py**:
