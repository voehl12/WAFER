import numpy as np 


def weighted_std(values, weights,axis):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights,axis=axis)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights,axis=axis)
    return average, np.sqrt(variance)


def calc_rsquared(y,y_m):
    """
    arguments:
    y - measured data points
    y_m - model to compare with

    returns: 
    R squared
    """

    sstot = np.sum(np.square(y-np.mean(y)))
    ssres = np.sum(np.square(y-y_m))
    return 1 - ssres / sstot

def calc_rmse(y,y_m):
    return np.sqrt(np.mean(np.square(y-y_m)))

def calc_rrmse(y,y_m):
    return np.sqrt(np.mean(np.square((y-y_m)/y_m)))

def fitfct_linear(x,a,b):
    return a*x + b