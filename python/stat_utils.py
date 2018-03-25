#! /usr/bin/env python
# coding: utf-8

"""
@file     stat_utils.py
@author   Vincenzo Battista
@date     21/11/2017

@brief    Collection of utilities for statistics
"""

import logging
log = logging.getLogger(__name__)

import numpy as np
import math

def ks_2samp_w(data1, data2, weights1, weights2):
    """
    Reimplementation of ks_2samp from scipy/stats that allows weighted samples.
    From:
    https://stackoverflow.com/questions/40044375/how-to-calculate-the-kolmogorov-smirnov-statistic-between-two-weighted-samples
    
    NOT QUITE SURE IT WORKS. HELPERS ARE WELCOME
    """
    
    from scipy.stats import kstwobign
    
    n1 = np.sum(weights1)
    n2 = np.sum(weights2)
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    weights1 = weights1[ix1]
    weights2 = weights2[ix2]
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1,data2])
    cwei1 = np.hstack([0, np.cumsum(weights1)/sum(weights1)])
    cwei2 = np.hstack([0, np.cumsum(weights2)/sum(weights2)])
    cdf1we = cwei1[[np.searchsorted(data1, data_all, side='right')]]
    cdf2we = cwei2[[np.searchsorted(data2, data_all, side='right')]]
    d = np.max(np.absolute(cdf1we-cdf2we))
    # Note: d absolute not signed distance
    en = np.sqrt(n1*n2/float(n1+n2))
    try:
        prob = kstwobign.sf((en + 0.12 + 0.11 / en) * d) #where this come from?
    except:
        prob = 1.0
    
    return d, prob
    
def weighted_percentile(data, percents, weights=None):
    """
    Returns a weighted percentiles, i.e. fractions of total
    weight are considered instead of fractions of total sample size.
    From:
    https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    
    if weights is None:
        return np.percentile(data, percents)
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]
    p=1.*w.cumsum()/w.sum()*100
    y=np.interp(percents, p, d)
    return y

def weighted_average(data):
    """
    Returns weighted average and its uncertainty.
    Weights are the inverse of the errors squared.
    It corresponds to mean and uncertainty obtained
    by a least-squares fit where the fitting
    function is a constant.
    It requires a list of ufloat objects as input
    """
    
    if len(data)==1:
        return data[0].n, data[0].s
    
    x_mu = 0.0
    norm = 0.0
    for x in data:
        x_mu += x.n * (1/x.s)**2
        norm += (1/x.s)**2
        
    x_mu /= norm
    x_std = math.sqrt( 1/norm )
    
    return x_mu, x_std

def weighted_mean(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def weighted_covariance(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))) / np.sum(w)

def weighted_correlation(x, y, w):
    """Weighted Correlation"""
    return weighted_covariance(x, y, w) / np.sqrt(weighted_covariance(x, x, w) * weighted_covariance(y, y, w))