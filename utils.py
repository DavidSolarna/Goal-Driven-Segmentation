# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:07:29 2020

@author: David
"""

import numpy as np
import math
import bisect
from scipy import special

import sys
import logging
import pathlib
import argparse


#---- I/O

def is_valid_file(arg):
    file = pathlib.Path(arg)
    if not file.is_file:
        raise argparse.ArgumentTypeError("{0} does not exist".format(arg))
    return arg


def get_parser():
    """Custom argument parser."""
    parser = argparse.ArgumentParser(description='Goal-Driven Segmentation.')
    parser.add_argument('--config', '-c', 
                        dest = "config",
                        required=True,
                        metavar="FILE",
                        type = lambda x: is_valid_file(x),
                        help='Path to the configuration file.')
    return parser

def get_valid_model(model_type,model_class):
    if model_type.upper()=='GAUSSIAN':
        model = model_class.GAUSSIAN
    elif model_type.upper()=='GAUSSIAN_MIXTURE':
        model = model_class.GAUSSIAN_MIXTURE
    elif model_type.upper()=='LOGNORMAL':
        model = model_class.LOGNORMAL
    elif model_type.upper()=='WEIBULL':
        model = model_class.WEIBULL
    elif model_type.upper()=='GAMMA':
        model = model_class.GAMMA
    return model

def format_path_string_on_platform(path):
    """Format a string based on Unix or Windows system."""
    if "win" in sys.platform and path[0] == "/":
        logging.getLogger().warning("Removing / from path: {}".format(path))
        return path[1:]

    if "win" not in sys.platform and path[0] != "/":
        if len(pathlib.Path(path).parents) > 1:
            logging.getLogger().warning("Adding / to path: {}".format(path))
            return "/" + path

    return path


#---- GRAPH


def flood_fill_core(original, weights, levelsCurve, seed, level):
   	if level < weights[seed[0], seed[1]]: # it operates only if the weight is larger than the current level
   		return original
   
   	queue = [seed]; # list of points to be assigned a level
   	while queue:
   		current = queue.pop(0)
   		x = current[0]
   		y = current[1]
   		original[x][y]=255
   		weights[x][y]=999
   		levelsCurve[x][y]=level
   
   		for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)): # compute the 4-neighbours
   			if 0<=x2<=weights.shape[0]-1 and 0<=y2<=weights.shape[1]-1:
   				if level >= weights[x2, y2]:
   					queue.append( (x2,y2) ) # append and iterate to this point
   					original[x2][y2]=255
   					weights[x2][y2]=999
   					levelsCurve[x][y]=level
   	return original


#---- IMAGE   


def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
 
    
def normalize_datacube_0_1(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array = np.copy(array.astype(np.float64))
    n_dim = np.ndim(array)
    if n_dim == 3:
        bands = np.shape(array)[2]
        for b in range(0, bands):
            band = array[:,:,b]
            band_min, band_max = band.min(), band.max()
            array[:,:,b] = ((band - band_min)/(band_max - band_min))
        return array
    elif n_dim == 2:
        array_min, array_max = array.min(), array.max()
        return ((array - array_min)/(array_max - array_min))
    else:
        raise ValueError("Expected array dimension to be 2 or 3. Received {}.".format(n_dim))


def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2 ,'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, 255):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst


#---- PARAMETERS APPROXIMATION


def f(L, k2):
    """k2 - polygamma for L estimation"""
    return (k2-special.polygamma(1, L))


#---- MODELS  
      
        
def gaussian_model_no_log(array, params):
    mean, var = params
    dev_std = math.sqrt(var)
    A = 1 / (dev_std * math.sqrt(2*math.pi))
    E = -0.5 * np.square((array - mean) / dev_std)
    return A * np.exp(E)


def gaussian_model(array, params):
    mean, var = params
    A = -0.5*math.log(2*math.pi)
    B = -0.5*math.log(var)
    C = -(1/(2*var))*np.square(array-mean)
    return A + B + C


def ln_gaussian_model(array, params):
    mean, var = params
    A = -0.5*math.log(2*math.pi)
    B = -0.5*math.log(var)
    C = -(1/(2*var))*np.square(array-mean)
    return A + B + C


def lognormal_model(array, params):
    k1, k2 = params
    dev_std = math.sqrt(k2)
    A = 1 / (dev_std * array * math.sqrt(2*math.pi))
    E = -0.5 * (np.square(np.log(array) - k1) / k2)
    return np.log(A * np.exp(E))


def weibull_model(array, params):
    array = array + 0.1
    eta, mu = params
    A = eta * np.power(array,eta-1) / np.power(mu,eta)
    E = np.power(array/mu, eta)
    return np.log(A * np.exp(- E))


def gamma_model(array, params):
    array = array + 0.1
    L, mu = params
    A = 1/special.gamma(L)
    B = np.power(L/mu, L)
    C = np.power(array, L-1)
    E = -L * array / mu
    return np.log(A * B * C * np.exp(E))