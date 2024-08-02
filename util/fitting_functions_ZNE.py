import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def exp_ZNE(x,y, weights=None, debug=False):
    if debug:
        plt.plot(x, y)
    exp_func = lambda input, A, tau, c: A*np.exp(-input/tau)+c
    popt, pcov = curve_fit(exp_func, x, y, p0 = [y[0], 1, 0], sigma = weights, absolute_sigma = False)
    #if debug:
        #print(popt)
    exp_fitted = lambda input: popt[0]*np.exp(-np.array(input)*1/popt[1]) + popt[2]
    if debug:
        plt.plot(x, exp_fitted(x))
        plt.show()
    return exp_fitted    

def cubic_ZNE(x, y, weights=None):

    cubic_func = lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d
    popt, pcov = curve_fit(cubic_func, x, y, sigma = weights, absolute_sigma = False)
    cubic_fitted = lambda x: popt[0]*x**3 + popt[1]*x**2 + popt[2]*x + popt[3]
    return cubic_fitted

def quadratic_only_ZNE(x, y, weights=None):
    cubic_func = lambda x,  b,  d:+ b*x**2  + d
    popt, pcov = curve_fit(cubic_func, x, y, sigma = weights, absolute_sigma = False)
    cubic_fitted = lambda x:  popt[0]*x**2 + popt[1]
    return cubic_fitted

def poly_ZNE(x, y, weights=None, order=1):
    poly_func = lambda x, *coeffs: sum([coeffs[i]*x**i for i in range(order+1)])
    popt, pcov = curve_fit(poly_func, x, y, sigma = weights, absolute_sigma = False, p0=[1]*(order+1))
    poly_fitted = lambda x: sum([popt[i]*x**i for i in range(order+1)])
    return poly_fitted

def third_no_first_ZNE(x, y, weights=None):
    cubic_func = lambda x, a, b, d: a*x**3 + b*x**2  + d
    popt, pcov = curve_fit(cubic_func, x, y, sigma = weights, absolute_sigma = False)
    cubic_fitted = lambda x: popt[0]*x**3 + popt[1]*x**2  + popt[2]
    return cubic_fitted

def fancy_exp_ZNE(x, y, weights=None):

    fancy_exp_func = lambda x, a,b,c: c*np.exp(-(a*x**2+b*x)) 
    try:
        popt, pcov = curve_fit(fancy_exp_func, x, y, p0 = [1, 0, y[0]], sigma = weights, absolute_sigma = False)
    except:
        popt = [0,0,0]
    fancy_exp_fitted = lambda x: popt[2]*np.exp(-(popt[0]*x**2+popt[1]*x)) 
    return fancy_exp_fitted

#TODO: log plot linear fit
#Weighted fits
#Higher order polynomial
#Squared theta

def gaussian_ZNE(x, y, weights=None):

    # gaussian_func = lambda x, A, sigma: A*np.exp(-(x)**2/(2*sigma**2))
    # try:
    #     popt, pcov = curve_fit(gaussian_func, x, y, maxfev=10000, p0 = [y[0], 0.1])
    # except:
    #     popt = [0,0]
    # gaussian_fitted = lambda x: popt[0]*np.exp(-(x)**2/(2*popt[1]**2))
    # return gaussian_fitted
    gaussian_func = lambda x, A, sigma, mu,c: A*np.exp((-x**2 + mu*x)/(2*sigma**2)) + c
    try:
        popt, pcov = curve_fit(gaussian_func, x, y, maxfev=10000, p0 = [y[0], 0.1, 0,0])
    except:
        popt = [0,1,0]
    gaussian_fitted = lambda x: popt[0]*np.exp((-x**2 + popt[2]*x)/(2*popt[1]**2)) + popt[3]
    return gaussian_fitted
    

def log_fit_exp_ZNE(x, y, weights=None, order=1, debug=False):
    poly_func = lambda x, *coeffs: sum([coeffs[i]*x**i for i in range(order+1)])
    if debug:
        plt.plot(x, np.log(np.abs(y)))
        plt.show()
    
    popt, pcov = curve_fit(poly_func, x, np.log(np.abs(y)), sigma = weights, absolute_sigma = False, p0=[1]*(order+1))
    sign = np.sign(y[0])
    log_fit = lambda x: sign*np.exp(sum([popt[i]*x**i for i in range(order+1)]))
    return log_fit

def linear_ZNE(x, y, weights=None):
    linear_func = lambda x, a, b: a*x + b
    popt, pcov = curve_fit(linear_func, x, y, sigma = weights, absolute_sigma = False, p0=[1, 1])
    linear_fitted = lambda x: popt[0]*x + popt[1]
    return linear_fitted

def Cetina_fit(x, y, weights=None, debug=False):
    Cetina_func = lambda  theta, A, Omega: A*np.sqrt(1 + (Omega*theta)**2)
    popt, pcov = curve_fit(Cetina_func, x, y, p0 = [y[0], 1], sigma = weights, absolute_sigma = False)
    Cetina_fitted = lambda theta: popt[0]*np.sqrt(1 + (popt[1]*theta)**2)
    if debug:
        plt.plot(x, y)
        plt.plot(x, Cetina_fitted(x))
        plt.show()
    return Cetina_fitted

ALL_FUNCTIONS = [exp_ZNE, cubic_ZNE, quadratic_only_ZNE, poly_ZNE, third_no_first_ZNE, fancy_exp_ZNE, gaussian_ZNE, log_fit_exp_ZNE, linear_ZNE]