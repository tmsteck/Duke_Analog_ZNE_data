import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.interpolate import pade


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

def order_poly_ZNE(x, y, weights=None, order=3, remove_first=True, debug=False, return_cov=False):
    #print(order)
    if remove_first:
        def poly_function(x, *coeffs):
            #Convert coeffs to (a, 0, b, c, ..). Ie, insert 0 for x^1
            coeffs = np.insert(coeffs, 1, 0)
            polynomial = Polynomial(coeffs)
            return polynomial(x)
        #Generate a p0:
        p0 = [y[0]] + [0 for _ in range(order-1)]
        popt, pcov = curve_fit(poly_function, x, y, sigma = weights, absolute_sigma = False,p0=p0, maxfev=10000)
        def return_poly(*coeffs):
            coeffs = np.insert(coeffs, 1, 0)
            polynomial = Polynomial(coeffs)
            return polynomial
        if debug:
            plt.scatter(x, y)
            print(return_poly(*popt))
            plt.plot(x, return_poly(*popt)(x))
            plt.xlim(0, max(x))
            plt.show()
        if return_cov:
            return return_poly(*popt), pcov
        else:
            return return_poly(*popt)
    else:
        poly_func = lambda x, *coeffs: sum([coeffs[i]*x**i for i in range(order+1)])
        p0 = [y[0]] + [0 for _ in range(order)]
        popt, pcov = curve_fit(poly_func, x, y, sigma = weights, absolute_sigma = False, p0=p0)
        poly_fitted = lambda x: sum([popt[i]*x**i for i in range(order+1)])
        if return_cov:
            return poly_fitted, pcov
        else:
            return poly_fitted

def order_poly_instance(order, remove_first=True, debug=False):
    return lambda x, y, weights=None: order_poly_ZNE(x, y, weights, order, remove_first, debug=debug)

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
        popt = [0,1,0,0]
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


def detect_divergence_point(step_size, debug=False):
    """
    Detects the index where the divergence in step size happens.
    
    Args:
        step_size (list or np.array): A list or array of step sizes.
        debug (bool): If True, plots the step sizes with divergence point marked.
    
    Returns:
        int: The index before the divergence starts.
    """
    # Iterate from the second element to check divergence
    for i in range(2, len(step_size)):
        # Check for divergence condition: two consecutive increases
        if step_size[i] > step_size[i - 1] and step_size[i - 1] > step_size[i - 2]:
            if debug:
                # Plot the step sizes with marked divergence point
                plt.scatter(range(len(step_size)), step_size, label="Step size", color="orange")
                plt.axvline(x=i - 1, color='red', linestyle='--', label="Divergence Start")
                plt.legend()
                plt.show()
            # Return the index before divergence starts
            return i - 2

    # If no divergence is detected, return the last stable index
    return len(step_size) - 1

def converge_ZNE_order(x, y, remove_first=True, debug=False, weights=None):
    #Iterates what the ZNE expectation is for a given order of polynomial. Returns the order fit where the data diverges
    max_order = min(len(y)-1, 20)
    values = [y[0]]
    step_size = [0]
    functions = [lambda x: y[0]]
    for i in range(1, max_order):
        try:
            fit = order_poly_ZNE(x, y, order=i, remove_first=remove_first, weights=weights, debug=False)
            values.append(fit(0))
            functions.append(fit)
            step_size.append(np.abs(fit(0) - values[i - 1]))
        except:
            break
    

    if debug:
        plt.scatter(range(len(values)), values, label="Values")
        plt.scatter(range(len(step_size)), step_size, label="Step size")
    # Check for two consecutive points where step sizes increase


    for i in range(2, len(step_size)):
        if step_size[i] > step_size[i - 1]:#and step_size[i] < step_size[i - 2]:
            try:
                if step_size[i+1] < step_size[i-1]:
                    continue
            except:
                pass
            i = i-1
            if debug:
                plt.vlines(i, 0, max(values), color='red', label='Divergence Start')
                plt.legend()
                plt.show()
                order_poly_ZNE(x, y, order=i, remove_first=remove_first, weights=weights, debug=debug)
            # Return the function and index before the divergence starts
            return (functions[i ], i)
    if debug:
        plt.vlines(len(step_size) - 2, 0, max(values), color='red', label='Convergence Failed')
        plt.legend()

        plt.show()
        
    return (functions[-2], len(functions) - 2)

    # # If no divergence is found, return the last computed function
    # return (functions[-2], len(functions) - 2)
        
        #if np.abs(fit(x[-1])-y[-1]) > 1:
        #    return i


def converge_ZNE_loocv(x, y, remove_first=True, debug=False, weights=None, return_cov=False, use_pade=False, tail=0):
    """From: https://medium.com/@snp.kriss/a-simple-method-for-determining-an-order-of-polynomial-fits-to-avoid-overfitting-f4e9dfa07e1e"""
    #Iterates what the ZNE expectation is for a given order of polynomial. Returns the order fit where the data diverges
    """Computes the fit leaving out a single data point, and then checks the rss of the fit. If the rss is greater than the previous fit, then the order is returned"""
    def loocv(x, y, fit, pred):
        """Leave one out cross validation(LOOCV) 
        RSS for fitting a polynomial model """
        n = len(x)
        idx = np.arange(n)
        rss = np.sum([(y - pred(fit(x[idx!=i], y[idx!=i]), x))**2.0 for i in range(n)])
        penalty = 0#order * min(np.abs(y))
        return rss + penalty
    
    def evalute(function,x):
        return function(x)
    
    
    residuals = [np.nan]
    for order in range(1, min(len(y), 15)):
        try:
            function = lambda x, y: order_poly_ZNE(x, y, order=order, remove_first=remove_first, weights=weights, debug=False)  
            rss = loocv(x, y, function, evalute)
            residuals.append(rss)
        except:
            residuals.append(np.nan)
    if debug:
        plt.scatter(range(len(residuals)), residuals)
        plt.show()
    #get the index of the minimum residual
    try:
        min_residual = np.nanargmin(residuals)
    except:
        min_residual = 3
    if use_pade:
        
        y_shift = y - tail
        function = order_poly_ZNE(x, y_shift, order=min_residual, remove_first=remove_first, weights=weights, debug=False)
        return function
        outputs = []
        for m in range(1, min_residual+1):
            for l in range(1, m):
                print(l,m)
                p,q = pade(function, l, m)
                print(p)
                print(p(0))
                outputs.append((m,l, p(0)/q(0)))
        return outputs
        
                    
                    
        
    
    if return_cov:
        function, cov = order_poly_ZNE(x, y, order=min_residual, remove_first=remove_first, weights=weights, debug=debug, return_cov=return_cov)
        J = np.array([0**i for i in range(min_residual)])
        sigma_0 = np.sqrt(J @ cov @ J.T)
        return function, sigma_0
    else:
        function = order_poly_ZNE(x, y, order=min_residual, remove_first=remove_first, weights=weights, debug=debug)
        return function
        
    
    
    
    max_order = min(len(y)-1, 20)
    values = [y[0]]
    step_size = [1]
    functions = [lambda x: y[0]]
    for i in range(1, max_order):
        try:
            fit = order_poly_ZNE(x, y, order=i, remove_first=remove_first, weights=weights, debug=False)
            values.append(fit(0))
            functions.append(fit)
            step_size.append(np.abs(fit(0) - values[i - 1]))
        except:
            break

    
    
    if debug:
        plt.scatter(range(len(values)), values, label="Values")
        plt.scatter(range(len(step_size)), step_size, label="Step size")
    # Check for two consecutive points where step sizes increase


    for i in range(2, len(step_size)):
        if step_size[i] > step_size[i - 1]:#and step_size[i] < step_size[i - 2]:
            try:
                if step_size[i+1] < step_size[i-1]:
                    continue
            except:
                pass
            i = i-1
            if debug:
                plt.vlines(i, 0, max(values), color='red', label='Divergence Start')
                plt.legend()
                plt.show()
                order_poly_ZNE(x, y, order=i, remove_first=remove_first, weights=weights, debug=debug)
            # Return the function and index before the divergence starts
            return (functions[i ], i)
    if debug:
        plt.vlines(len(step_size) - 2, 0, max(values), color='red', label='Convergence Failed')
        plt.legend()

        plt.show()
        
    return (functions[-2], len(functions) - 2)