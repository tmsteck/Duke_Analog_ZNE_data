import numpy as np
from scipy.optimize import curve_fit, minimize
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

def order_poly_ZNE(x, y, weights=None, order=3, remove_first=True, debug=False, return_cov=False, even_only=False):
    #print(order)
    if remove_first:
        def poly_function(x, *coeffs):
            #Convert coeffs to (a, 0, b, c, ..). Ie, insert 0 for x^1
            coeffs = np.insert(coeffs, 1, 0)
            polynomial = Polynomial(coeffs)
            return polynomial(x)
        #Generate a p0:
        #First derivative: 
        #d1 = np.gradient(y, x)
        #Second derivative:
        #d2 = np.gradient(d1, x)
        #print(d2)
        
        p0 = [y[0]] + [0 for _ in range(order-1)]
        #print(p0)
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


def converge_ZNE_loocv(x, y, remove_first=True, debug=False, weights=None, return_cov=False,  even_only=False, return_order=False):
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
    for order in range(1, len(y)):
        try:
            function = lambda x, y: order_poly_ZNE(x, y, order=order, remove_first=remove_first, weights=weights, debug=False, even_only=True)  
            rss = loocv(x, y, function, evalute)
            residuals.append(rss)
        except Exception as e:
            #print(e)
            #print('Fitting failed for order = ', order)
            residuals.append(np.nan)
    if debug:
        plt.scatter(range(len(residuals)), residuals)
        plt.show()
    #get the index of the minimum residual
    try:
        min_residual = np.nanargmin(residuals)
    except:
        print('Loocv failed')
        #print(residuals)
        min_residual = 3
    if return_cov:
        function, cov = order_poly_ZNE(x, y, order=min_residual, remove_first=remove_first, weights=weights, debug=debug, return_cov=return_cov)
        J = np.array([0**i for i in range(min_residual)])
        sigma_0 = np.sqrt(J @ cov @ J.T)
        if return_order:
            return function, sigma_0, min_residual
        else:
            return function, sigma_0
    else:
        function = order_poly_ZNE(x, y, order=min_residual, remove_first=remove_first, weights=weights, debug=debug)
        if return_order:
            return function, min_residual
        else:
            return function

def pade_fit_ZNE(x, y, tail=0, debug=False):
    """Computes the pade fit for the data using P_l/Q_m for l = m for tail > 0, or m > l for tail = 0"""
    maximum_total_order = len(y) - 1
    #if tail == 0:
    residuals = np.ones((maximum_total_order, maximum_total_order))*200
    
    y = y - tail
    if debug:
        plt.plot(x, y)
    for m in range(1, maximum_total_order):
        for l in range(1, m):
            
            def pade_function(x, *params):
                P = Polynomial(params[:l+1])
                Q = Polynomial(params[l+1:])
                if Q(0) == 0:
                    return np.nan
                return P(x)/Q(x)
            try:
                params, params_cov = curve_fit(pade_function, x, y, p0 = [1 for _ in range(m+l+2)], maxfev=100000)
                residuals[l, m] = np.sum((y - pade_function(x, *params))**2)
            except:
                residuals[l, m] = np.nan
            if debug:
                plt.plot(x, pade_function(x, *params))
    #Get the index of the smallest residual
    min_residual = np.unravel_index(np.nanargmin(residuals), residuals.shape)
    if debug:
        print(residuals)
    l, m = min_residual
    print(l, m)
    def pade_function(x, *params):
        P = Polynomial(params[:l+1])
        Q = Polynomial(params[l+1:])
        if Q(0) == 0:
            return np.inf
        return P(x)/Q(x)
    params, params_cov = curve_fit(pade_function, x, y, p0 = [1 for _ in range(m+l+2)])
    return pade_function(0 ,*params) + tail
        

        
#for l in range(1, maximum_total_order//2):
        

    
    
    
# max_order = min(len(y)-1, 20)
# values = [y[0]]
# step_size = [1]
# functions = [lambda x: y[0]]
# for i in range(1, max_order):
#     try:
#         fit = order_poly_ZNE(x, y, order=i, remove_first=remove_first, weights=weights, debug=False)
#         values.append(fit(0))
#         functions.append(fit)
#         step_size.append(np.abs(fit(0) - values[i - 1]))
#     except:
#         break



# if debug:
#     plt.scatter(range(len(values)), values, label="Values")
#     plt.scatter(range(len(step_size)), step_size, label="Step size")
# # Check for two consecutive points where step sizes increase


# for i in range(2, len(step_size)):
#     if step_size[i] > step_size[i - 1]:#and step_size[i] < step_size[i - 2]:
#         try:
#             if step_size[i+1] < step_size[i-1]:
#                 continue
#         except:
#             pass
#         i = i-1
#         if debug:
#             plt.vlines(i, 0, max(values), color='red', label='Divergence Start')
#             plt.legend()
#             plt.show()
#             order_poly_ZNE(x, y, order=i, remove_first=remove_first, weights=weights, debug=debug)
#         # Return the function and index before the divergence starts
#         return (functions[i ], i)
# if debug:
#     plt.vlines(len(step_size) - 2, 0, max(values), color='red', label='Convergence Failed')
#     plt.legend()
#     plt.show()
# return (functions[-2], len(functions) - 2)


# def ZNE_Pade_V2(x, y, tail = 0, debug=False, remove_first_order = False, dense_data = None):
#     """This function generates an optimal Pade-Approximant for the data x and y. 
    
#     The key steps in generating the optimal fit are:
#     1. Compute the optimal Polynomial fits for the data
#     2. map these back to a Rational function to initialize the optimizers
#     3. Include cost terms for real axis (or, near-real axis) singularities (that are not removed by the numerator)
#     4. Include cost terms for non-zero derivatives at x = 0
    
#     """
    
#     y = y - tail
#     max_total_order = len(y) - 1
#     lm_residuals = np.zeros((max_total_order, max_total_order))
#     lm_params = np.zeros((max_total_order, max_total_order), dtype=object)
    
#     if debug:
#         plt.scatter(x, y, s=110, zorder=-1, color='k')
#         if dense_data is not None:
#             plt.plot(dense_data[0], dense_data[1]-tail)
#     for m in range(1, max_total_order):
#         for l in range(1, max_total_order):
#             if l + m > max_total_order:
#                 continue
            
#             #Generate the polynomial fit, convert to Pade approximant, then convert that to an initial value for the minimize funciotn
#             def cost_function(x, *params):
#                 P = Polynomial(params[:l+1])
#                 Q = Polynomial(params[l+1:])
#                 zeros = Q.roots()
#                 #Check if there are any roots between 0 and max(x)
#                 imaginary_bound = 1e-1
#                 failed = False
#                 penalty = 100
#                 outputs = P(x)/Q(x)
#                 residuals = np.sum((y - outputs)**2)
#                 for root in zeros:
#                     if np.abs(root.image) < imaginary_bound and root.real > 0 and root.real < max(x):
#                         if np.abs(P(root)) < 1e-5:
#                             failed = False
#                         else:
#                             failed = True
#                         break
#                 if failed:
#                     return residuals*penalty
#                 else:
#                     #Compute the first derivative:
#                     dP = P.deriv()
#                     dQ = Q.deriv()
#                     firstD = dP(0)/dQ(0)
#                     return residuals + penalty*np.abs(firstD)
#                 p0= 
#             optimal_params = minimize(cost_function, x, y, method='Nelder-Mead')
#             lm_params[l, m] = optimal_params
            
    
    
#     for m in range(1, maximum_total_order):
#         for l in range(1, m):
#             if l + m > maximum_total_order:
#                 continue
#             def cost_function(x)
            
        
#             def pade_function(x, *params):
#                 P = Polynomial(params[:l+1])
#                 Q = Polynomial(params[l+1:])
#                 if Q(0) == 0:
#                     return np.nan
#                 return P(x)/Q(x)
#             try:
#                 #Fit to an order l polynomial first:
#                 poly_fit = Polynomial.fit(x, y, l)
#                 p0 = np.concatenate([poly_fit.coef, np.ones(m+1)])
#                 params, params_cov = curve_fit(pade_function, x, y, p0 = p0, maxfev=1000000)
#                 residuals[l, m] = np.sum((y - pade_function(x, *params))**2)
#                 params_array[l,m] = params
#             except:
#                 counter = 0
#                 while counter < 100:
#                     try:
#                         counter += 1
#                         #print(counter)
#                         p0 = np.random.rand(m+l+2)*10
#                         params, params_cov = curve_fit(pade_function, x, y, p0 = p0, maxfev=1000000)
#                         residuals[l, m] = np.sum((y - pade_function(x, *params))**2)
#                         params_array[l,m] = params
#                         counter = 1000
#                     except:
#                         pass
#                 if counter < 900:
#                     print('Failed for l,m = ' + str(l) + ', ' + str(m))
#                     residuals[l, m] = np.inf
#             if debug:
#                 plt.scatter(x, pade_function(x, *params))#, label=str(l) + ', ' + str(m))
#                 #try:
#                 #    #plt.scatter(dense_data[0], pade_function(dense_data[0], *params))
#                 #except:
#                 #    pass
#     #Get the index of the smallest residual
#     failed = True
#     if debug:
#         function = converge_ZNE_loocv(x, y, remove_first=True)
#         plt.plot(dense_data[0], function(dense_data[0]), label='ZNE Loocv')
#     while failed:
#         min_residual = np.unravel_index(np.nanargmin(residuals), residuals.shape)
#         if debug:
#             print(residuals)
#         l, m = min_residual
#         print(l,m)
#         #Check roots:
#         #Find the zeros of the denominator. If there is a real zero betwen 0 and max(x), then discard the fit and try again:
#         def find_denom_roots(l, current_params):
#             Q = Polynomial(current_params[l+1:])
#             return Q.roots()
#         P = Polynomial(params_array[l,m][:l+1])
#         #Check if there are any roots between 0 and max(x)
#         print('Parameters to check:')
#         print(params_array[l,m])
#         roots = find_denom_roots(l, params_array[l,m])
#         print(roots)
#         print([P(root) for root in roots])  
#         #print(roots)
#         #Iterate through each root:
#         failed = False
#         if not failed:
#             for root in roots:
#                 if np.abs(root.imag) <= 0.1 and root.real > 0 and root.real < max(x):
#                     if np.abs(P(root)) < 1e-5:
#                         failed = False
#                     else:
#                         failed = True
#                     break
#         if failed:
#             residuals[min_residual] = np.inf
    
    
    
    
#     def pade_function(x, *params, debug=False):
#         P = Polynomial(params[:l+1])
#         Q = Polynomial(params[l+1:])
#         if debug:
#             print(P)
#             print(Q)
#         if Q(0) == 0:
#             return np.inf
#         return P(x)/Q(x)
#     params, params_cov = curve_fit(pade_function, x, y, p0 = [1 for _ in range(m+l+2)])
#     if debug:
#         plt.scatter(x, pade_function(x, *params), label=str(l) + ', ' + str(m))
#         #if dense_data is not None:
#         plt.plot(dense_data[0], pade_function(dense_data[0], *params), label=str(l) + ', ' + str(m))
#         #Also plot the ZNE Loocv fit:
        
#         plt.legend()
        
#     return pade_function(0 ,*params, debug=True) + tail