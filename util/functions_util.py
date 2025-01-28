import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, prange
from numba import jit, float64, int32, boolean
from util.samplers_util import thermal_rejection
from scipy.optimize import curve_fit, minimize

@njit()
def gaussian_envelope_shifted(t, theta, Omega):
    """"A frequency shifted gaussian envelope function

    Args:
    t: float, time
    theta: float, the shift of the gaussian envelope
    Omega: float, the frequency of the gaussian envelope
    """

    return np.exp(-1 *np.power(t*theta*Omega,2)/2) * np.cos(Omega*(1-theta) * t)

@njit()
def gaussian_envelope(t, theta, Omega):
    """A gaussian envelope function that has no frequency shift

    Args:
    t: float, time
    theta: float, the shift of the gaussian envelope
    Omega: float, the frequency of the gaussian envelope
    """

    return np.exp(-1 *np.power(t*theta*Omega/(1 - theta),2)/2) * np.cos(Omega * t)

@njit()
def exp_envelope(t, gamma, Omega):
    """An exponential envelope function with no frequency shift"""
    return np.exp(-gamma*t) * np.cos(Omega * t)


@njit()
def cetina_thermal(t, theta, Omega):
    """The full Cetina function for the single COM mode
    
    Args:
    t: float, time
    theta: float, the shift of the gaussian envelope
    Omega: float, the frequency of the gaussian envelope
    
    Returns:
    float: the value of the function at time t
    """
    phi = np.arctan(Omega*theta*t)
    C = 1/np.sqrt((1 + (theta*Omega*t)**2))
    return C * np.cos(Omega*t - phi)

@njit()
def cetina_thermal_exp(t, theta, Omega):
    """The full Cetina function for the single COM mode
    
    Args:
    t: float, time
    theta: float, the shift of the gaussian envelope
    Omega: float, the frequency of the gaussian envelope
    
    Returns:
    float: the value of the function at time t
    """
    phi = np.arctan(Omega*theta*t)
    C = 1/np.sqrt((1 + (theta*Omega*t)**2))
    return 0.5 - C * np.cos(Omega*t - phi)/2

#@njit()
def cetina_envelope(t, theta, Omega):
    """The full Cetina function for the single COM mode
    
    Args:
    t: float, time
    theta: float, the shift of the gaussian envelope
    Omega: float, the frequency of the gaussian envelope
    
    Returns:
    float: the value of the function at time t
    """
    C = 1/np.sqrt((1 + (theta*t)**2))
    return  C * np.cos(Omega*t)

def cetina_envelope_exp(t, theta, Omega):
    """The full Cetina function for the single COM mode
    
    Args:
    t: float, time
    theta: float, the shift of the gaussian envelope
    Omega: float, the frequency of the gaussian envelope
    
    Returns:
    float: the value of the function at time t
    """
    C = 1/np.sqrt((1 + (Omega*theta*t)**2))
    return  0.5 - C * np.cos(Omega*t)/2


#@njit((float64[:], float64 ,float64[:], int32), parallel=True)
def generate_experimental_data(times:np.ndarray, Omega:float, sds:np.ndarray, shots:int, return_std=False):
    """Generates the experimental data for the single COM mode. Returns the averaged results with shape (times, standard_deviations)
    
    Args:
    times: np.array, the times to evaluate the function at
    Omega: float, the frequency of the gaussian envelope
    sds: np.array, the standard deviation of the thermal distribution
    shots: int, the number of shots to take
    
    Returns:
    np.array: the experimental data
    """
    data = np.zeros((len(times), len(sds)), dtype=np.float64)
    std = np.zeros((len(times), len(sds)), dtype=np.float64)
    for t_i in prange(len(times)):
        for sd_i in prange(len(sds)):
            t = times[t_i]
            sd = sds[sd_i]
            data[t_i, sd_i] = np.average(np.cos(Omega*t*(1 - thermal_rejection(sd, shots))))
            std[t_i, sd_i] = np.std(np.cos(Omega*t*(1 - thermal_rejection(sd, shots))))
    if return_std:
        return data, std
    else:
        return data

@njit((float64[:], float64, float64))
def rabi_flop_gauss_fit_shifted(t, theta, Omega):
    return (1 - np.exp(-1*np.power(Omega*t*theta,2)/2)*np.cos(Omega*t*(1 - theta)))/2

@njit((float64[:], float64, float64))
def rabi_flop_gauss_fit(t, theta, Omega):
    return (1 - np.exp(-1*np.power(Omega*t*theta,2)/2)*np.cos(Omega*t))/2

def calibrate_sim_Omegas(theta_list, Omega_target, times, debug=False, return_scale_factors=True):
    #Step 1: Generate the basic sim data:
    sim_data = np.zeros((len(theta_list), len(times)))
    for i, theta in enumerate(theta_list):
        sim_data[i, :] = cetina_thermal_exp(times, theta, Omega_target)
    #Step 2: Fit the data to the basic Cetina Envelope function
    Omega_round_1 = np.zeros(len(theta_list))
    for i in range(len(theta_list)):
        popt, pcov = curve_fit(cetina_envelope_exp, times, sim_data[i, :], p0=[0.05, Omega_target])
        #if debug:
        #plt.plot(times, sim_data[i, :])
        #plt.plot(times, cetina_envelope_exp(times, *popt))
        #print(theta_list[i], popt[0])
        Omega_round_1[i] = popt[1]
    
    
    #Step 3: Get the Scale Factors:
    if debug:
    #print(Omega_round_1)
    #print(Omega_target)
        plt.plot(theta_list, Omega_round_1, 'o')
    #Perform a linear fit to get the scale factors:
    #########
    popt, pcov = curve_fit(lambda x, a, b: a*x + b, theta_list, Omega_round_1)
    linear_Omegas = popt[0]*np.array(theta_list) + popt[1]
    scale_factors = Omega_target/linear_Omegas
    if debug:
        Omega_round_2 = np.zeros(len(theta_list))
        sim_data = np.zeros((len(theta_list), len(times)))
        for i, theta in enumerate(theta_list):
            sim_data[i, :] = cetina_thermal_exp(times, theta, Omega_target*scale_factors[i])
        for i in range(len(theta_list)):
            popt, pcov = curve_fit(cetina_envelope_exp, times, sim_data[i, :], p0=[0.05, Omega_target])
            Omega_round_2[i] = popt[1]
        plt.plot(theta_list, Omega_round_2, 'X')
        plt.plot(theta_list, linear_Omegas)
        plt.hlines(Omega_target, theta_list[0], theta_list[-1], linestyles='dashed')
        plt.show()
    if return_scale_factors:
        return scale_factors
    else:
        return Omega_target*np.sqrt(scale_factors)

def zero_temperature_Omega(theta, Omega, times, debug=False, return_scale_factors=True):
    #Step 1: Generate the basic sim data:
    theta_list = np.linspace(0, theta, 20, endpoint=True)
    
    sim_data = np.zeros((len(theta_list), len(times)))
    for i, theta in enumerate(theta_list):
        sim_data[i, :] = cetina_thermal_exp(times, theta, Omega)
        
    #Step 2: Fit the data to the basic Cetina Envelope function
    Omega_round_1 = np.zeros(len(theta_list))
    for i in range(len(theta_list)):
        popt, pcov = curve_fit(cetina_envelope_exp, times, sim_data[i, :], p0=[0.05, Omega])
        Omega_round_1[i] = popt[1]
        
    #Correction at theta:
    correction = Omega_round_1[-1]/Omega
    #We now have theta vs. Omega -- get Omega at zero:
    return Omega_round_1[0]/correction

    #Step 3: Get the Scale Factors:
    if debug:
        plt.plot(theta_list, Omega_round_1, 'o')
        plt.show()
        
    scale_factors = Omega/Omega_round_1
    if return_scale_factors:
        return scale_factors
    else:
        return Omega_round_1

def calibrate_Js(J_target, thetas, times, debug=False, return_function=False):
    """Returns a calibrated list of Js matching the input of thetas such that the fit of the frequency over the time scale is equal to J

    Args:
        J (float): Target value of J
        thetas (array: float): values of theta to calibrate over
        times (array: float): times to optimize over
        debug(bool): whether to plot the results
    """
    #This is the function that we want to fit to fake data -- It extracts a fixed Omega value
    def fit_function(t, theta, Omega):
        """The full Cetina function for the single COM mode
        shots
        
        Args:
        t: float, time
        theta: float, the shift of the gaussian envelope
        Omega: float, the frequency of the gaussian envelope
        
        Returns:
        float: the value of the function at time t
        """
        #It's just an envelope of the expected form and a constant Cos function
        C = 1/np.sqrt((1 + (Omega*theta*t)**2))
        return  C * np.cos(2*Omega*t)
    
    #This generates the fake Jij data by averaging over 100 thermal samples
    def generate_fake_data(times, J_value, theta):
        averages = 100
        output_expectations = np.zeros(len(times))
        for _ in range(averages):  
            theta_sample = thermal_rejection(theta, 1)[0]     
            output_expectations += np.cos(2*J_value*((1-theta_sample)**2)*times)
        return output_expectations/averages
    #The cost function takes in the value of x we want to optimize over, adn also the argument for theta to pass ot the fake data
    def cost_function(J_test, theta):
        get_Y_data_for_J = generate_fake_data(times, J_test, theta)
        #Get the frequency of the fake data for that J
        popt, pcov = curve_fit(fit_function, times, get_Y_data_for_J, p0=[0.05, J_test[0]])
        #The cost function is the value of the observed Omega vs. the target value which is J
        return np.abs(popt[1]-J_target)


    output_Js = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):
        result = minimize(cost_function, J_target, args=(theta))
        output_Js[i] = result.x[0]
    if debug:
        plt.plot(thetas, output_Js, 'o')
        plt.show()
        plt.plot(times, generate_fake_data(times, output_Js[0], thetas[0]))
        plt.show()
    return output_Js
