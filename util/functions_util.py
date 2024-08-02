import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, prange
from numba import jit, float64, int32, boolean
from util.samplers_util import thermal_rejection

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