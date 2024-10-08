import numpy as np
from numba import jit, njit, prange
from numba import float64, int32, boolean




NORMALIZER = float64(280/0.15)

MAX_SAMPLES = int32(5000)

@njit((float64, float64))
def pr_n(n: float, n_bar:float):
        """Returns the probability of sampling the value $n$ from the thermal distribution"""
        return (1/(1 + n_bar)) * np.power((n_bar/(1 + n_bar)),n)


@njit((float64, int32), parallel=True)
def thermal_rejection(sd:float, n_samples:int):
    """Rejection sampling for the thermal distribution. Returns an array of size n_samples"""
    if np.abs(sd) < 1e-10:
        return np.zeros(n_samples, dtype=np.float64)
    n_bar = int(sd*NORMALIZER)
    n_max = 100*n_bar
    pr_0 = pr_n(0, n_bar)
    
    outputs = np.zeros(n_samples, dtype=np.float64)
    for n_i in prange(n_samples):
        n = np.random.randint(0, n_max)
        r = np.random.uniform(0, pr_0)
        pr_n_sample = pr_n(n, n_bar)
        while r > pr_n_sample:
            n = np.random.randint(0, n_max)
            r = np.random.uniform(0, pr_0)
            pr_n_sample = pr_n(n, n_bar)
        outputs[n_i] = n
    return outputs/NORMALIZER



@njit((float64,), parallel=True)
def get_pr_n_array(sd:float):
    """ Uses n_0 = 280, """
    n_bar = int(sd*NORMALIZER)
    n_grid_points = 5000
    n_grid = np.arange(0, n_grid_points)
    pr_n_grid = np.zeros(n_grid_points)
    for i in prange(n_grid_points):
        pr_n_grid[i] = pr_n(n_grid[i], n_bar)
    return pr_n_grid, n_grid/NORMALIZER


@jit((float64, float64, int32, float64, int32, boolean), nopython=True)
def returnThermalSamples_V2(var_min:float, var_max:float, variance_list_size:int, t_max:float, times_list_size:int, zero_mean=False):
    """ Returns an array of shape (times, variances) which is the thermal distribution summed over the first 5000 values of n
    
    Args:
    var_min: float, the minimum variance to sample from
    var_max: float, the maximum variance to sample from
    variance_list_size: int, the number of variances to sample from
    t_max: float, the maximum time to sample from
    times_list_size: int, the number of times to sample from
    zero_mean: bool, whether to zero the mean of the thermal distribution
    """
    times = np.linspace(0, t_max, times_list_size)
    variances = np.linspace(var_min, var_max, variance_list_size)
    frequencies = np.zeros((times_list_size, MAX_SAMPLES))
    for n_i in range(MAX_SAMPLES):
        frequencies[:, n_i] = np.cos(times*(1 - n_i/NORMALIZER))

        #print(n_i * times/normalizer)
    output = np.zeros((times_list_size, variance_list_size))
    for ti in range(times_list_size):
        for vi in prange(variance_list_size):
            pr_array, n_array = get_pr_n_array(variances[vi])
            if zero_mean:
                for n_i in prange(MAX_SAMPLES):
                    frequencies[:, n_i] = np.cos(times*(1 - n_i/NORMALIZER)/(1 - variances[vi]))
            else:
                pass
            output[ti, vi] = np.sum(frequencies[ti, :]*pr_array)
    return output



@njit((float64[:],float64[:]))
def f(time:np.ndarray, frequency:np.ndarray):
    return_array = np.zeros((len(time), len(frequency)))
    for t_i in range(len(time)):
        for f_i in range(len(frequency)):
            return_array[t_i, f_i] = np.cos(frequency[f_i]*time[t_i])
    return return_array

#@njit()
def returnGaussianSamples(n_samples:int, mean:float, var_min: float, var_max:float, variances:int, t_max:float, times:int):
    times = np.linspace(0, t_max, times)
    variances = np.linspace(var_min, var_max, variances)
    if mean == 0:
        means = np.zeros(len(variances))
    else:
        #means = np.sqrt(variances)
        means = variances
    
    #Output indexed [samples, times, variances]
    output = np.zeros((n_samples, len(times), len(variances)))
    for i in range(n_samples):
        output[i] = f(times, 1-np.random.normal(means, variances))
    return output



        
    