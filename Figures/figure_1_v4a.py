import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
import qutip as quit
from qutip import (about, basis, expect, mesolve, qeye, sigmax, sigmay, sigmaz, tensor, mcsolve)
from qutip.measurement import measure_observable, measurement_statistics
from qutip.expect import expect
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
#Add util to the path
import sys
sys.path.append('../util')
sys.path.append('../')

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
#import savgol:
from scipy.signal import savgol_filter
from fitting_functions_ZNE import third_no_first_ZNE, order_poly_ZNE, order_poly_instance, converge_ZNE_order, converge_ZNE_loocv, pade_fit_ZNE

from qutip import (about, basis, expect, mesolve, qeye, sigmax, sigmay, sigmaz, tensor, mcsolve, sesolve, mesolve)
from basic_TFIM_funcs import get_s_lists
from joblib import Parallel, delayed
from functions_util import cetina_thermal_exp, cetina_envelope_exp,calibrate_sim_Omegas
from samplers_util import thermal_rejection
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import qutip as quit
from qutip import (about, basis, expect, mesolve, qeye, sigmax, sigmay, sigmaz, tensor, mcsolve)
from qutip.measurement import measure_observable, measurement_statistics
from qutip.expect import expect
from scipy.optimize import curve_fit

#Add util to the path
import sys
sys.path.append('../util')
from samplers_util import thermal_rejection
from fitting_functions_ZNE import third_no_first_ZNE, converge_ZNE_loocv
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""DATA GENERATION"""
def exp_fit(t, A, B, C, D):
    return A*np.exp(-B*t)*np.cos(C*t+D) + A
def rabi_fit(t, Omega):
    return (1-np.cos(Omega*t))/2
KHz = 1e3
micro = 1e-6
Omega =  2210*KHz
sigma = .1*Omega #MODIFIED
delta_samples = 300
dense_samples = 100#000
nano = 1e-9
n_bar = 800
time_points = 100

t = np.linspace(0, 40*micro, time_points)
def Rabi(Omega_array, t_array):
    output = np.zeros((len(Omega_array), len(t_array)))
    for i, Omega in enumerate(Omega_array):
        output[i] = (1-np.cos(Omega*t_array/2))/2
    return output

NORMALIZER = 280/0.15

def sample_thermal_n(n_bar:int):
    n_max = 20*n_bar
    
    n_bar = int(n_bar*NORMALIZER)
    n_max = int(n_max*NORMALIZER)
    if n_bar == 0:
        return 0
    #print(n_bar, n_max)
    def pr_n(n: int, n_bar:int):
        """Returns the probability of sampling the value $n$ from the thermal distribution"""
        return (1/(1 + n_bar)) * np.power((n_bar/(1 + n_bar)),n)
        return (1 - n_bar) * n_bar**n
    pr_0 = pr_n(0, n_bar)
    
    n = np.random.randint(0, n_max)
    r = np.random.uniform(0, pr_0)
    pr_n_sample = pr_n(n, n_bar)
    while r > pr_n_sample:
        n = np.random.randint(0, n_max)
        r = np.random.uniform(0, pr_0)
        pr_n_sample = pr_n(n, n_bar)
    return n/NORMALIZER - n_bar/NORMALIZER

def sample_normal(theta:float):
    return np.random.normal(0, theta)
    #return n
    
theta_alpha = -42409
Omega_test = 3750000
max_time = 10
times = np.linspace(0, max_time*micro, time_points)
n_bar = 0.04
delta_thermal = np.array([sample_normal(n_bar) for i in range(delta_samples)])
delta_thermal_dense = np.array([sample_normal(n_bar) for i in range(dense_samples)])

P_thermal = Rabi(Omega_test*(1+delta_thermal), times)
P_avg_thermal = np.mean(P_thermal, axis=0)

times_dense = times#np.linspace(0, *micro, time_points)
times_dense_plotting = np.linspace(0, max_time, time_points)
P_thermal_dense = Rabi(Omega_test*(1+delta_thermal), times_dense)
P_thermal_dense_dense = Rabi(Omega_test*(1+delta_thermal_dense), times_dense)

P_avg_thermal_dense = np.mean(P_thermal_dense, axis=0)

def gen_noisy_rabi(Omega, theta, delta_samples, target_time=None):
    if target_time is None:
        target_time = times_dense
    delta_thermal = np.array([sample_normal(theta) for i in range(delta_samples)])
    P_thermal = Rabi(Omega*(1+delta_thermal), target_time)
    #print(P_thermal.shape)
    P_avg_thermal = np.mean(P_thermal, axis=0)
    P_std_thermal = np.std(P_thermal, axis=0)/np.sqrt(delta_samples)
    return P_avg_thermal, P_std_thermal




"""Formatting the plot/figure"""
"""fig_aure includes Rabi oscillations for Gaussian noise, and regions for Rydberg and trapped ion error mechanisms. 
|---|--|
|   |  |
|-|-|--|
| | |  |
|-|-|--|
"""

columns = 1
rows = 2

"""
scol=(3+3/8) #single column width of PRL,
dcol=2*scol # double column,
"""

## Nat Comm
mmtoinch = 1/25.4
scol = 88 * mmtoinch
dcol = 180 * mmtoinch
size_col = [scol, dcol][columns-1]
if columns == 1:
    size_row = scol*rows
else:
    size_row = dcol*rows

fullwidth=6.3,
gr=(np.sqrt(5.0) - 1.0) / 2.0 #golden ratio,
#print(gr)

mpl.rcParams.update({
    "ytick.direction": "in",
    "xtick.direction": "in",
    "axes.labelpad": 0,
    "font.size": 7,
    "legend.frameon": True,
    "figure.dpi":200,
    "figure.figsize": [size_col, size_row * gr],
    "font.family": "Helvetica",
    #"axes.prop_cycle": cycler('color', palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)
    # "axes.prop_cycle": cycler('color', palettable.colorbrewer.sequential.Reds_9.mpl_colors[1:])
})

#Create a matplotlib subplots in the layout above
fig_a = plt.figure()#figsize=(12, 6))

#gs = gridspec.GridSpec(2,5)  # 2x2 grid
#ax_a = fig_a.add_subplot(gs[0,:])
#ax_bi = fig_a.add_subplot(gs[1,0:3])

gs = gridspec.GridSpec(8,7)  # 2x2 grid
ax_a = fig_a.add_subplot(gs[0:3,:])
ax_bi = fig_a.add_subplot(gs[3:7,0:4])



#gs = gridspec.GridSpec(2,5)
#ax_1 = fig.add_subplot(gs[0,:])
#ax_2 = fig.add_subplot(gs[1,0:3])



#ax_a = fig_a.add_subplot(211)
#ax_bi = fig_a.add_subplot(223)
#The numbers, what do they mean??
#row gird, column grid, index in grid



"""CREATE THE INSET AXIS"""

# Move the labels inside the plot

"""ZNE Section/ Plotting"""

#### BASE PLOTTING TIME DOMAIN
for i in range(0,P_thermal_dense.shape[0], delta_samples//100):
    ax_a.plot(times_dense_plotting, P_thermal_dense[i,:], 'b', alpha=0.02, zorder=-1) 
ax_a.plot(times_dense_plotting, P_avg_thermal_dense, 'b', alpha=0.9, zorder=-1, label='Noisy Trajectories')
ax_a.plot(times_dense_plotting, P_avg_thermal_dense, c='k', label='Averaged Expectation')



#### COLOR BAR CREATION FOR TEMPERATURES



ZNE_points = np.sqrt(np.array([0, 1, 1.1, 1.3, 1.6, 2 ,2.5]))#,2.5, 3, 3.5])
#ZNE_points = np.array([0,1.0, 1.0810140527710055, 1.3174929343376374, 1.6902785321094298])#, 2.1691699739962274, 2.71537032345343])
#ZNE_points = np.array([0, 1, 1.1, 1.3, 1.6, 2 ])
def create_linear_function(y_at_0_5):
    # Given points
    x1, y1 = 0.5, y_at_0_5
    x2, y2 = 1, 1
    
    # Calculate slope (a) and intercept (b)
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    
    # Return the linear function
    def linear_function(x):
        return a * x + b
    
    return linear_function

linear_func = create_linear_function(0.3)
normalized_ZNE_points_for_color = linear_func(ZNE_points/max(ZNE_points))
#cm_subsection = np.linspace(0.2, 1.0, len(ZNE_points) )
Reds = [ cm.Reds(x) for x in normalized_ZNE_points_for_color ]
cmap = LinearSegmentedColormap.from_list('name', Reds[1:])
#print(Reds)
baseline=n_bar#np.sqrt(n_bar)  # MODIFIED
#get a blue to red colormap
colors = Reds#cmap(np.linspace(0, 1, len(ZNE_points)))

colorbar = plt.cm.ScalarMappable(cmap=cmap)

# Add color bar to ax_bii
test_labels = np.asarray([0.001,  0.002, 0.003])#np.power(np.array([0.04, 0.05, 0.06, 0.07]),2)
#labels = np.array(['{:.4f}'.format(test_labels[i-1]) for i in range(1,len(ZNE_points))])

normalized_ticks = normalized_ZNE_points_for_color[1:]
ticks_test = linear_func(test_labels/((max(ZNE_points)*baseline)**2))
cbar = plt.colorbar(colorbar, ax=ax_bi, ticks=ticks_test, orientation='horizontal')#normalized_ZNE_points_for_color[1:])
cbar.set_ticklabels(test_labels)

ZNE_data = np.zeros((len(times_dense), len(ZNE_points)))
ZNE_std = np.zeros((len(times_dense), len(ZNE_points)))
for i in range(len(ZNE_points)):
    color = colors[i]
    
    ZNE_data[:,i], ZNE_std[:,i] = gen_noisy_rabi(Omega_test, baseline*ZNE_points[i], delta_samples)
    if i == 0:
        ax_bi.plot(times_dense_plotting, ZNE_data[:,i], c='k', ls='--', zorder=1000, alpha=1, label='Zero Temperature')#, label='Analytic')
    else:
        ax_bi.plot(times_dense_plotting, ZNE_data[:,i], c=color, alpha=1)
    
#Draw a vertial line at the 3rd peak:
peak_index = 84
#ax_bi.axvline(x=times_dense_plotting[peak_index], c='k')#, linestyle='-')

#print(dense_data.shape)
#print(dense_data_std.shape)
def analytic_function(t, Omega, sigma):
    return (1-np.exp(-np.power(t*sigma*Omega,2)/8)*np.cos(Omega_test*t/2))/2



#ax_bii.plot(np.power(ZNE_points_dense,2), analytic_function(times_dense[peak_index], Omega_test, ZNE_points_dense), color='purple', zorder=-1, label='Analytic')#, alpha=0.2)#, linewidths=0)

#Get the ZNE points
ZNEd_data = np.zeros((len(times_dense)))
ZNE_orders = np.zeros((len(times_dense)))
residuals = np.zeros((len(times_dense)))


## Ensures the order of the fit doesn't jump around -- we expect it to start around 1, go up to around 2
for i in range(len(times_dense)):
    data = ZNE_data[i,:]
    function, residual_error, order = converge_ZNE_loocv(np.power(ZNE_points[1:]*baseline,2), data[1:], return_order=True, remove_first=False, return_cov=True)
    ZNEd_data[i] = function(0)
    ZNE_orders[i] = order
    residuals[i] = residual_error
    
    try:
        #print(order, ZNE_orders[i-1])
        if i < 5:
            raise Exception('Escape')
        if np.abs(order - ZNE_orders[i-1]) > 1:
            ZNE_orders[i] = ZNE_orders[i-1]
            function, residuals[i] = converge_ZNE_order(np.power(ZNE_points[1:]*baseline,2), data[1:], order=ZNE_orders[i-1],remove_first=False, return_cov=True)
            ZNEd_data[i] = function(0)
            #converge_ZNE_loocv(ZNE_points[1:]*baseline, data[1:], return_order=True, debug=True)
    except:
        pass
#print(residuals)
#print(ZNE_orders)


# Calculate bounds
upper_bound = ZNEd_data + residuals
lower_bound = ZNEd_data - residuals

# Replace errorbar with fill_between
ax_bi.fill_between(times_dense_plotting, 
                   lower_bound, 
                   upper_bound, 
                   color='blue', 
                   alpha=0.3)  # Add transparency
ax_bi.plot(times_dense_plotting, ZNEd_data, c='blue', label='Extrapolated')

# Optional: Add central line
#ax_bi.plot(times_dense_plotting, ZNEd_data, 'b-', linewidth=1)
#ax_bi.scatter(times_dense_plotting, ZNE_orders/max(ZNE_orders))#, c='k', ls='--', label='Extrapolated')



ax_a.set_xlabel(r'Time ($\Omega t$)', usetex=1)
ax_a.set_ylabel('Population Transfer')

#Text Labels:
#\Xi pointing to error bars
#\langle O(T)\rangle_{\mathcal{D}_{\theta_i}}
#\langle O(T) \rangle_{\mathcal{D}_{0}}
#E^{*}
# Using data coordinates instead of normalized coordinates
# Assuming x range is 0 to (max(ZNE_points)*baseline)**2
# and y range matches ZNE_data range

x_pos = (max(ZNE_points)*baseline)**2 * 0.7  # 70% of x-axis range

#<O(T)>_{D_theta_i} + \xi_i coordinates: line points to: 0, ZNE_data[peak_index,3]
#<O(T)>_{D_0} coordinates: line points to: 0, ZNE_data[peak_index,0]
#E^{*}, line points to: 0, function_best(0)


# Position text at different y-levels
# Get x position for labels (e.g. 70% of x-axis range)
x_pos = (max(ZNE_points)*baseline)**2 * 0.7



# Using same data coordinates as plot



ax_bi.set_xlabel(r'Time ($\Omega t$)', usetex=1)
ax_bi.set_ylabel(r'Population Transfer', usetex=1)

#ax_bi.legend(loc='lower right')


ax_bi.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax_a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])



fig_a.tight_layout(pad=0.01, h_pad=0.2, w_pad=0.2)


#All Subfigure stuff
x, y = 1.035, -0.42
ax_bii = inset_axes(ax_bi, width="90%", height="97%", bbox_to_anchor=(x, y, 0.6, 1.38), bbox_transform=ax_bi.transAxes, loc='lower left')

# Define the x-range and y-range to zoom in on
x_inset_range = (8, 9)
y_inset_range = (0.65, 1.05)

# Plot data in the inset axes
# ax_ci.plot(x, y, label='Inset Data')  # Uncomment and replace with actual data
ax_bii.set_xlim(x_inset_range)
ax_bii.set_ylim(y_inset_range)

# Optional: Add a box showing the zoomed area
inset_zoom = ax_bi.indicate_inset_zoom(ax_bii, edgecolor="k", )
connector_lines = inset_zoom[1]  # The connector lines are returned as a tuple
connector_lines[0].set_visible(True)  # Hide the lower-left connector line
connector_lines[1].set_visible(True)   # Show the upper-left connector line
connector_lines[2].set_visible(False)   # Show the lower-right connector line
connector_lines[3].set_visible(False)  # Hide the upper-right connector line
#Add ticks to the inset indicator:

#Plot the extrapolation:
for i in range(len(ZNE_points)):
    color = colors[i]
    if i == 0:
        color = 'k'
        #ax_bii.errorbar(np.power(ZNE_points[i]*baseline,2), ZNE_data[peak_index,i], yerr=ZNE_std[peak_index,i], color=color, fmt='o', capsize=0, markersize = 3)
        ax_bii.hlines(ZNE_data[peak_index,i],0, (max(ZNE_points)*baseline)**2,  color=color, linestyle='--', label='Noiseless')
    else:
        ax_bii.errorbar(np.power(ZNE_points[i]*baseline,2), ZNE_data[peak_index,i], yerr=ZNE_std[peak_index,i], color=color, fmt='o', capsize=2, markersize = 3)



function_best, fit_error = converge_ZNE_loocv(np.power(ZNE_points[1:]*baseline,2), ZNE_data[peak_index,1:], remove_first=False, return_order=False, return_cov=True)
ZNE_points_dense = np.linspace(0, max(ZNE_points), 50)*baseline
dense_data     = np.zeros((len(times_dense_plotting), len(ZNE_points_dense)))
dense_data_std = np.zeros((len(times_dense_plotting), len(ZNE_points_dense)))
for i in tqdm(range(len(ZNE_points_dense))):
    dense_data[:,i], dense_data_std[:,i] = gen_noisy_rabi(Omega_test, ZNE_points_dense[i], delta_samples)


ax_bii.plot(np.power(ZNE_points_dense,2), function_best(np.power(ZNE_points_dense,2)), c='blue', label='Zero Noise Extrapolation')
#Make this a fill between:
ax_bii.errorbar(0, function_best(0), yerr=fit_error, c='blue', fmt='o', capsize=2, markersize = 3)


ax_bii.annotate(r'$\langle \hat{O}(T)\rangle_{\theta_i} + \xi_i$',
                xy=((ZNE_points[4]*baseline)**2, ZNE_data[peak_index,4]),  # point to annotate
                xytext=(0.0, 0.7),  # text position
                ha='left',
                va='center',
                arrowprops=dict(
                    arrowstyle='->',
                    color=colors[4],
                    connectionstyle='arc3,rad=0'
                ), 
                color=colors[3], usetex=1)

ax_bii.annotate(r'$\langle \hat{O}(T) \rangle_{0}$',
                xy=(0.0005, ZNE_data[peak_index,0]+0.01),
                xytext=(x_pos*0.7, ZNE_data[peak_index,0]+0.03),
                ha='left',
                va='center',
                arrowprops=dict(
                    arrowstyle='->',
                    color='k',
                    connectionstyle='arc3,rad=0'
                ), 
                color="k", usetex=1)

ax_bii.annotate(r'$E^{*}$',
                xy=(0+0.0002, function_best(0)),
                xytext=(x_pos, function_best(0)-0.05),
                ha='left',
                va='center',
                arrowprops=dict(
                    arrowstyle='->',
                    color='blue',
                    connectionstyle='arc3,rad=0'
                ), 
                color="blue", usetex=1)

ax_bii.tick_params(axis='x', direction='in', labeltop=False, labelbottom=True)
ax_bii.set_ylabel('Population Transfer')
ax_bii.set_xlabel(r'Noise Strength ($\theta = \sigma^{2}$)', usetex=1)

ax_bii.set_xlim(-0.0005, np.power(max(ZNE_points)*baseline,2)+0.0005)
ax_bii.yaxis.tick_right()
ax_bii.yaxis.set_label_position('right')
ax_bii.set_xticks(np.array([0, 0.002, 0.004]))
#ax_bii.set_xticks(np.power(np.array([0, 0.04, 0.08]),2))
#ax_bii.set_xticklabels(['0', '0.0016', '0.0064'])#, '8.6', '8.8'])
ax_bii.set_yticks([1, 0.9, 0.8, 0.7])#Put these on the left
ax_bii.set_yticklabels([1, 0.9, 0.8, 0.7])
ax_bii.tick_params(axis='y', direction='in', labelleft=False, labelright=True)
ax_bii.tick_params(axis='x', direction='in', labeltop=False, labelbottom=True)



cbar.set_label(r'Noise Strength ($\theta = \sigma^{2}$)', labelpad=5, usetex=1)
ax_bii.legend(
    loc='upper center',  # Anchor point for the legend
    bbox_to_anchor=(-0.4, -0.15),  # (x, y) coordinates relative to the axes
    ncol=2,  # Keep two columns
    borderaxespad=0.  # Padding between axes and legend
)
ax_a.legend(loc='lower right',ncol=2)

#plt.show()

#print(fig_a)
fig_a.savefig('Figure_1_v4_test.png', dpi=600)
#fig_a.savefig('Figure_1_v4_test.svg', dpi=600)
#fig_a.savefig('Figure_1_v4_test.pdf', dpi=600)

#fig2.savefig('Figure_1_v4b.png', dpi=600)
#fig2.savefig('Figure_1_v4b.svg', dpi=600)
#fig2.savefig('Figure_1_v4b.pdf', dpi=600)
