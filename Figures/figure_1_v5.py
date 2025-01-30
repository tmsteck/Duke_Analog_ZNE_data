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
delta_samples = 600
dense_samples = 600#000
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


ORDER = 3

theta_alpha = -42409
Omega_test = 3750000
max_time = 10
times = np.linspace(0, max_time*micro, time_points)
n_bar = 0.08
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
    print(P_avg_thermal.shape)
    #P_std_thermal = np.std(P_thermal, axis=0)/np.sqrt(delta_samples)
    P_std_thermal = (P_thermal)*(1-P_thermal)/np.sqrt(delta_samples)
    P_std_thermal = np.zeros(P_avg_thermal.shape)
    for i in range(len(target_time)):
        P_std_thermal[i] += (P_avg_thermal[i]*(1 - P_avg_thermal[i]))/np.sqrt(delta_samples)
    return P_avg_thermal, P_std_thermal

data_file = np.load('Figure_1_full_data_ZNE.npz')
x = data_file['x']
y = data_file['y']
print(y.shape)
y_error = data_file['y_error']
y_ideal = data_file['y_ideal']
times = data_file['times'] /micro
ZNE_results = data_file['ZNE_results']
ZNE_error = data_file['ZNE_error']
dense_x = data_file['dense_x']
ZNE_y_dense = data_file['ZNE_y_dense']

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

fig_a = plt.figure()#figsize=(12, 6))

gs = gridspec.GridSpec(8,7)  # 2x2 grid
ax_a = fig_a.add_subplot(gs[0:3,:])
ax_bi = fig_a.add_subplot(gs[3:7,0:4])



"""CREATE THE INSET AXIS"""


"""ZNE Section/ Plotting"""

#### BASE PLOTTING TIME DOMAIN
trajectories_color = 'darkgreen'
for i in range(0,P_thermal_dense.shape[0], delta_samples//100):
    ax_a.plot(times_dense_plotting, P_thermal_dense[i,:], trajectories_color, alpha=0.02, zorder=-1)

ax_a.plot(times_dense_plotting, P_avg_thermal_dense, trajectories_color, alpha=0.9, zorder=-1, label='Noisy Trajectories')
ax_a.plot(times_dense_plotting, P_avg_thermal_dense, c='k', label='Averaged Expectation')



#### COLOR BAR CREATION FOR TEMPERATURES



ZNE_points = np.sqrt(np.array([0, 1, 1.1, 1.3, 1.6, 2 ,2.5]))#,2.5, 3, 3.5])
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

print(y.shape)


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
#Whole number multiples of 0.001 between n_bar**2 and (n_bar)**2*ZNE_points[-1]
minimum_label = (n_bar**2//0.001)
maximum_label = ((n_bar*ZNE_points[-2])**2//0.001)
test_labels = np.arange(minimum_label, maximum_label+1, 1)*0.001 #include endpoints:
middle_label = (maximum_label)//2
test_labels = np.array([middle_label, maximum_label])*0.001
#test_labels = [test_labels[i] for i in range(len(test_labels)) if i == 2 or i == len(test_labels) - 1]#Check if any integers between 

#labels = np.array(['{:.4f}'.format(test_labels[i-1]) for i in range(1,len(ZNE_points))])

normalized_ticks = normalized_ZNE_points_for_color[1:]
ticks_test = linear_func(test_labels/((max(ZNE_points)*baseline)**2))
cbar = plt.colorbar(colorbar, ax=ax_bi, ticks=ticks_test, orientation='horizontal')#normalized_ZNE_points_for_color[1:])
cbar.set_ticklabels(test_labels)
print(y.shape)
assert times.shape[0] == y[:,0].shape[0], 'Shape mismatch: times: {}, y: {}'.format(times.shape, y[:,0].shape[0])
for i in range(len(x)):
    color = colors[i]
    
    if i == 0:
        #ax_bi.plot(times_dense_plotting, ZNE_data[:,i], c='k', ls='--', zorder=1000, alpha=1, label='Zero Temperature')#, label='Analytic')
        ax_bi.plot(times, y_ideal, c='k', ls='--', zorder=1000, alpha=1, label='Zero Temperature')#, label='Analytic')
    else:
        #ax_bi.plot(times_dense_plotting, ZNE_data[:,i], c=color, alpha=1)
        ax_bi.plot(times, y[:,i], c=color, alpha=1)
        ax_bi.fill_between(times, y[:,i] - y_error[:,i], y[:,i] + y_error[:,i], color=color, alpha=0.3)
    
#Draw a vertial line at the 3rd peak:
peak_index = 84

def analytic_function(t, Omega, sigma):
    return (1-np.exp(-np.power(t*sigma*Omega,2)/8)*np.cos(Omega_test*t/2))/2




# Calculate bounds

ax_bi.fill_between(times, 
                   ZNE_results - ZNE_error,
                   ZNE_results + ZNE_error,
                   color='blue', 
                   alpha=0.3)  # Add transparency
ax_bi.plot(times, ZNE_results, c='blue', label='Extrapolated')



ax_a.set_xlabel(r'Time ($\Omega t$)', usetex=1)
ax_a.set_ylabel('Population Transfer')


x_pos = (max(ZNE_points)*baseline)**2 * 0.7  # 70% of x-axis range


x_pos = (max(ZNE_points)*baseline)**2 * 0.7



# Using same data coordinates as plot



ax_bi.set_xlabel(r'Time ($\Omega t$)', usetex=1)
ax_bi.set_ylabel(r'Population Transfer', usetex=1)

#ax_bi.legend(loc='lower right')


ax_bi.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax_a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])



fig_a.tight_layout(pad=0.01, h_pad=0.2, w_pad=0.2)


#All Subfigure stuff
x_coord, y_coord = 1.035, -0.42
ax_bii = inset_axes(ax_bi, width="90%", height="97%", bbox_to_anchor=(x_coord, y_coord, 0.6, 1.38), bbox_transform=ax_bi.transAxes, loc='lower left')

# Define the x-range and y-range to zoom in on
x_inset_range = (8, 9)
y_inset_range = (np.min(y[peak_index])-.05, 1.05)

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
for i in range(len(x)):
    color = colors[i]
    print(color)
    if i == 0:
        color = 'k' 
        ax_bii.hlines(y_ideal[peak_index],0, x[-1],  color=color, linestyle='--', label='Noiseless')
    else:
        ax_bii.errorbar(x[i], y[peak_index,i], yerr=y_error[peak_index,i], color=color, fmt='o', capsize=2, markersize = 3, zorder=1000)





ax_bii.plot(dense_x, ZNE_y_dense[peak_index], c='blue', label='Zero Noise Extrapolation')
#Make this a fill between:
ax_bii.errorbar(0, ZNE_results[peak_index], yerr=ZNE_error[peak_index], c='blue', fmt='o', capsize=2, markersize = 3)


ax_bii.annotate(r'$\langle \hat{O}(T)\rangle_{\theta_i} + \xi_i$',
                xy=(x[3], y[peak_index,3]),  # point to annotate
                xytext=(0.0, 0.55),  # text position
                ha='left',
                va='center',
                arrowprops=dict(
                    arrowstyle='->',
                    color=colors[3],
                    connectionstyle='arc3,rad=0'
                ), 
                color=colors[3], usetex=1)

ax_bii.annotate(r'$\langle \hat{O}(T) \rangle_{0}$',
                xy=(0.0005, y_ideal[peak_index]+0.01),
                xytext=(x_pos*0.7, y_ideal[peak_index]+0.03),
                ha='left',
                va='center',
                arrowprops=dict(
                    arrowstyle='->',
                    color='k',
                    connectionstyle='arc3,rad=0'
                ), 
                color="k", usetex=1)

ax_bii.annotate(r'$E^{*}$',
                xy=(0+0.0002, ZNE_results[peak_index]),
                xytext=(x_pos, ZNE_results[peak_index]-0.05),
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
ax_bii.set_xticks(np.concatenate((np.array([0]), test_labels)))#np.array([0, 0.002, 0.004]))
#ax_bii.set_xticks(np.power(np.array([0, 0.04, 0.08]),2))
#ax_bii.set_xticklabels(['0', '0.0016', '0.0064'])#, '8.6', '8.8'])
ax_bii.set_yticks([1, 0.9, 0.8, 0.7, 0.6])#Put these on the left
ax_bii.set_yticklabels([1, 0.9, 0.8, 0.7, 0.6])
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

#Export: X, Y, Y_error, Y_ideal
# (n_bar*ZNE_points)**2, ZNE_data[:,1:], ZNE_std[:1,:], ZNE_data[:,0]
#np.savez('figure_1_data.npz', x=(n_bar*ZNE_points[1:])**2, y=ZNE_data[:,1:], y_error=ZNE_std[:,1:], y_ideal=ZNE_data[:,0], times=times)
# ZNE_data[1:,], ZNE_std[1:,]


#print(fig_a)
fig_a.savefig('Figure_1_v6.png', dpi=600)
fig_a.savefig('Figure_1_v6.svg', dpi=600)
fig_a.savefig('Figure_1_v6.pdf', dpi=600)

#fig2.savefig('Figure_1_v4b.png', dpi=600)
#fig2.savefig('Figure_1_v4b.svg', dpi=600)
#fig2.savefig('Figure_1_v4b.pdf', dpi=600)
