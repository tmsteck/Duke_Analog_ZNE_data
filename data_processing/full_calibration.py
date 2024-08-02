# Requirements: Matplotlib, Numpy, functions_util.py, index.py#, fitting_functions_ZNE.py, h5py

import os
import sys


sys.path.insert(0, './scripts')
sys.path.insert(1, './')
#print(sys.path)
#print(os.getcwd())
#Move the directory back to Duke_Data
#Check if the current directory is Duke_Data

PATH = os.path.expanduser('~') + '/Duke_Analog_ZNE_data/'
os.chdir(PATH)

#print(os.getcwd())
from util.index import print_index, get_experiment, index_folder
from util.functions_util import cetina_thermal_exp, cetina_envelope_exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import h5py


import tkinter as tk
from tkinter import filedialog


from util.functions_util import gaussian_envelope_shifted, cetina_thermal, gaussian_envelope, generate_experimental_data, rabi_flop_gauss_fit, rabi_flop_gauss_fit_shifted, cetina_thermal_exp,cetina_envelope
from util.fitting_functions_ZNE import log_fit_exp_ZNE, gaussian_ZNE, cubic_ZNE, linear_ZNE, Cetina_fit, exp_ZNE, third_no_first_ZNE, quadratic_only_ZNE


COUNTS_THRESHOLD = 1
ACTIVE_QUBITS = [0, 1, 2, 3]

#Basic parameters:
KHz = 1e3
omega =  900*KHz
us = 1e-6
theta = 0.05



def fit(function, x_axis, y_axis, return_cov=False, p0=[theta,omega], weights=None):
    # Fit the data to the function
    try:
        popt, pcov = curve_fit(function, x_axis, y_axis, p0=p0, sigma=weights, absolute_sigma=True, maxfev=10000)
    except:
        'Fit Failed, returning default'
        popt = p0
        pcov = None
    if return_cov:
        return popt, pcov
    return popt


def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    full_path = filedialog.askdirectory(initialdir=os.getcwd() + 'DUKE_ANALOG_ZNE_DATA/data')
    #only keep the last section after the last /
    folder_selected = full_path.split('/')[-1]
    return folder_selected, full_path

foldername, full_path = select_folder()
print(f"Selected folder: {foldername}")



# Indexes the folder to add to the experiment list
#index_tf = input('Do you want to index the folder? (y/n): ')
#if index_tf == 'y':
#    name = input('Enter extra metadata')
#    index_folder(full_path, title=name)
#else:
#    pass    


foldername = 'data/' + foldername
try:
    h5_files = [f for f in os.listdir(foldername) if f.endswith('.h5')]
except:
    print(os.listdir())
    print(os.getcwd())
    print(foldername)
    raise FileNotFoundError('No h5 files in the specified folder')
if h5_files == []:
    print(os.getcwd())
    print(os.listdir())
    print(os.getcwd())
    print(foldername)
    raise FileNotFoundError('No h5 files in the specified folder')



############# FORMATS THE EXPERIMENT NAMES

experiment_names = []
t_w_machine_time = []

with open( foldername + '/data.txt', 'r') as file:
    #print([line for line in file])
    for line in file:
        if len(experiment_names) == 0:
            experiment_names = line.split(',')
            #replace quotes iwht ''ArithmeticError
            
            #replace ',' with ''
            experiment_names = [name.replace("'", "") for name in experiment_names]
            experiment_names = [name.replace(" ", "") for name in experiment_names]
            experiment_names = [name.replace("\n", "") for name in experiment_names]
        else:
            #replace ',' with ''
            
            t_w_machine_time = line.split(',')
            t_w_machine_time = [float(time) for time in t_w_machine_time]
if len(experiment_names) == 0:
    print(experiment_names)
    print(os.getcwd())
    print(foldername + '/data.txt')
    print(os.listdir(foldername))
    raise ValueError('No experiment names found')
if len(t_w_machine_time) == 0:
    print(os.getcwd())
    print(os.listdir(foldername))
    print(foldername + '/data.txt')
    raise ValueError('No experiment names or machine times found')
            

experiment_names = [str(name) for name in experiment_names]

os.chdir(foldername)


############### READS IN THE DATA VALUES
data_dict = {}
t_w = []
converter = 1/409.6
#print(os.listdir())
for file_name in h5_files:
    file = h5py.File(file_name, 'r')
    #In the file name, add the float after Delay_ and before ms to the list of delays
    #delay = float(file_name.split('-RFSoC')[0])
    #match delay to an element of experiment_names
    try:
        for name in experiment_names:
            #print(name)
            if name in file_name:
                #print(name)
                id_index = experiment_names.index(name)
                t_w_current = t_w_machine_time[id_index]*converter*1e3
                break
        #print(file_name)
        t_w.append(t_w_current)
    except:
        print(file_name)
        print(experiment_names)
        raise ValueError('No matching experiment name found')
    #delay = ids[delay]
    
    #t_w.append(delay)
    archive = file['archive']
    dataset = file['datasets']
    x_vals = dataset['data.rfsoc_pulse_sequence.x_values']
    raw_counts = dataset['data.rfsoc_pulse_sequence.raw_counts']
    x_array = np.asarray(x_vals)
    #Convert counts to probabilities
    #plt.plot(x_array)
    counts = np.asarray(raw_counts)
    raw_counts = np.asarray(raw_counts)
    #print((raw_counts.shape))
    thresholded_counts = raw_counts > COUNTS_THRESHOLD
    probs = np.mean(thresholded_counts, axis=1)
    data = {'x': x_array, 'probs': probs}
    data_dict[t_w_current] = data
    
#sort t_w in ascending order
t_w.sort()
#print(t_w)
#print(data_dict.keys())


#For each key, extract Omega and theta from the data by fitting to cetina_thermal_exp and cetina_envelope_exp

qubits = data_dict[t_w[0]]['probs'].shape[0]
print('Qubits used:')
print('Total Qubits: ',qubits)
#Get the center qubit index:
shift = 0
center_qubit_index = int(qubits/2)+shift
good_qubits = np.array(ACTIVE_QUBITS) + center_qubit_index

print('Center Qubit Index (Zero Indexed): ',center_qubit_index)
print([i-(center_qubit_index) for i in range(qubits)])
print('Active Qubits, Zero Index: ',good_qubits)
print('Active Qubits, center Index: ',ACTIVE_QUBITS)

thetas = np.zeros((len(t_w), len(good_qubits)))
Omegas = np.zeros((len(t_w), len(good_qubits)))
#For each element of t_w, and each good qubit, curve fit to the data to get Omega and theta
for i, t in enumerate(t_w):
    data = data_dict[t]
    x = data['x']*us
    probs = data['probs']
    for j, q in enumerate(good_qubits):
        #print(j, q)
        try:
            popt, pcov = curve_fit(cetina_thermal_exp, x, probs[q, :], p0=[theta, omega], maxfev=10000)
        except:
            print('Fit Failed, returning previous value. For Qubit {}. t_w: {}'.format(q, t))
            popt = [theta, omega]
        theta = popt[0]
        try:
            popt, pcov = curve_fit(cetina_envelope_exp, x, probs[q, :], p0=[theta, omega], maxfev=10000)
        except:
            print('Fit Failed, returning previous value. For Qubit {}. t_w: {}'.format(q, t))
            popt = [theta, omega]
        omega = popt[1]
        thetas[i, j] = theta
        Omegas[i, j] = omega
Omega = 1000000



#For calibration: 
Target_Omega_List = np.zeros(23)
Target_Omega_List[good_qubits] = Omegas[0,:]


#Computes the scale factors for each Omega and saves plots for each qubit
cmap = LinearSegmentedColormap.from_list("mycmap", ["blue", "red"])
Omega_scale_factor = np.zeros((len(t_w), len(good_qubits)))
Omegas = np.zeros((len(t_w), len(good_qubits)))
thetas_mat = np.zeros((len(t_w), len(good_qubits)))
for i in range(len(good_qubits)):
    qubit = good_qubits[i]
    fig = plt.figure()
    for Omega_index in range(len(t_w)):
        t_w_data = data_dict[t_w[Omega_index]]
        x = t_w_data['x']*us
        y = t_w_data['probs'][qubit]
        # Get the color for the current plotfga
        color = cmap(Omega_index / len(t_w))
        fit_output = fit(cetina_thermal_exp, x, y, p0=[.01, Omega])

        theta_fit = fit_output[0]
        fit_output = fit(cetina_envelope_exp, x, y, p0=[.01, Omega])

        Omega_fit = fit_output[1]
        
        plt.plot(x, y + Omega_index*2, 'o', color=color, label='Experimental Data') 
        plt.plot(x, cetina_envelope_exp(x, *fit_output) + Omega_index*2, color='k', label='Fit') 
        plt.text(0, Omega_index*2 - 0.5, r'$\theta$ expect: {:.4f}, $\theta$ est.: {:.4f}'.format(0, theta_fit), color=color)
        plt.text(max(x)-4*us, Omega_index*2 - 0.5, r'$\Omega$ = {:.3f}'.format(Omega_fit), color=color)
        Omegas[Omega_index, i] = Omega_fit
        Omega_scale_factor[Omega_index,i] = Target_Omega_List[qubit]/Omega_fit
        thetas_mat[Omega_index,i] = theta_fit
    plt.xlabel('Time (us)')
    plt.ylabel('Probability, shifted')
    plt.title('Qubit {}'.format(qubit))
    plt.ylim(-1, 2*len(t_w))
    plt.savefig('Qubit_{}'.format(qubit))
    #plt.show()
cmap = LinearSegmentedColormap.from_list("mycmap", ["blue", "red"])


#Extracts the heating rate for each qubit
plt.figure()
for i in range(len(good_qubits)):
    plt.plot(t_w, thetas_mat[:, i], label='Qubit {}'.format(good_qubits[i]))
plt.xlabel('Delay (ms)')
plt.ylabel('Theta')
plt.legend()
plt.title('Theta vs. Delay')
#Extract the average of the thetas slope. Fit each qubit to a line and extract the slope, then average
thetas_slopes = np.zeros(len(good_qubits))
intercepts = np.zeros(len(good_qubits))
for i in range(len(good_qubits)):
    popt, pcov = np.polyfit(t_w, thetas_mat[:, i], 1, cov=True)
    thetas_slopes[i] = popt[0]
    intercepts[i] = popt[1]
    print('Qubit {}: Theta slope: {}'.format(good_qubits[i], popt[0]))
thetas_slope = np.mean(thetas_slopes)
intercept = np.mean(intercepts)
plt.plot(t_w, thetas_slope*np.array(t_w) + intercept, label='Average Theta Slope', linestyle='--', lw=3)
#Add text with the average theta slope and intercept:
plt.text(0.3, 0.02, r'Fit for Center Ions: {:.4f}$t_w$ + {:.4f}'.format(thetas_slope, intercept), transform=plt.gca().transAxes)
#save the plot:
#print(os.getcwd())
plt.savefig('Theta_vs_Delay.png', dpi=600)
#plt.show()
plt.close()
#print()



#Compute Updated t_w times:
target_thetas = [1, 1.1, 1.3, 1.6, 1.8, 2, 2.5, 3, 4]
#Using theta_slop and intercept, compute the updated wait times so that target_thetas*thetas_intercept = tw*thetas_slope + thetas_intercept
updated_t_w = (np.array(target_thetas)*intercept - intercept)/thetas_slope
updated_thetas = thetas_slope*updated_t_w + intercept
with open('Updated_Delay.txt', 'w') as f:
    f.write('gamma,\tUpdated_Delay(ms)\n')
    for i in range(len(updated_t_w)):
        f.write('{},\t{}\n'.format(target_thetas[i], updated_t_w[i]))
    f.write('\n')
    f.write('Theta Slope: {}\n'.format(thetas_slope))
    f.write('Intercept: {}\n'.format(intercept))
    
export_info = ''
export_info += 'Updated times in ms:\n'
print('Updated times in ms:')

build_string = '['
for item in range(len(updated_t_w)):
    build_string += '{:.6f}'.format(updated_t_w[item])
    if item != len(updated_t_w)-1:
        build_string += ', '
build_string += ']'
print(build_string)
export_info += build_string + '\n'

print('Updated times in machine time:')
export_info += 'Updated times in machine time:\n'
build_string = '['

for item in range(len(updated_t_w)):
    build_string += '{:.6f}'.format(updated_t_w[item]/1e3 /converter)
    if item != len(updated_t_w)-1:
        build_string += ', '
build_string += ']'
print(build_string)
export_info += build_string + '\n'

#print(print('[' + ", ".join(map(str, updated_t_w/1e3 /converter)) + ']'))
#export_info +='[' + ", ".join(map(str, updated_t_w/1e3 /converter)) + ']\n'





#fit to a line:
def line(x, m, b):
    return x*m  + b
mb_mat = np.zeros((len(good_qubits), 2))
Omega_comp_fit_matrix = np.zeros(Omega_scale_factor.shape)
m_list = []
b_list = []
for qubit_index in range(len(good_qubits)):
    #pick a color from the cmap:
    color = cmap(qubit_index/len(good_qubits))
    plt.plot(thetas_mat[:, qubit_index], Omega_scale_factor[:,qubit_index], label='Qubit {}'.format(i), color=color)
    popt, pcov = curve_fit(line, thetas_mat[:,qubit_index], Omega_scale_factor[:,qubit_index])
    m = popt[0]
    b = popt[1]
    m_list.append(m)
    b_list.append(b)
    
    #b = 1- thetas[0,qubit_index]*m
    #print(m*thetas[:,qubit_index] + b)
    #Compute the expected theta values for each qubit at the new wait times:
    theta_slope_qubit = thetas_slopes[qubit_index]
    theta_intercept_qubit = intercepts[qubit_index]
    new_thetas = theta_slope_qubit*updated_t_w + theta_intercept_qubit
    #Compute the new compensations for each qubit and wait time:
    Omega_comp_fit = line(new_thetas, m, b)
    Omega_comp_fit_matrix[:,qubit_index] = Omega_comp_fit 
    plt.plot(new_thetas, Omega_comp_fit, label='Qubit {} Fit'.format(qubit_index), color=color, linestyle='--')   
popt, pcov = curve_fit(line, thetas_mat.flatten(), Omega_scale_factor.flatten())
#print('collective fit:')
m_collective = popt[0]
b_collective = popt[1]
#print(m_collective)
#print(b_collective)
flatten_theta = thetas_mat.flatten()
flatten_theta.sort()
#print(flatten_theta)
plt.plot(flatten_theta, line(flatten_theta, m_collective, b_collective),  label='Collective Fit', linestyle='--', lw=5)#, label='')
#print('Average:')
#print(np.average(m_list))
average_m = np.average(m_list)
#print(np.average(b_list))
average_b = np.average(b_list)
plt.plot(flatten_theta, line(flatten_theta, average_m, average_b), label='Average Fit', linestyle='--', lw=5)#, label='')
plt.legend()

#Compute the Omega_comp matrix for the updated times:
Omega_comp_fit_matrix = np.zeros(Omega_scale_factor.shape)
for i in range(len(good_qubits)):
    Omega_comp_fit_matrix[:,i] = line(updated_thetas, average_m, average_b)
Omega_old_comp = np.zeros(Omega_scale_factor.shape)
for i in range(len(good_qubits)):
    Omega_old_comp[:,i] = line(np.array(t_w)*thetas_slope + intercept, average_m, average_b)


plt.title('Theta vs. Omega compensation factor')
plt.xlabel('Theta')
plt.ylabel('Omega Compensation Factor')
#Compute Updated Omega_comp from the fits



for i in range(len(t_w)):
    #print('t_w:{:.6f} ms '.format(updated_t_w[i]))
    print('t_w:{:.6f} machine time '.format(updated_t_w[i]*1e-3*409.6))
    #replace each '  ' with ' '
    #print('[' + ", ".join(map(str, Omega_comp_fit_matrix[i])) + ']')
    print(Omega_comp_fit_matrix[i,0])
    export_info +='t_w:{:.6f} ms \n'.format(updated_t_w[i])
    export_info +='t_w:{:.6f} machine time \n'.format(updated_t_w[i]*1e-3*409.6)
    export_info += str(Omega_comp_fit_matrix[i,0]) + '\n'
    #export_info +='[' + ", ".join(map(str, Omega_comp_fit_matrix[i])) + ']\n'
#Save export_info to a .txt file called 'Udpated parameters:'
with open('Updated_parameters.txt', 'w') as f:
    f.write(export_info)

    
    
    
#### OMEGE PLOTS
plt.figure()

plt.imshow(Omega_scale_factor, cmap=cmap)
plt.xlabel('Qubit, indexed from first used qubit {}'.format(ACTIVE_QUBITS[0]+ shift))
plt.ylabel('t_w index')
plt.title('Omega Scaling Factor Update')
cbar = plt.colorbar()
cbar.set_label('Omega Relative Scale Factor')
plt.savefig('Omega_Scale_Factor_raw.png', dpi=600)
print('Actual Thetas:')
print(np.array(t_w)*thetas_slope + intercept)
plt.close()


plt.figure()

plt.imshow(Omega_comp_fit_matrix, cmap=cmap)
plt.xlabel('Qubit, indexed from first used qubit {}'.format(ACTIVE_QUBITS[0]+ shift))
plt.ylabel('t_w index')
plt.title('Omega Scaling Factor Linear Fit')
cbar = plt.colorbar()
cbar.set_label('Omega Relative Scale Factor, Linear Fit')
plt.savefig('Omega_Scale_Factor_fit.png', dpi=600)
print('Updated Thetas')
print(updated_thetas)
plt.close()


plt.figure()

plt.imshow(Omega_comp_fit_matrix/Omega_scale_factor, cmap=cmap)
plt.xlabel('Qubit, indexed from first used qubit {}'.format(ACTIVE_QUBITS[0]+ shift))
plt.ylabel('t_w index')
plt.title('Omega_fit/Omega_scale_raw error')
cbar = plt.colorbar()
cbar.set_label('Omega_fit/Omega_scale_raw error')
plt.savefig('Omega_fitoverOmega_scale_raw error.png', dpi=600)
plt.close()



plt.figure()

plt.imshow(Omegas, cmap=cmap)
plt.xlabel('Qubit, indexed from first used qubit {}'.format(ACTIVE_QUBITS[0]+ shift))
plt.ylabel('t_w index')
plt.title('Omega experimental')
cbar = plt.colorbar()
cbar.set_label('Omega experimental')
plt.savefig('Omega_experimental.png', dpi=600)
plt.close()


plt.figure()

plt.imshow(Omegas*Omega_old_comp, cmap=cmap)
plt.xlabel('Qubit, indexed from first used qubit {}'.format(ACTIVE_QUBITS[0]+ shift))
plt.ylabel('t_w index')
plt.title('Omega experimental')
cbar = plt.colorbar()
cbar.set_label('Omega experimental')
#plt.savefig('Omega_experimental.png', dpi=600)



plt.figure()

plt.imshow(thetas_mat, cmap=cmap)
plt.xlabel('Qubit, indexed from first used qubit {}'.format(ACTIVE_QUBITS[0]+ shift))
plt.ylabel('t_w index')
plt.title('theta experimental')
cbar = plt.colorbar()
cbar.set_label('Omega experimental')
plt.savefig('theta_experimental.png', dpi=600)
plt.close()



def save_plot(qubit_index):
    plt.figure()

    """Make a plot with the shape
    |----|----|----|
    |    |    |    |
    |----|----|----|
    |              |
    |--------------|
    Which is 3 small plots over one long plot. So a 3x2 grid, but the lower plots are merged
    """
    thetas = intercept + np.array(t_w)*thetas_slope
    thetas = thetas_mat[:, qubit_index]
    
    #print('Average:')
    #print(np.average(Omegas[:,qubit_index]))
    average_Omega = np.average(Omegas[:,qubit_index])
    #print(Omegas[:,qubit_index])
    #print('Target:')
    #rint(Target_Omega_List[qubit_index])
    trim = -2
    if trim == 0:
        trim = len(thetas)
    
    thetas = thetas[:trim]
    t_w_temp = t_w[:trim]
    n = len(t_w_temp)
    #print(n)
    for i in range(n):
        t_w_index = t_w[i]
        t_w_data = data_dict[t_w_index]
        times = t_w_data['x']*us
        raw_data = t_w_data['probs'][qubit_index]
    data = np.zeros((n, len(times)))
    for i in range(n):
        t_w_index = t_w[i]
        t_w_data = data_dict[t_w_index]
        times = t_w_data['x']*us
        raw_data = t_w_data['probs'][qubit_index]
        data[i, :] = raw_data
    fig = plt.figure()
    ax2 = fig.add_subplot(332)
    ax1 = fig.add_subplot(331, sharex=ax2)
    ax3 = fig.add_subplot(333, sharex=ax2)
    dense_theta = np.linspace(0, max(thetas), 1000)
    #ax1.set_xlim(min(dense_theta), .15)
    #ax2.yaxis.set_visible(False)
    #ax3.yaxis.set_visible(False)

    ax4 = fig.add_subplot(312)
    #ax1.set_xlim(-0.02, max(thetas)+0.02)
    max_time = max(times)
    marked_times = np.linspace(0, max_time, 4, endpoint=False)[1:]
    marked_times[2] = marked_times[2] + 1e-6
    marked_times[0] = marked_times[0] - 1e-6
    
    index_baseline = 0
    ax4.vlines(marked_times, -4, 4, color='black', linestyle='--')
    for i in range(n):
        if i >= index_baseline:
            pass
        else:
            #go to next iteration
            continue
        color = cmap((i-index_baseline) / (n-index_baseline))
        ax4.plot(times, data[i, :].T, color=color, alpha=0.4)
        ax4.fill_between(times, data[i, :], data[i, :], color=color, alpha=0.2)
    #good_data = generate_experimental_data(times, Target_Omega_List[qubit_index], np.array([0.]), 1000)
    #good_data = 0.5 - (good_data/ 2)
    good_data = cetina_thermal_exp(times, 0, Target_Omega_List[qubit_index])
    ax4.plot(times, good_data, color='black', label='Ideal Data')
    #ax4.plot(times, generate_experimental_data(times, 1, np.array([0.]), 1000), color='black', label='Ideal Data')
    ax4.set_ylim(-0.3, 1.3)



    function_colors = ['purple', 'orange', 'blue', 'brown']
    fit_functions = [linear_ZNE, cubic_ZNE, log_fit_exp_ZNE]
    function_names = ['Linear', 'Cubic', 'Exponential']

    function_colors = ['orange', 'green', 'brown']
    fit_functions = [linear_ZNE, third_no_first_ZNE]
    function_names = ['Linear Extrapolation', 'Cubic']


    axes = [ax1, ax2, ax3]



    for t_i in range(len(marked_times)):
        #Take the data slice at the time:
        axis = axes[t_i]
        #Get the nearest index in time to the marked time:
        time_i = np.argmin(np.abs(times - marked_times[t_i]))
        time = times[time_i]
        #print(time_i)
        vs_theta = data[:,time_i]
        axis.errorbar(thetas[:index_baseline], vs_theta[:index_baseline],  fmt='o', label='Experimental Data')
        #Baseline
        axis.errorbar(thetas[index_baseline], vs_theta[index_baseline], fmt='o', label='Experimental Data')
        if t_i == 0:
            axis.vlines(thetas[index_baseline], min(vs_theta)+0.04, max(vs_theta),color='black', linestyle='--')#, label='Experimental Limit'
            axis.text(thetas[index_baseline]-0.02, min(vs_theta), 'Experimental Limit', color='black')
        #Right of baseline:
        axis.errorbar(thetas[index_baseline+1:], vs_theta[index_baseline+1:], fmt='o', label='Experimental Data')
        axis.scatter(0, good_data[time_i])
        for f_i in range(len(fit_functions)):
            fit_function = fit_functions[f_i]
            fit_line = fit_function(thetas[index_baseline:], vs_theta[index_baseline:])
            axis.plot(dense_theta, fit_line(dense_theta), color=function_colors[f_i], label=function_names[f_i])

        #Plot expected from Cetina:
        #func_cetina = lambda x: 1/np.sqrt(1 + (target_Omega*time*x)**2)*np.cos(target_Omega*time)# - np.arctan(target_Omega*time*x))
        #axis.plot(dense_theta, func_cetina(dense_theta), color='green', label='Cetina')
        axis.set_title('Time: {:.2f}'.format(time))
        axis.set_ylabel('Population Transfer')


    ax2.set_xlabel(r'Measured $\theta$')
    fitted_values = np.zeros((len(times), len(fit_functions)))
    for t_i in range(len(times)):
        time_slice = data[:,t_i]
        for f_i in range(len(fit_functions)):
            fit_function = fit_functions[f_i]
            fit_line = fit_function(thetas[index_baseline:], time_slice[index_baseline:])
            fitted_values[t_i, f_i] = fit_line(0)
            #ax4.plot(dense_theta, fit_line(dense_theta), color=function_colors[f_i], label=function_names[f_i])

    for f_i in range(len(fit_functions)):
        ax4.plot(times, fitted_values[:,f_i], color=function_colors[f_i], label=function_names[f_i])

    #ax4.plot(times, full_ZNE_fancy_exp, color='brown', label='Fancy Exponential Fit')
    fig.set_size_inches(10,8)
    fig.suptitle(r'ZNE for Frequency Corrected Rabi Flopping, $\Omega = 1$')
    fig.tight_layout()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(thetas)))
    sm.set_array([])
    #Add a label to the color bar:

    cbar = fig.colorbar(sm, ax=ax4)

    cbar.set_label(r'$\theta$')


    # save = input('Save Figure? (y/n)')
    # if save == 'y':
    #     metadata = input('Please provide some metadata for the figure:')
    #     #Strip spaces and illegal save characters:
    #     metadata = metadata.replace(' ', '_')
    #     metadata = metadata.replace('/', '_')
    #     metadata = metadata.replace('\\', '_')
    #     plt.savefig('ZNE_Fit_Rabi_data_{}.png'.format(metadata), dpi = 600)
        #plt.savefig('ZNE_Fit_Rabi_data.png', dpi = 600)
    #Add the color bar to axis 4:
    #fig.colorbar()
    plt.savefig('Extrapolation_Test_Qubit_{}'.format(qubit_index), dpi=600)
    #
    plt.close()



    #ax4 = fig.add_subplot(313)

for qubit_index in range(len(good_qubits)):
    save_plot(qubit_index)