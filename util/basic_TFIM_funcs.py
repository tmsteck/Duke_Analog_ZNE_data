
import qutip as qt
from qutip import (about, basis, expect, mesolve, qeye, sigmax, sigmay, sigmaz, tensor, mcsolve, sesolve, mesolve)
from qutip.measurement import measure_observable, measurement_statistics
from qutip.expect import expect
#import parallel:
from joblib import Parallel, delayed


from util.samplers_util import thermal_rejection

import numpy as np
import scipy


def get_s_lists(N):
    """Returns the tensor product Pauli matricies sx, sy, sz for a system of N spins"""
    sx_list, sy_list, sz_list = [], [], []
    for i in range(N):
        op_list = [qeye(2)] * N
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list))
        op_list[i] = sigmay()
        sy_list.append(tensor(op_list))
        op_list[i] = sigmaz()
        sz_list.append(tensor(op_list))
    return sx_list, sy_list, sz_list

def get_middle_block(matrix, N):
    """Returns the center of an array/Matrix"""
    M = matrix.shape[0]  # Assuming the matrix is square (MxM)

    # Calculate the starting indices for the middle block
    start_idx = (M - N) // 2

    # Calculate the ending indices for the middle block
    end_idx = start_idx + N

    # Return the middle block of the matrix
    return matrix[start_idx:end_idx, start_idx:end_idx]


def get_H_generator(Jij_mat, h, options = None):
    try:
        COM_participation_vector = options['COM_participation']
    except:
        COM_participation_vector = np.ones(Jij_mat.shape[0])
    try:
        theta = options['theta']
    except:
        theta = 0
    try:
        baseline_coupling = options['baseline_coupling']
    except:
        baseline_coupling = 280/0.15
    try:
        Omega_compensation = options['Omega_compensation']
    except:
        Omega_compensation = np.ones(Jij_mat.shape[0])
    try:
        axis = options['axis']
    except:
        axis = 'zz'
    

    N = Jij_mat.shape[0]
    sx_list, sy_list, sz_list = get_s_lists(N)
    if axis == 'zz':
        IM_list_to_use  = sz_list
        TF_list_to_use = sx_list
    if axis == 'xx':
        IM_list_to_use = sx_list
        TF_list_to_use = sz_list
    if axis == 'yy':
        XY = True
        IM_list_to_use = sx_list
        IM_list_to_use_2 = sy_list
        TF_list_to_use = sz_list
    else:
        XY = False
        IM_list_to_use_2 = sy_list
    
    #print(len(sx_list))
    def genH():
        H = 0
        delta = thermal_rejection(theta, 1)
        #print(delta)
        for i in range(N):
            for j in range(i + 1, N):
                #print(abs(delta*COM_participation_vector[i]))
                if XY:
                    matrix_element = Jij_mat[i, j]* (1 - delta*COM_participation_vector[i])*(1 - delta*COM_participation_vector[j]) * Omega_compensation*Omega_compensation
                    H += matrix_element[0] * (IM_list_to_use[i] * IM_list_to_use[j])
                    H += matrix_element[0] * (IM_list_to_use_2[i] * IM_list_to_use_2[j])
                else:
                    matrix_element = Jij_mat[i, j]* (1 - delta*COM_participation_vector[i])*(1 - delta*COM_participation_vector[j]) * Omega_compensation*Omega_compensation
                    H += matrix_element[0] * (IM_list_to_use[i] * IM_list_to_use[j])
        for i in range(N):
            if not XY:
                H += h * TF_list_to_use[i]
        return H
    return genH

def spin_up_state(N):
    """Returns the state with all spins up"""
    return tensor([basis(2, 0)] * (N))

def spin_down_state(N):
    """Returns the state with all spins up"""
    return tensor([basis(2, 1)] * (N))

def plus_x_state(N):
    """Returns the state with all spins up"""
    plus = 1/np.sqrt(2)*(basis(2, 0) + basis(2,1))
    return tensor([plus] * (N))

def minus_x_state(N):
    """Returns the state with all spins up"""
    minus = 1/np.sqrt(2)*(basis(2, 0) - basis(2,1))
    return tensor([minus] * (N))


def localized_1_state(N):
    """Returns the state with the first spin up"""
    assert N % 2, "N must be odd"
    indexing = N//2
    return_psi = tensor([basis(2, 0)] * (indexing) + [basis(2, 1)] + [basis(2, 0)] * (indexing))
    return return_psi
    #return tensor([basis(2, 0)] + [basis(2, 1)] * (N - 1))
def localized_X_state(N):
    """Returns the state with the first spin up"""
    assert N % 2, "N must be odd"
    indexing = N//2
    plus = 1/np.sqrt(2)*(basis(2, 0) + basis(2,1))
    minus = 1/np.sqrt(2)*(basis(2,0)- basis(2,1))
    return_psi = tensor([plus] * (indexing) + [minus] + [plus] * (indexing))
    return return_psi
    #return tensor([basis(2, 0)] + [basis(2, 1)] * (N - 1))
def return_measurement_op_spin(N):
    """Returns the measurement operator for the spin"""
    sx, sy, sz = get_s_lists(N)
    return sum(sz[i] for i in range(N))

def mixed_zz(N):
    assert N % 2, "N must be odd"
    indexing = N//2
    plus = 1/np.sqrt(2)*(basis(2, 0) + basis(2,1))
    minus = 1/np.sqrt(2)*(basis(2,0)- basis(2,1))
    return_psi = tensor([plus] * (indexing) + [basis(2,0)] + [plus] * (indexing))
    return return_psi

def mixed_xx(N):
    assert N % 2, "N must be odd"
    indexing = N//2
    plus = 1/np.sqrt(2)*(basis(2, 0) + basis(2,1))
    minus = 1/np.sqrt(2)*(basis(2,0)- basis(2,1))
    zero = basis(2,0)
    one = basis(2,1)
    #return_psi = tensor([one] * (indexing) + [minus] + [one] * (indexing))
    return_psi = tensor([zero] * (indexing) + [plus] + [zero] * (indexing))
    return return_psi

def generate_evolution_data_localized(Jij,N,h, t_list, variance, shots, options=None):
    """Generates the evolution data for the spin system. Assumes pre-truncated Jij matrix"""
    assert N % 2, "N must be odd"
    #psi0 = localized_1_state(N)
    
    #psi0 = localized_X_state(N)
    sx, sy, sz = get_s_lists(N)
    if options['axis'] == 'zz':
        psi0 = mixed_zz(N)
        measurement_op = sz[N//2]
    elif options['axis'] == 'xx':
        psi0 = mixed_xx(N)
        #psi0 = mixed_zz(N)
        measurement_op = sx[N//2]
        #measurement_op = sz[N//2]
    if variance == 0:
        Hgen = get_H_generator(Jij, h, options=options)
        H = Hgen()
        result = (sesolve(H, psi0, t_list, [measurement_op]).expect[0], np.zeros((len(t_list) )))
    else:
        H_gen = get_H_generator(Jij, h, options=options)
        #psi0 = spin_up_state(N)
        results_matrix = np.zeros((len(t_list), shots ))
        def run_shot():
            H = H_gen()
            return sesolve(H, psi0, t_list, [measurement_op]).expect[0]
        results_matrix = Parallel(n_jobs=6)(delayed(run_shot)() for _ in range(shots))
        results_matrix = np.array(results_matrix)
        result = (np.mean(results_matrix, axis=0), np.std(results_matrix, axis=0))
        #result = 
    return result

def plus_z_state(N):
    """Returns the state with all spins up"""
    return tensor([basis(2, 0)] * (N))
def minus_z_state(N):
    """Returns the state with all spins up"""
    return tensor([basis(2, 1)] * (N))

def demoPlotJ():
    N = 2
    psi0 = spin_up_state(N)
    sx, sy, sz = get_s_lists(N)
    measurement = sz[0]
    H = sx[0]*sx[1]
    t_list = np.linspace(0, 10, 100)
    result = sesolve(H, psi0, t_list, [measurement]).expect[0]
    return t_list, result

def demoPloth(h):
    N = 1
    psi0 = plus_x_state(N)
    sx, sy, sz = get_s_lists(N)
    measurement = sx[0]
    H = sz[0]*h
    t_list = np.linspace(0, 10, 100)
    result = sesolve(H, psi0, t_list, [measurement]).expect[0]
    return t_list, result

#@jit(parallel=True)
def generate_evolution_data(Jij, N, h, t_list, variance, shots, options=None):
    """Generates the evolution data for the spin system. Assumes pre-truncated Jij matrix"""
    psi0 = localized_1_state(N)
    if variance == 0:
        Hgen = get_H_generator(Jij, h, options=options)
        H = Hgen()
        #psi0 = spin_up_state(N)
        #print(psi0.isherm)
        #print(psi0.shape)
        measurement_op = return_measurement_op_spin(N)
        #measurement_op
        #measurement_op = return_measurement_op_spin(N)
        result = (sesolve(H, psi0, t_list, [measurement_op]).expect[0], np.zeros((len(t_list) )))
    else:
        H_gen = get_H_generator(Jij, h, options=options)
        #psi0 = spin_up_state(N)
        measurement_op = return_measurement_op_spin(N)
        results_matrix = np.zeros((len(t_list), shots ))
        def run_shot():
            H = H_gen()
            return sesolve(H, psi0, t_list, [measurement_op]).expect[0]
        results_matrix = Parallel(n_jobs=6)(delayed(run_shot)() for _ in range(shots))
        results_matrix = np.array(results_matrix)
        result = (np.mean(results_matrix, axis=0), np.std(results_matrix, axis=0))
        #result = 
    return result


def generate_evolution_data_phase_transition(Jij, N, h, t_list, variance, shots, options={}):
    """Generates the evolution data for the spin system. Assumes pre-truncated Jij matrix"""
    
    try:
        axis = options['axis']
    except:
        axis = 'zz'
    #options['axis'] = 'xx'
    if axis == 'zz':
        psi0 = spin_up_state(N)
    elif axis == 'xx':
        psi0 = plus_x_state(N)
    #'+-+'
    plus = 1/np.sqrt(2)*(basis(2, 0) + basis(2,1))
    minus = 1/np.sqrt(2)*(basis(2,0)- basis(2,1))
    psi0 = tensor([plus, minus, plus])
    #measurement_op 
    measurement_op = NN_measures(N, axis=axis)
    #print(psi0)

    if variance == 0:
        Hgen = get_H_generator(Jij, h, options=options)
        H = Hgen()
        #print(H)
        result = (sesolve(H, psi0, t_list, measurement_op).expect, np.zeros((len(t_list) )))
    else:
        H_gen = get_H_generator(Jij, h, options=options)
        results_matrix = np.zeros((len(t_list), shots ))
        def run_shot():
            H = H_gen()
            return sesolve(H, psi0, t_list, measurement_op).expect
        results_matrix = Parallel(n_jobs=6)(delayed(run_shot)() for _ in range(shots))
        results_matrix = np.array(results_matrix)
        result = (np.mean(results_matrix, axis=0), np.std(results_matrix, axis=0))
    return result


def generate_evolution_data_NN(Jij, N, h, t_list, variance, shots, options=None):
    """Generates the evolution data for the spin system. Assumes pre-truncated Jij matrix"""
    psi0 = spin_up_state(N)
    measurement_op = NN_measures(N)
    sx, sy, sz = get_s_lists(N)
    measurement_op = [sx[0] * sy[1]]

    if variance == 0:
        Hgen = get_H_generator(Jij, h, options=options)
        H = Hgen()
        result = (sesolve(H, psi0, t_list, measurement_op).expect, np.zeros((len(measurement_op),len(t_list) )))
    else:
        H_gen = get_H_generator(Jij, h, options=options)
        results_matrix = np.zeros((len(t_list),len(measurement_op), shots ))
        def run_shot():
            H = H_gen()
            return sesolve(H, psi0, t_list, measurement_op).expect
        results_matrix = Parallel(n_jobs=6)(delayed(run_shot)() for _ in range(shots))
        results_matrix = np.array(results_matrix)
        result = (np.mean(results_matrix, axis=0), np.std(results_matrix, axis=0))
    return result



def generate_evolution_data_GHZ(Jij, N, h, t_list, variance, shots, options=None):
    """Generates the evolution data for the spin system. Assumes pre-truncated Jij matrix"""
    psi0 = plus_x_state(N)

    X_GHZ_obs = lambda t: plus_x_state(N)  + 1j*minus_x_state(N) * np.exp(1j*t*h)
    X_GHZ = plus_x_state(N) + 1j*minus_x_state(N)
    X_GHZ = X_GHZ*1/np.sqrt(2)
    #print(X_GHZ)
    outer_GHZ = X_GHZ * X_GHZ.dag()
    measurement_op = [outer_GHZ]

    if variance == 0:
        Hgen = get_H_generator(Jij, h, options=options)
        H = Hgen()
        #print(H)
        #states = sesolve(H, psi0, t_list, [])
        #result = np.array([expect( X_GHZ_obs(t_list[i]), states.states[i])[0] for i in range(len(t_list))])
        #result = (result, np.zeros((len(result), len(t_list) )))
        result = (sesolve(H, psi0, t_list, measurement_op).expect[0], np.zeros((len(measurement_op),len(t_list) )))
    else:
        H_gen = get_H_generator(Jij, h, options=options)
        results_matrix = np.zeros((len(t_list),1, shots ))
        def run_shot():
            H = H_gen()
            result = sesolve(H, psi0, t_list, measurement_op)
            #print(states.states[0])
            #print(X_GHZ_obs(t_list[0]))
            #result = np.array([expect(X_GHZ_obs(t_list[i]), states.states[i])[0] for i in range(len(t_list))])
            #print(result)
            return result.expect[0]
        results_matrix = Parallel(n_jobs=6)(delayed(run_shot)() for _ in range(shots))
        results_matrix = np.array(results_matrix)
        result = (np.mean(results_matrix, axis=0), np.std(results_matrix, axis=0))
    return result

def generate_evolution_data_GHZ_parity(Jij, N, h, t_list, variance, shots, phase=0, options=None):
    """Generates the evolution data for the spin system. Assumes pre-truncated Jij matrix"""
    psi0 = plus_x_state(N)
    sx, sy, sz = get_s_lists(N)
    Sy = tensor([sigmay()]*N)
    #Sx = sum([sx[i] for i in range(N)]) * -1j/2*phase #* np.pi
    measure_observable = [Sy]
    Sz_oscillation = sum([sz[i] for i in range(N)]) * -1j*phase/2
    Rz_oscillation = Sz_oscillation.expm()

        
    Sy_rotation = sum([sy[i] for i in range(N)])*1j*np.pi/4
    Ry = Sy_rotation.expm()
    
    tfim = False
    if options['axis'] == 'xx':
        #z axis rotation:
        Sz = lambda t: sum([sz[i] for i in range(N)]) * 1j*t*h
        Rz = lambda t: Sz(t).expm()
        rotation_to_apply = lambda t: Rz_oscillation*Ry*Rz(t)
        tfim = True
    
    #### NOISY
    try:
        noise_rate = options['noise']
        noise = True
    except KeyError:
        noise = False
        noise_rate = 0
    #if options['noise']
    noisy_ops = [sz[i]*np.sqrt(noise_rate) for i in range(N)]
    #noisy_ops = np.array(sz)*noise_rate
    #ODE_options = qt.solver.Options()
    #ODE_options.ntraj = 1
    #Rx = Sx.expm()
    #Also need to extract all down or all up in the x basis:
    if variance == 0:
        Hgen = get_H_generator(Jij, h, options=options)
        H = Hgen()
        #print(H)
        if noise:
            states = mesolve(H, psi0, t_list,noisy_ops, []).states#njtraj=1
        else:
            states = sesolve(H, psi0, t_list, []).states
        states_copy = states.copy()
        for i in range(len(t_list)):
            states_copy[i] = rotation_to_apply(t_list[i])*states[i]*rotation_to_apply(t_list[i]).dag()
        result = np.array([expect(measure_observable, states_copy[i])[0] for i in range(len(t_list))])
        result = (result, np.zeros((len(result), len(t_list)))) 
    else:
        H_gen = get_H_generator(Jij, h, options=options)
        results_matrix = np.zeros((len(t_list),1, shots ))
        def run_shot():
            H = H_gen()
            if noise:
                states = mesolve(H, psi0, t_list, noisy_ops,[]).states#######
            else:
                states = sesolve(H, psi0, t_list, []).states
            states_copy = states.copy()
            for i in range(len(t_list)):
                states_copy[i] = rotation_to_apply(t_list[i])*states[i]*rotation_to_apply(t_list[i]).dag()
            result = np.array([expect(measure_observable, states_copy[i])[0] for i in range(len(t_list))])
            return result
        results_matrix = Parallel(n_jobs=6)(delayed(run_shot)() for _ in range(shots))
        results_matrix = np.array(results_matrix)
        result = (np.mean(results_matrix, axis=0), np.std(results_matrix, axis=0))
    return result

def generate_evolution_data_GHZ_up_sector(Jij, N, h, t_list, variance, shots, phase=0, options=None):
    """Generates the evolution data for the spin system. Assumes pre-truncated Jij matrix"""
    psi0 = plus_x_state(N)
    #upN = plus_x_state(N) * plus_x_state(N).dag()
    #downN = minus_x_state(N) * minus_x_state(N).dag()
    upN = plus_z_state(N) * plus_z_state(N).dag()
    downN = minus_z_state(N) * minus_z_state(N).dag()
    measure_observable = [upN, downN]#, minus_x_state(N)*minus_x_state(N).dag()]
    #print(measure_observable)
    sx, sy, sz = get_s_lists(N)
    #Rthetaphi = lambda t: (sum([sx[i]*np.cos(h*t/2) + sy[i]*np.sin(h*t/2) for i in range(N)]) * -1j*np.pi/4*N).expm()
    #Sz  = sum([sz[i] for i in range(N)]) * 1j*h #* np.pi
    #Rz = lambda t: (Sz * t).expm()
    Sy = sum([sy[i] for i in range(N)])*1j*np.pi/4
    Ry = Sy.expm()
    #plus = 1/np.sqrt(2)*(basis(2, 0) + basis(2,1))
    #plusplus = tensor([plus] * N)
    #print(Ry*plusplus)
    #print(Sy)
    #print(Ry)
    #full_Sx = [sum([sx[i] for i in range(N)]) ]
    #full_Sy = [sum([sy[i] for i in range(N)])]

    #### NOISY
    tfim = False
    if options['axis'] == 'xx':
        #z axis rotation:
        Sz = lambda t: sum([sz[i] for i in range(N)]) * 1j*t*h
        Rz = lambda t: Sz(t).expm()
        rotation_to_apply = lambda t: Ry*Rz(t)
        tfim = True
    try:
        noise_rate = options['noise']
        noise = True
        if abs(noise_rate - 1e-6) < 0:
            noise = False
    except KeyError:
        noise = False
        noise_rate = 0
    #if options['noise']
    noisy_ops = [sz[i]*np.sqrt(noise_rate) for i in range(N) ]

    if variance == 0:
        Hgen = get_H_generator(Jij, h, options=options)
        H = Hgen()
        #print(H)
        if noise:
            states = mesolve(H, psi0, t_list,noisy_ops, []).states#njtraj=1
        else:
            states = sesolve(H, psi0, t_list, []).states
        states_copy = states.copy()
        for i in range(len(t_list)):
            if tfim: 
                #print('TFIM in the loop')
                #states_copy[i] = rotation_to_apply(t_list[i])*states[i]
                #print(states[i])
                states_copy[i] = rotation_to_apply(t_list[i])*states[i]*rotation_to_apply(t_list[i]).dag()
                #print('copy')
                #print(states_copy[i].dag()*states_copy[i] )
                #print('state')
                #print(states[i].dag()*states[i] )
                pass
            else:
                pass
        result = np.array([sum(expect(measure_observable, states_copy[i])) for i in range(len(t_list))])
        result = (result, np.zeros((len(result), len(t_list))))
    else:
        H_gen = get_H_generator(Jij, h, options=options)
        results_matrix = np.zeros((len(t_list),1, shots ))
        def run_shot():
            H = H_gen()
            if noise:
                states = mesolve(H, psi0, t_list, noisy_ops,[]).states#######
            else:
                states = sesolve(H, psi0, t_list, []).states
            states_copy = states.copy()
            for i in range(len(t_list)):
                #states_copy[i] = Rx*states[i]
                if tfim:
                    states_copy[i] = rotation_to_apply(t_list[i])*states[i]*rotation_to_apply(t_list[i]).dag()
                else:
                    pass
            result = np.array([sum(expect(measure_observable, states_copy[i])) for i in range(len(t_list))])
            return result
        results_matrix = Parallel(n_jobs=6)(delayed(run_shot)() for _ in range(shots))
        results_matrix = np.array(results_matrix)
        result = (np.mean(results_matrix, axis=0), np.std(results_matrix, axis=0))
    return result

def generate_evolution_data_GHZ_up_sector_temp_old(Jij, N, h, t_list, variance, shots, phase=0, options=None):
    """Generates the evolution data for the spin system. Assumes pre-truncated Jij matrix"""
    psi0 = plus_x_state(N)
    upN = plus_x_state(N) * plus_x_state(N).dag()
    downN = minus_x_state(N) * minus_x_state(N).dag()
    measure_observable = lambda t:[upN, downN]#, minus_x_state(N)*minus_x_state(N).dag()]
    sx, sy, sz = get_s_lists(N)
    Sy = sum([sy[i] for i in range(N)])*1j*np.pi/4 ### CHECK THIS
    Rypi2 = Sy.expm()
    #Rypi2 
     #### NOISY
     
    if options['axis'] == 'xx':
        #z axis rotation:
        Sz = lambda t: sum([sz[i] for i in range(N)]) * 1j*t*h
        Rz = lambda t: Sz(t).expm()
        #measure_observable = lambda t: [Rz(-t) *Rypi2* upN *Rypi2.dag()* Rz(t), Rz(-t) *Rypi2* downN *Rypi2.dag()* Rz(t)]
    try:
        noise_rate = options['noise']
        noise = True
    except KeyError:
        noise = False
        noise_rate = 0
    #if options['noise']
    noisy_ops = [sz[i]*np.sqrt(noise_rate) for i in range(N) ]

    if variance == 0:
        Hgen = get_H_generator(Jij, h, options=options)
        H = Hgen()
        if noise:
            result = sum(mcsolve(H, psi0, t_list,noisy_ops, measure_observable(0), ntraj=shots).expect)
        else:
            result = sum(sesolve(H, psi0, t_list, measure_observable(0)).expect)
        result = (result, np.zeros((len(result), len(t_list))))
    else:
        H_gen = get_H_generator(Jij, h, options=options)
        results_matrix = np.zeros((len(t_list),1, shots ))
        def run_shot():
            H = H_gen()
            
            if noise:
                result = sum(mcsolve(H, psi0, t_list,noisy_ops, measure_observable(0), ntraj=1).expect)
            else:
                result = sum(sesolve(H, psi0, t_list, measure_observable(0)).expect)
            return result
            #result = sum(sesolve(H, psi0, t_list, measure_observable).expect)
            #return result
        results_matrix = Parallel(n_jobs=6)(delayed(run_shot)() for _ in range(shots))
        results_matrix = np.array(results_matrix)
        result = (np.mean(results_matrix, axis=0), np.std(results_matrix, axis=0))
    return result

def generate_evolution_data_GHZ_TFIM(Jij, N, h, t_list, variance, shots, options=None):
    """Generates the evolution data for the spin system. Assumes pre-truncated Jij matrix"""
    psi0 = plus_x_state(N)


    def X_GHZ_obs(t):
        ket = plus_x_state(N)  + 1j*minus_x_state(N) #* np.exp(1j*t*h/2)
        return [1/2 * (ket * ket.dag())]
    

    #X_GHZ_obs = lambda t: plus_x_state(N)  + 1j*minus_x_state(N) * np.exp(1j*t*h)
    #X_GHZ = plus_x_state(N) + 1j*minus_x_state(N)
    #X_GHZ = X_GHZ*1/np.sqrt(2)
    #print(X_GHZ)
    #outer_GHZ = X_GHZ * X_GHZ.dag()
    #measurement_op = [outer_GHZ]

    if variance == 0:
        Hgen = get_H_generator(Jij, h, options=options)
        H = Hgen()
        #print(H)
        states = sesolve(H, psi0, t_list, [])
        result = np.array([expect( X_GHZ_obs(t_list[i]), states.states[i])[0] for i in range(len(t_list))])
        result = (result, np.zeros((len(result), len(t_list) )))
        #result = (sesolve(H, psi0, t_list, measurement_op).expect, np.zeros((len(measurement_op),len(t_list) )))
    else:
        H_gen = get_H_generator(Jij, h, options=options)
        results_matrix = np.zeros((len(t_list),1, shots ))
        def run_shot():
            H = H_gen()
            states = sesolve(H, psi0, t_list, [])
            #print(states.states[0])
            #print(X_GHZ_obs(t_list[0]))
            result = np.array([expect(X_GHZ_obs(t_list[i]), states.states[i])[0] for i in range(len(t_list))])
            #print(result)
            return result
        results_matrix = Parallel(n_jobs=-1)(delayed(run_shot)() for _ in range(shots))
        results_matrix = np.array(results_matrix)
        result = (np.mean(results_matrix, axis=0), np.std(results_matrix, axis=0))
    return result

def generate_evolution_data_full_correlators(Jij, N, h, t_list, variance, shots, options=None):
    """Generates the evolution data for the spin system. Assumes pre-truncated Jij matrix"""
    if variance == 0:
        Hgen = get_H_generator(Jij, h, options=options)
        H = Hgen()
        psi0 = spin_up_state(N)
        #print(psi0.isherm)
        #print(psi0.shape)
        measurement_op = NN_measures(N)
        print(len(measurement_op))
        #measurement_op = return_measurement_op_spin(N)
        result = (sesolve(H, psi0, t_list, measurement_op).expect, np.zeros((len(measurement_op),len(t_list) )))
    else:
        H_gen = get_H_generator(Jij, h, options=options)
        psi0 = spin_up_state(N)
        measurement_op = NN_measures(N)
        results_matrix = np.zeros((len(t_list),len(measurement_op), shots ))
        print(len(measurement_op))
        def run_shot():
            H = H_gen()
            return sesolve(H, psi0, t_list, measurement_op).expect
        results_matrix = Parallel(n_jobs=6)(delayed(run_shot)() for _ in range(shots))
        results_matrix = np.array(results_matrix)
        result = (np.mean(results_matrix, axis=0), np.std(results_matrix, axis=0))
        #result = 
    return result

def NN_measures(N, axis='zz'):
    sx, sy, sz = get_s_lists(N)
    if axis == 'zz':
        list_to_use = sz
    elif axis == 'xx':
        list_to_use = sx
    measurements_two_point_NN = [list_to_use[i] * list_to_use[i+1] for i in range(N-1)]
    return list_to_use + measurements_two_point_NN

def full_measures(N):
    sx, sy, sz = get_s_lists(N)
    measurements_two_point = [((i!=j) + 1)*sz[i] * sz[j] for i in range(N) for j in range(i, N)]
    return sz + measurements_two_point

def NNcorrelator(measurement_matrix, times, variances, size):
    """Assumes formatting as size + size-1, where size-1 is the number of two point measures
    """
    single_point = measurement_matrix[:size]
    two_point = measurement_matrix[size:]
    C = np.zeros((size-1,len(times), len(variances)))
    for i in range(size-1):
        print((two_point[i] - single_point[i]*single_point[i+1]).shape)
        C[i] = two_point[i] - single_point[i]*single_point[i+1]
    return np.average(C, axis=0)

def full_correlators_seperation(measurement_matrix, times, variances, size):
    """Assumes formatting as size + size-1, where size-1 is the number of two point measures
    """
    single_point = measurement_matrix[:size]
    two_point = measurement_matrix[size:]
    C = np.zeros((size-1,len(times), len(variances)))
    for i in range(size-1):
        C[i] = two_point[i]
    return np.average(C, axis=0)