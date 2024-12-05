"""This file generates simulation for the Jij data by attempting to match the frequency information and decay rates. 

Currently:
Given a collection of data and temperatures:
1. For each, extracts the post selected decay rate (depolarizing) leakage subspace. Attempts to match the leakage using:
   a. SPAM error calibrated to the t=0 point
   b. Jij ramp error calibrated at the intercept of the slope which is NOT including t=0
   c. Using the input temperatures, performs shot-to-shot simulations to match the thermal noise effects. 
"""
from collections.abc import Iterable
import h5py
import numpy as np


class JijSim:
    def __init__(self):
        self.folder = None
        self.file_list = None
        self.qubits = None
        self.data_prefix = None
    
    def set_prefix(self, prefix):
        self.data_prefix = prefix
        if self.set_file_list is not None:
            if not prefix in self.file_list[0]:
                self.file_list = [prefix + file for file in self.file_list]
            
            #raise ValueError("The prefix does not match the first file in the list.")
            
    def set_folder(self, folder):
        self.folder = folder
    
    def set_file_list(self, file_list):
        assert isinstance(file_list, Iterable), "file_list must be an iterable (e.g. list, tuple)"
        self.file_list = file_list
        if self.data_prefix is not None:
            if not self.data_prefix in file_list[0]:
                self.file_list = [self.data_prefix + file for file in file_list]
        
    
    def set_qubits(self, qubits):
        assert isinstance(qubits, Iterable), "qubits must be an iterable (e.g. list, tuple)"
        assert len(qubits) == 2, "qubits must be a pair"
        self.qubits = qubits
    
    def process(self):
        for current_file in self.file_list:
            self.process_file(current_file)
    
    def process_file(self, file):
        file = h5py.File(self.folder + file + '-RFSoCSequence.h5', 'r')
        archive = file['archive']
        dataset = file['datasets']
        x_vals = dataset['data.rfsoc_pulse_sequence.x_values']
        raw_counts = dataset['data.rfsoc_pulse_sequence.raw_counts']
        counts = np.asarray(raw_counts)
        #Number of qubits to return, # of shots, # of time steps
        measurements = np.zeros((2, counts.shape[1], counts.shape[2]))
        
        shot_number = counts.shape[1]
        time_steps = counts.shape[2]
        leakage_matrix = np.zeros((time_steps))
        measurements[self.qubits[0],:,:] = 1 < counts[self.qubits[0],:,:]
        measurements[self.qubits[1],:,:] = 1 < counts[self.qubits[1],:,:]
        #Convert the measurements to bitstrings. Convert to string, then append the like indices
        counts_bitstrings = np.zeros((shot_number, time_steps), dtype='<U5')
        expectations = np.zeros((shot_number, time_steps))
        for i in range(150):
            for j in range(time_steps):
                expectation = ((measurements[0,i,j]*2-1) - (measurements[1,i,j]*2-1))/2
                bitstring = str(int(measurements[0,i,j])) + str(int(measurements[1,i,j]))
                #print(bitstring)
                counts_bitstrings[i,j] = bitstring
                expectations[i,j] = expectation
        #average expectations across the 150 axis:
        expectations_avg = np.mean(expectations, axis=0)
        #Convert counts_bitstrings to a dictionary of counts
        counts_array  = np.zeros((shot_number,), dtype=dict)
        for i in range(time_steps):
            counts_array[i] = {}
            for j in range(shot_number):
                if counts_bitstrings[j,i] in counts_array[i]:
                    counts_array[i][counts_bitstrings[j,i]] += 1
                else:
                    counts_array[i][counts_bitstrings[j,i]] = 1
                    
        leakage = np.zeros((time_steps,))
        for i in range(time_steps):
            if '00' in counts_array[i]:
                leakage[i] = counts_array[i]['00']/shot_number
            if '11' in counts_array[i]:
                leakage[i] += counts_array[i]['11']/shot_number
        leakage_conv = np.zeros((time_steps,))
        for i in range(time_steps):
            leakage_conv[i] = np.mean(leakage[i:i+3])
        leakage_matrix[:,file_id_index] = leakage#leakage_conv
        flopping_matrix[:,file_id_index] = expectations_avg
        flopping_std = np.std(expectations, axis=0)
        flopping_std_matrix[:,file_id_index] = flopping_std
    

        