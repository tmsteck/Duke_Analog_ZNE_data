# import all the useful lib in python
import h5py as h5
from matplotlib import pyplot as plt
from matplotlib import artist as art


import numpy as np
import scipy as sp
# from sklearn.cluster import AgglomerativeClustering as AgCluster

#import pandas as pd
import pathlib
import os

import datetime as dt
import time

from IPython.display import clear_output

import scipy as sp
import itertools



# define utility functions

def gen_bit_sequence(ind,N):
    bit_seq = []

    if N==0:
        if ind != N:
            return['0','1']
        else:
            return['1']

    sub_seq = gen_bit_sequence(ind,N-1)
    for seq in sub_seq:
        if ind != N:
            bit_seq.append(seq+'0')
        bit_seq.append(seq+'1')
    return bit_seq

def get_join_prob(file_path,ion_ind):
    # to do, convert this to a n ion out of N ion join prob
    file = h5.File(file_path)

    tmp = [k for k in file["datasets"].keys()]

    data_name = dict([(x.split(".")[-1],x) for x in tmp])
    y_name = data_name["raw_counts"]
    x_name = data_name["x_values"]

    data = file["datasets"][y_name][...]
    x_vals = file["datasets"][x_name][...]

    n_pmt,n_shots,n_scan = data.shape
    ind = np.argsort(x_vals)
    x_vals = x_vals[ind]


    state = data>1
    state = state.astype(int)

    bit_seq_all = gen_bit_sequence(len(ion_ind)+1,len(ion_ind)-1)
    prob_join = {}

    for seq in bit_seq_all:
        prob_join[seq] = 0

    prob_join = dict(map(lambda x:(x,np.zeros((n_scan))),bit_seq_all))

    for i_scan in range(n_scan):

        for i_shot in range(n_shots):
            current_seq = ''.join(state[:,i_shot,i_scan].astype(str))
            tmp_seq = [current_seq[ion] for ion in ion_ind]
            current_seq = ''.join(tmp_seq)
            prob_join[current_seq][i_scan] += 1


    prob_join = dict(map(lambda x:(x[0],x[1][ind]/n_shots),prob_join.items()))
    prob_join_error = dict(map(lambda x:(x[0],np.sqrt(x[1]*(1-x[1])/n_shots)),prob_join.items()))

    return x_vals, prob_join, prob_join_error, n_shots

def get_join_prob_Nions(file_path):
    file = h5.File(file_path)

    tmp = [k for k in file["datasets"].keys()]

    data_name = dict([(x.split(".")[-1],x) for x in tmp])
    y_name = data_name["raw_counts"]
    x_name = data_name["x_values"]

    data = file["datasets"][y_name][...]
    x_vals = file["datasets"][x_name][...]

    n_pmt,n_shots,n_scan = data.shape
    ind = np.argsort(x_vals)
    x_vals = x_vals[ind]


    state = data>1
    state = state.astype(int)

    bit_seq_all = gen_bit_sequence(n_pmt+1,n_pmt-1)
    prob_join = {}

    for seq in bit_seq_all:
        prob_join[seq] = 0

    prob_join = dict(map(lambda x:(x,np.zeros((n_scan))),bit_seq_all))

    for i_scan in range(n_scan):

        for i_shot in range(n_shots):
            current_seq = ''.join(state[:,i_shot,i_scan].astype(str))
            prob_join[current_seq][i_scan] += 1


    prob_join = dict(map(lambda x:(x[0],x[1][ind]/n_shots),prob_join.items()))
    prob_join_error = dict(map(lambda x:(x[0],np.sqrt(x[1]*(1-x[1])/n_shots)),prob_join.items()))

    return x_vals, prob_join, prob_join_error, n_shots

def get_bitflip_prob(numions,edge_ions,bit_flips):
    numions=numions
    edge_ions=edge_ions
    bit_flips=bit_flips

    comb = np.array(list(itertools.combinations(range(numions), bit_flips)))
    grid = np.zeros((len(comb), numions), dtype="int8")
    bit_str=[]
    grid[np.arange(len(comb))[None].T,comb]=1
    #add zeros for edge ions
    for i in range(len(comb)):
        tmp=np.append(grid[i],0)
        tmp=np.insert(tmp,0,0)
        bit_str.append(''.join(str(e) for e in tmp))
    return bit_str

def compare_data(files, flag,bit_str):

    cmp = plt.get_cmap('viridis')

    ax_w = 8
    ax_h = 5
    n_ax = len(files)

# fig = plt.figure(figsize=(ax_w,ax_h*n_ax))
    if flag == 0:
        fig, ax = plt.subplots(1)
        fig.set_size_inches(ax_w,ax_h)
    else:
        fig, ax = plt.subplots(n_ax,1)
        fig.set_size_inches(ax_w,ax_h*n_ax)

    lgd = []
    for i in range(n_ax):
        expid = str(bytes(files[i])).split("\\")
        expid = expid[-1].split("-")
        expid = expid[0]

        lgd.append(expid)

        x1, p1, _ = get_join_prob_Nions(files[i])
        xdata = x1
        ydata = p1[bit_str]
        if flag == 0:
            ax.plot(xdata,ydata, 'o-', color = cmp((i+1)/n_ax))
        else:
            ax[i].plot(xdata,ydata,'o-', color = cmp((i+1)/n_ax))
            ax[i].set_title(expid)

    if flag == 0:
        ax.legend(lgd)
    # plt.show()

    return fig, ax


def compare_data_errorbar(files, flag, bit_str):

    cmp = plt.get_cmap('viridis')

    ax_w = 8
    ax_h = 5
    n_ax = len(files)
    plt_data = {}

# fig = plt.figure(figsize=(ax_w,ax_h*n_ax))
    if flag == 0:
        fig, ax = plt.subplots(1)
        fig.set_size_inches(ax_w,ax_h)
    else:
        fig, ax = plt.subplots(n_ax,1)
        fig.set_size_inches(ax_w,ax_h*n_ax)

    lgd = []
    for i in range(n_ax):
        expid = str(bytes(files[i])).split("\\")
        expid = expid[-1].split("-")
        expid = expid[0]

        lgd.append(expid)

        x1, p1, p1_error = get_join_prob_Nions(files[i])
        xdata = x1
        ydata = p1[bit_str]
        ydata_err = p1_error[bit_str]
        if flag == 0:
            ax.errorbar(xdata,ydata, yerr = ydata_err/2, color = cmp((i+1)/n_ax), marker = '.')
        else:
            ax[i].errorbar(xdata,ydata,yerr = ydata_err/2, color = cmp((i+1)/n_ax), marker = '.')
            ax[i].set_title(expid)

        plt_data[expid] = np.array([xdata, ydata, ydata_err])

    if flag == 0:
        ax.legend(lgd)
    # plt.show()

    return fig, ax, plt_data

def get_full_path(rid, file_path):

    files = []
    for i in rid:
        tmp = list(file_path.glob("*/*" + i + "*"))
        files.append(tmp[0])

    return files

def set_figsize_nature(fig,fig_ratio = 4/3, fig_column = 1):

    # fig_ratio = 4/3
    if fig_column == 1:
        fig_width = 89/25.4
    else:
        fig_withd = 183/25.4

    pt_2_inch = 0.01389
    fig_height = fig_width/fig_ratio
    fig.set_size_inches(fig_width,fig_height)
    fig.set_facecolor((1,1,1,1))


def match_Line2D_fun(obj):
    return str(type(obj)) == "<class 'matplotlib.lines.Line2D'>"

def match_Legend_fun(obj):
    return str(type(obj)) == "<class 'matplotlib.legend.Legend'>"

def set_ps_nature(ax, label_size = 10, tick_label_size = 10, legend_size = 8, marker_size = 2, line_width = 0.5, fontname = "Arial"):

    # set label format
    lb_font = {"fontname":fontname, "fontsize":label_size}
    plt.rcParams['text.usetex'] = True
    xlb = ax.get_xlabel()
    ax.set_xlabel(xlb, **lb_font)
    ylb = ax.get_ylabel()
    ax.set_ylabel(ylb, **lb_font)
    plt.rcParams['text.usetex'] = False
    # set ticklabel format:

    for lb in (ax.get_xticklabels()+ax.get_yticklabels()):
        # lb.set_font(fontname)
        lb.set_fontsize(tick_label_size)

    objs = ax.get_children()
    for obj in objs:
        # set line2D
        if match_Line2D_fun(obj):
            art.setp(obj,markersize = marker_size, linewidth = line_width)


def get_file_path_by_date(rid, date_str = []):

    # data_store_path = "//10.236.89.210/lab/CompactTrappedIonModule/Data/artiq_data/"
    # data_store_path = "/mnt/readynas/CompactTrappedIonModule/Data/artiq_data/"
    data_store_path = "/mnt/euriqa-nas/CompactTrappedIonModule/Data/artiq_data/"
    if date_str == []:
        file_path = data_store_path + str(dt.date.today())
    else:
        file_path = data_store_path + date_str

    file_path = pathlib.Path(file_path)
    files = []
    for i in rid:
        tmp = list(file_path.glob("*/*" + i + "*"))
        files.append(tmp[0])
    return files

def get_exp_path_by_date(exp_flag,date_str= [],latest_flag = 0):

    # data_store_path = "//10.236.89.210/lab/CompactTrappedIonModule/Data/artiq_data/"
    # data_store_path = "/mnt/readynas/CompactTrappedIonModule/Data/artiq_data/"
    data_store_path = "/mnt/euriqa-nas/CompactTrappedIonModule/Data/artiq_data/"
    if date_str == []:
        file_path = data_store_path + str(dt.date.today())
    else:
        file_path = data_store_path + date_str

    file_path = pathlib.Path(file_path)

    files = list(file_path.glob("*/*" + exp_flag + "*"))
    n_files = len(files)

    if n_files > 0:
        files.sort(key = os.path.getatime,reverse=False)
        if latest_flag > 0 and latest_flag <= n_files:
            new_files = files[n_files - latest_flag:n_files]
            return new_files
        else:
            return files


def get_recent_file_by_date(date_str = [], n_file = 3):
    # data_store_path = "//10.236.89.210/lab/CompactTrappedIonModule/Data/artiq_data/"
    # data_store_path = "/mnt/readynas/CompactTrappedIonModule/Data/artiq_data/"
    data_store_path = "/mnt/euriqa-nas/CompactTrappedIonModule/Data/artiq_data/"
    if date_str == []:
        file_path = data_store_path + str(dt.date.today())
    else:
        file_path = data_store_path + date_str

    file_path = pathlib.Path(file_path)

    all_files = list(file_path.glob("*/*.h5"))
    all_files.sort(key = os.path.getatime,reverse=True)
    latest_file = all_files[0:n_file+1]

    return latest_file

def show_exp_result_live(wait_time = 5, pmt = [8]):

    run_no = 0
    data_file = get_latest_file_by_date()
    show_ss_cooling_data(file_path = data_file, ax = None)
    plot_population_from_file(data_file)
    print(str(data_file) + "| run:{0}".format(run_no))

    old_data_file = data_file
    while 1:
        time.sleep(wait_time)
        run_no = run_no + 1
        data_file = get_latest_file_by_date()
        if data_file != old_data_file:
            show_ss_cooling_data(file_path = data_file, ax = None)
            plot_population_from_file(data_file)
            print(str(data_file) + "| run:{0}".format(run_no))
            old_data_file = data_file

def show_exp_result(file_path = None, ax = None):

    if file_path == None:
        file_path = get_latest_file_by_date()

    file = h5.File(file_path)

    tmp = [k for k in file["datasets"].keys()]
    if "data.loading.load_attempts" in tmp:
        # todo: add display loaing data here
        plot_loading_result(file_path)
        return 0

    data_name = dict([(x.split(".")[-1],x) for x in tmp])
    # y_name = data_name["raw_counts"]
    x_name = data_name["x_values"]
    ssc_name = data_name["sscooling"]

    ssc_data = file["datasets"][ssc_name][...]
    x_vals = file["datasets"][x_name][...]

    cmp = plt.get_cmap('viridis')

    if ax == None:
        clear_output()
        plt.figure(figsize=(10,5))

    n_pmts, n_scan = ssc_data.shape

    for i in range(n_pmts):
        plt.plot(x_vals,ssc_data[i],"o-",color = cmp(i/n_pmts))

    plt.title(file_path)
    if ax == None:
        plt.show()

    return 1

def get_latest_file_by_date(date_str = []):
    # data_store_path = "//10.236.89.210/lab/CompactTrappedIonModule/Data/artiq_data/"
    # data_store_path = "/mnt/readynas/CompactTrappedIonModule/Data/artiq_data/"
    data_store_path = "/mnt/euriqa-nas/CompactTrappedIonModule/Data/artiq_data/"
    if date_str == []:
        file_path = data_store_path + str(dt.date.today())
    else:
        file_path = data_store_path + date_str

    file_path = pathlib.Path(file_path)

    all_files = list(file_path.glob("*/*.h5"))
    latest_file = max(all_files, key=os.path.getmtime)

    return latest_file

def show_ss_cooling_live(wait_time = 5):

    run_no = 0
    data_file = get_latest_file_by_date()

    show_flag = exclude_calib_files(data_file)

    if show_flag:
        show_ss_cooling_data(file_path = data_file, ax = None)
        plot_population_from_file(data_file)
        print(str(data_file) + "| run:{0}".format(run_no))

    old_data_file = data_file
    while 1:
        time.sleep(wait_time)
        run_no = run_no + 1
        data_file = get_latest_file_by_date()
        show_flag = exclude_calib_files(data_file)
        if data_file != old_data_file and show_flag:
            show_ss_cooling_data(file_path = data_file, ax = None)
            plot_population_from_file(data_file)
            print(str(data_file) + "| run:{0}".format(run_no))
            old_data_file = data_file

def exclude_calib_files(file):
    file_name = str(file)
    if file_name.find("Calibration") == -1:
        return True
    else:
        return False




def show_ss_cooling_data(file_path = None, ax = None, figsize = (16,9)):

    if file_path == None:
        file_path = get_latest_file_by_date()

    file = h5.File(file_path)

    tmp = [k for k in file["datasets"].keys()]
    if "data.loading.load_attempts" in tmp:
        # todo: add display loaing data here
        plot_loading_result(file_path)
        return 0

    data_name = dict([(x.split(".")[-1],x) for x in tmp])
    # y_name = data_name["raw_counts"]
    x_name = data_name["x_values"]
    ssc_name = data_name["sscooling"]

    ssc_data = file["datasets"][ssc_name][...]
    x_vals = file["datasets"][x_name][...]

    cmp = plt.get_cmap('viridis')

    if ax == None:
        clear_output()
        plt.figure(figsize=figsize)

    n_pmts, n_scan = ssc_data.shape

    for i in range(n_pmts):
        plt.plot(x_vals,ssc_data[i],"o-",color = cmp(i/n_pmts))

    plt.title(file_path)
    if ax == None:
        plt.show()

    return 1, ssc_data, x_vals


def plot_loading_result(files,figsize = (16,9)):
    data = h5.File(files[0])
    inspect_counts = np.squeeze(data["datasets"]["data.loading.inspect_counts"][...])
    loading_pmt_counts = data["datasets"]["data.loading.pmt_counts"][...]
    cmp = plt.get_cmap("viridis")

    d1, d2 = loading_pmt_counts.shape

    clear_output()
    plt.figure(figsize=figsize)

    for i in range(d1):
        plt.plot(loading_pmt_counts[i],'o-',color = cmp(i/d1))

    plt.plot(inspect_counts,'s-k')
    plt.show()





# define function to calcualte spin correlation function

def get_population_from_file(file_path):
    data, x_values = get_data_from_file(file_path)
    prob = np.squeeze(np.average(data,axis = 1))
    return prob, x_values

def get_statistics_from_file(file_path):
    data, x_values = get_data_from_file(file_path)
    n_pmts, n_shots, n_scan = data.shape

    prob = np.squeeze(np.average(data,axis = 1))

    prob_error = np.sqrt(prob*(1-prob)/n_shots)

    return prob, prob_error,x_values

def get_mag_from_file(rid, date_str = [], pmts = [], stag_flag = False):
    file_path = get_file_path_by_date(rid = [rid], date_str = date_str)

    data, x_values = get_data_from_file(file_path[0])
    n_pmts, n_shots, n_scan = data.shape

    if pmts == []:
        pmt_ind = np.array([i for i in range(n_pmts)])
    else:
        pmt_ind = np.array(pmts)

    data = data[pmt_ind]
    data = (data*2 - 1)/2

    if stag_flag:
        data[::2] = -1*data[::2]

    mag = np.squeeze(np.average(data,axis = 0))
    mag_mean = np.squeeze(np.mean(mag,axis = 0))

    mag_error =  np.squeeze(np.std(mag,axis = 0))/np.sqrt(n_shots)

    return mag_mean, mag_error,x_values

def plot_population_from_file(file_path, clear_flag = False):
    prob, x_values = get_population_from_file(file_path)
    n_pmts,n_scans = prob.shape
    if clear_flag:
        clear_output()

    plt.figure(figsize=(10,5))
    cmp = plt.get_cmap('viridis')
    for i in range(n_pmts):
        plt.plot(x_values,prob[i],'o-',color = cmp(i/n_pmts))

    plt.title(file_path)
    plt.grid(True)
    plt.xlabel("scan")
    plt.ylabel("Population")
    plt.ylim(0,1)
    plt.show()


def get_data_from_file(file_path):
    file = h5.File(file_path)

    tmp = [k for k in file["datasets"].keys()]

    data_name = dict([(x.split(".")[-1],x) for x in tmp])

    if "data.loading.load_attempts" in tmp:
        # todo: add display loaing data here
        plot_loading_result(file_path)
        return 0, 0

    y_name = data_name["raw_counts"]
    x_name = data_name["x_values"]

    data = file["datasets"][y_name][...]
    x_vals = file["datasets"][x_name][...]
    # print('datadim=',data.shape)
    data = (data>1).astype(float)
    return data, x_vals

def calculate_connected_1_j_corr(data,edge_ind):
    # data should be a np.array the state of the ions, 0 , 1
    state  = data*2 - 1
    n_pmts, n_shots, n_scans = state.shape
    prob = np.repeat(np.mean(state,axis=1,keepdims=True),n_shots,axis = 1)
    state = state - prob
    new_state = np.zeros(shape = (n_pmts,n_shots,n_scans))

    for i_shot in range(n_shots):
        for i_scan in range(n_scans):
            new_state[:,i_shot,i_scan] = state[:,i_shot,i_scan]*np.repeat(state[edge_ind,i_shot,i_scan],n_pmts,axis = 0)

    corr = np.squeeze(np.average(new_state,axis = 1))

    return corr

def calculate_1_j_corr(data,edge_ind):
    # data should be a np.array the state of the ions, 0 , 1
    state  = data*2 - 1

    n_pmts, n_shots, n_scans = state.shape
    new_state = np.zeros(shape = (n_pmts,n_shots,n_scans))

    for i_shot in range(n_shots):
        for i_scan in range(n_scans):
            new_state[:,i_shot,i_scan] = state[:,i_shot,i_scan]*np.repeat(state[edge_ind,i_shot,i_scan],n_pmts,axis = 0)

    corr = np.squeeze(np.average(new_state,axis = 1))

    return corr

def calculate_connected_i_j_corr(data,ij_ind):
    # data should be a np.array the state of the ions, 0 , 1
    state  = data*2 - 1
    n_pmts, n_shots, n_scans = state.shape
    prob = np.repeat(np.mean(state,axis=1,keepdims=True),n_shots,axis = 1)
    state = state - prob

    tmp = np.ones(shape=(n_shots,n_scans))
    for i in ij_ind:
        tmp = tmp*state[i,:,:]
    corr = np.average(tmp,axis = 0)
    return corr


def calculate_i_j_corr(data,ij_ind):
    # data should be a np.array the state of the ions, 0 , 1
    state  = data*2 - 1
    n_pmts, n_shots, n_scans = state.shape

    tmp = np.ones(shape=(n_shots,n_scans))
    for i in ij_ind:
        tmp = tmp*state[i,:,:]
    corr = np.average(tmp,axis = 0)
    return corr


def calculate_connected_all_i_j_corr(data):
    # data should be a np.array the state of the ions, 0 , 1
    state  = data*2 - 1
    n_pmts, n_shots, n_scans = state.shape
    prob = np.repeat(np.mean(state,axis=1,keepdims=True),n_shots,axis = 1)
    state = state - prob

    corr = np.zeros(shape = (n_pmts,n_pmts, n_scans))

    for i in range(n_pmts):
        for j in range(i, n_pmts):
            tmp = state[j,:,:]*state[i,:,:]
            corr[i,j] = np.average(tmp,axis = 0)
            corr[j,i] = corr[i,j]
    return corr

def calculate_all_i_j_corr(data):
    # data should be a np.array the state of the ions, 0 , 1
    state  = data*2 - 1
    n_pmts, n_shots, n_scans = state.shape

    corr = np.zeros(shape = (n_pmts,n_pmts, n_scans))

    for i in range(n_pmts):
        for j in range(i, n_pmts):
            tmp = state[j,:,:]*state[i,:,:]
            corr[i,j] = np.average(tmp,axis = 0)
            corr[j,i] = corr[i,j]
    return corr

def calculate_connected_delta_j_corr(data):

    state  = data*2 - 1
    n_pmts, n_shots, n_scans = state.shape
    prob = np.repeat(np.mean(state,axis=1,keepdims=True),n_shots,axis = 1)
    state = state - prob


    djs = [i for i in range(0,n_pmts)]

    corr = np.zeros(shape = (len(djs),n_scans))

    for dj in djs:
        for i_scan in range(n_scans):
            tmp1 = state[:,:,i_scan][0:(n_pmts-dj)]
            tmp2 = state[:,:,i_scan][dj:n_pmts]
            tmp = np.squeeze(np.average(tmp1*tmp2,axis = 0))
            tmp = np.average(tmp)
            corr[dj,i_scan] = tmp

    return corr

def calculate_connected_delta_j_corr_offset(data, edge_ion = 0):

    d1, d2, d3 = data.shape
    state  = data[edge_ion:d1-edge_ion]*2 - 1

    n_pmts, n_shots, n_scans = state.shape
    prob = np.repeat(np.mean(state,axis=1,keepdims=True),n_shots,axis = 1)
    state = state - prob


    djs = [i for i in range(0,n_pmts)]

    corr = np.zeros(shape = (len(djs),n_scans))

    for dj in djs:
        for i_scan in range(n_scans):
            tmp1 = state[:,:,i_scan][0:(n_pmts-dj)]
            tmp2 = state[:,:,i_scan][dj:n_pmts]
            tmp = np.squeeze(np.average(tmp1*tmp2,axis = 0))
            tmp = np.average(tmp)
            corr[dj,i_scan] = tmp

    return corr

def calculate_delta_j_corr_offset(data, edge_ion = 0):

    d1, d2, d3 = data.shape
    state  = data[edge_ion:d1-edge_ion]*2 - 1

    n_pmts, n_shots, n_scans = state.shape


    djs = [i for i in range(0,n_pmts)]

    corr = np.zeros(shape = (len(djs),n_scans))

    for dj in djs:
        for i_scan in range(n_scans):
            tmp1 = state[:,:,i_scan][0:(n_pmts-dj)]
            tmp2 = state[:,:,i_scan][dj:n_pmts]
            tmp = np.squeeze(np.average(tmp1*tmp2,axis = 0))
            tmp = np.average(tmp)
            corr[dj,i_scan] = tmp

    return corr

def calculate_delta_j_corr(data):

    state  = data*2 - 1
    n_pmts, n_shots, n_scans = state.shape

    djs = [i for i in range(0,n_pmts)]

    corr = np.zeros(shape = (len(djs),n_scans))

    for dj in djs:
        for i_scan in range(n_scans):
            tmp1 = state[:,:,i_scan][0:(n_pmts-dj)]
            tmp2 = state[:,:,i_scan][dj:n_pmts]
            tmp = np.squeeze(np.average(tmp1*tmp2,axis = 0))
            tmp = np.average(tmp)
            corr[dj,i_scan] = tmp

    return corr



# define simple fit functions
def simple_gaussian(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/2/sigma**2)

def offset_gaussian(x,a,x0,b,sigma):
    return a*np.exp(-(x-x0)**2/2/sigma**2) + b


def fit_simple_gaussian(x,y):
    a0 = np.max(y)
    x0 = x[np.argmax(y)]
    sigma = x[1]-x[0]

    popt, pcov = sp.optimize.curve_fit(simple_gaussian, x,y, p0 = [a0,x0,sigma])

    fun_evaluate = lambda x: simple_gaussian(x,*popt)
    return popt, pcov, fun_evaluate

def fit_gaussian(x,y, p0 = None):
    if p0 == None:
        a0 = np.max(y)
        x0 = x[np.argmax(y)]
        b0 = np.min(y)
        sigma = x[1]-x[0]
    else:
        a0 = p0[0]
        x0 = p0[1]
        b0 = p0[2]
        sigma = p0[3]

    popt, pcov = sp.optimize.curve_fit(offset_gaussian, x,y, p0 = [a0,x0,b0,sigma])

    fun_evaluate = lambda x: offset_gaussian(x,*popt)
    return popt, pcov, fun_evaluate

def fit_simple_sin(x,y, p0):

    fun = lambda x, a, b, c, T:a*np.cos(x/T + b)+c

    popt, pcov = sp.optimize.curve_fit(fun,x,y,p0 = p0)


    fun_val = lambda x:fun(x,*popt)
    return popt,pcov,fun_val


def monitor_center_checks(exp_flag,init_offset = 5,rid=[], date_str = [],plot_rec = [], wait_time = 2, cost_order = 4, show_live = True, edge_ions = 0):
    exp_flag=exp_flag
    print(exp_flag)
    center_files=get_exp_path_by_date(exp_flag =exp_flag,date_str = date_str,latest_flag=init_offset)
    #center_files=get_file_path_by_date(rid = [rid],date_str = date_str)
    print(center_files)

    if len(center_files)>plot_rec:
        plot_files = center_files[len(center_files)-plot_rec:len(center_files)]
    else:
        plot_files = center_files

    plot_center_checks_ions(plot_files, edge_ions = edge_ions)
    center_values = []
    center_cost = []
    center_rid = []

    for file in center_files:
        x0, sigma, rid = get_center_check_result(file)
        ion_ind = [j for j in range(edge_ions,len(x0)-edge_ions)]
        x0 = np.array([x0[j] for j in ion_ind])
        center_values.append(x0)
        center_rid.append(rid)

        x0_diff = np.sum(np.abs((x0 - np.mean(x0))*1000)**cost_order)
        center_cost.append(x0_diff)

    plt.figure(figsize=(10,6))
    plt.plot(center_rid,center_cost,'o-b')
    plt.grid(True)
    plt.yscale("log")
    plt.show()

    old_file = center_files[-1]

    while 1 and show_live:
        time.sleep(wait_time)
        new_files = get_exp_path_by_date(exp_flag = "Calibrate_CheckCenter15",date_str = date_str,latest_flag=1)
        if new_files[0] != old_file:
            old_file = new_files[0]
            center_files.append(old_file)

            x0, sigma, rid = get_center_check_result(new_files[0])
            ion_ind = [j for j in range(edge_ions,len(x0)-edge_ions)]
            x0 = np.array([x0[j] for j in ion_ind])
            sigma = np.array([sigma[j] for j in ion_ind])
            center_values.append(x0)
            center_rid.append(rid)

            x0_diff = np.sum(np.abs((x0 - np.mean(x0))*1000)**cost_order)
            center_cost.append(x0_diff)

            clear_output()
            if len(center_files)>plot_rec:
                plot_files = center_files[len(center_files)-plot_rec:len(center_files)]
            else:
                plot_files = center_files
            plot_center_checks_ions(plot_files,edge_ions)
            plt.figure(figsize=(10,6))
            plt.plot(center_rid,center_cost,'o-b')
            plt.grid(True)
            plt.yscale("log")
            plt.show()


def get_center_check_result(file):
    f = h5.File(file)
    x0 = f["datasets"]['data.check.Center.fitparam_x0'][...]
    sigma = f["datasets"]['data.check.Center.fitparam_sigma'][...]
    rid = f["rid"][...]
    return x0, sigma, rid

def plot_center_checks(files):

    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cmp = plt.get_cmap("viridis")
    plot_lgd = []
    for i, file in enumerate(files):
        x0, sigma, rid = get_center_check_result(file)
        x0 = np.array(x0)
        x0 = x0 - np.mean(x0)

        print('x0 vals',x0)
        x_plot = [i+1 for i in range(len(x0))]
        ax1.plot(x_plot,x0,'o-',color = cmp(i/len(files)))
        ax2.plot(x_plot,sigma,'o-',color = cmp(i/len(files)))
        ax1.set_grid(True)
        plot_lgd.append(rid)

    # plt.grid(True)
    plt.legend(plot_lgd)
    plt.show()

def plot_center_checks_ions(files, edge_ions):
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    cmp = plt.get_cmap("viridis")
    plot_lgd = []
    for i, file in enumerate(files):
        x0, sigma, rid = get_center_check_result(file)
        ion_ind = [j for j in range(edge_ions,len(x0)-edge_ions)]
        x0 = np.array([x0[j] for j in ion_ind])
        sigma = np.array([sigma[j] for j in ion_ind])
        x0 = x0 - np.mean(x0)

        print('x0 vals',x0)

        x_plot = [i+1-(len(x0)+1)/2 for i in range(len(x0))]
        ax1.plot(x_plot,x0,'o-',color = cmp(i/len(files)))
        ax2.plot(x_plot,np.abs(x0),'o-',color = cmp(i/len(files)))
        ax3.plot(x_plot,sigma,'o-',color = cmp(i/len(files)))
        plot_lgd.append(rid)

    ax1.grid(True)
    ax2.grid(True)
    ax1.set_ylabel("Mean X0")
    ax2.set_ylabel("Abs X0")
    ax3.set_ylabel("sigma")
    plt.grid(True)
    plt.legend(plot_lgd)
    plt.show()

def get_voltage_settings(rid,date_str = []):
    files = get_file_path_by_date(rid = rid,date_str=date_str)

    voltages = dict(map(lambda x:(x,{"X2":0,"X3":0,"X4":0,"QZY":0 }),rid))

    for r,file in zip(rid,files):
        f = h5.File(file)

        voltages[r]["X1"] = f["archive"]["global.Voltages.X1"][...]
        voltages[r]["X2"] = f["archive"]["global.Voltages.X2"][...]
        voltages[r]["X3"] = f["archive"]["global.Voltages.X3"][...]
        voltages[r]["X4"] = f["archive"]["global.Voltages.X4"][...]
        voltages[r]["QZY"] = f["archive"]["global.Voltages.QZY"][...]
        voltages[r]["QZZ"] = f["archive"]["global.Voltages.QZZ"][...]
        voltages[r]["QXZ"] = f["archive"]["global.Voltages.QXZ"][...]


        voltages[r]["DX"] = f["archive"]["global.Voltages.Offsets.DX"][...]
        voltages[r]["DY"] = f["archive"]["global.Voltages.Offsets.DY"][...]
        voltages[r]["DZ"] = f["archive"]["global.Voltages.Offsets.DZ"][...]
        voltages[r]["X2_offset"] = f["archive"]["global.Voltages.Offsets.X2"][...]
        voltages[r]["X4_offset"] = f["archive"]["global.Voltages.Offsets.X4"][...]
        voltages[r]["QZY_offset"] = f["archive"]["global.Voltages.Offsets.QZY"][...]
        voltages[r]["QZZ_offset"] = f["archive"]["global.Voltages.Offsets.QZZ"][...]
        voltages[r]["QXZ_offset"] = f["archive"]["global.Voltages.Offsets.QXZ"][...]
        voltages[r]["Center"] = f["archive"]["global.Voltages.center"][...]

    return voltages


# pakced star_shift calibration for each ions
def Calibrate_stark_shift(rid, date_str = [], n_scans = 9, n_pmts = 13, T_estimate_div = 1.4, thresh = 0.25):

    def fit_exp_cos(x,y, p0):

        fun = lambda x, a, b, c, e , T:a*np.sin(2*np.pi*x/T + b)*np.exp(-x/e*1)+c
        popt, pcov = sp.optimize.curve_fit(fun,x,y,p0 = p0,maxfev=5000,bounds= ([-0.5,0,0.499,0,0],[0.5,np.pi,0.501,np.Inf,np.Inf]))
        fun_val = lambda x:fun(x,*popt)
        return popt,pcov,fun_val

    def guess_T(data, t_step = 1):
        data_fft = np.fft.fft(data)
        pow = np.abs(data_fft)**2
        ind = int(len(pow)/2)
        pow = pow[0:ind]
        ind_max = np.argwhere(pow == np.max(pow))
        ind_max[0]
        if ind_max >0:
            return len(data)/ind_max*t_step
        else:
            return len(data)*2*t_step

    # rids= ["294813"]

    # rids= ["294817"]
    rids= rid
    files = get_file_path_by_date(rid = rids, date_str = date_str)

    p,x =  get_population_from_file(files[0])
    # n_scans = n_scans
    # n_pmts = 13
    T_fit = []

    n_row = np.max([int(np.sqrt(n_pmts)),2])
    n_col = np.max([int(np.floor(n_pmts/n_row))+1,2])

    fig, ax = plt.subplots(n_row,n_col)
    fig.set_size_inches((n_row*8,n_col*3))
    ax = [x for xs in ax for x in xs]

    plt.figure(figsize = (10,6))
    cmp = plt.get_cmap("viridis")
    n_estimate = T_estimate_div
    thresh_contrast = thresh


    for i in range(n_pmts):
        # x_data = x[n_scans*i:n_scans*(i+1)]
        x_data = x[0:n_scans]
        y_data = np.squeeze(p[i])
        # y_data = y_data[n_scans*i:n_scans*(i+1)]
        ax[i].plot(x_data,y_data,'--o',color = cmp(i/n_pmts))
        ax[i].set_ylim([0,1])
        ax[i].grid(True)
        ax[i].set_title("PMT_{0}".format(i))

        print(y_data.shape)
        print(p.shape)
        if y_data.shape[0] == 0:
            y_con = 0
        else:
            y_con = np.max(y_data)-np.min(y_data)
        if y_con <= thresh_contrast:
            T_fit.append([0,0,0])
            print("Low contrast, no fit")
            print([0,0,0])
            continue

        if y_data[1]>y_data[0]:
            a0 = np.max(y_data) - 0.5
            b0 = 0
            # print("fit_sign_positive")
        else:
            a0 = np.min(y_data) - 0.5
            b0 = 0
            # print("fit_sign_negative")

        t_step = x_data[1]-x_data[0]
        T0 = guess_T(y_data - np.mean(y_data),t_step)/n_estimate
        c0 = 0.5
        e0 = 1*T0
        p0 = [a0,b0,c0,e0,T0]
        param, pcov, fun = fit_exp_cos(x_data,y_data,p0)
        T_fit.append([param[-1],param[0], param[-2]])

        x_plot = np.linspace(x_data[0],np.max([x_data[-1],1*param[-1]]),100)
        y_plot = fun(x_plot)
        ax[i].plot(x_plot,y_plot,'-',color = cmp(i/n_pmts))

        print([param[-1],param[0], param[-2]])

    plt.show()

    fix, ax = plt.subplots(2,1,figsize=(10,10))
    T_fit = np.array(T_fit)
    ax[0].plot(T_fit[:,0],'o-k',fillstyle = "none")
    ax[0].set_xlabel("PMT #")
    ax[0].set_ylabel("T_2PI (ms)")
    ax[0].set_title(rid)
    ax[0].grid(True)
    f_shift = [1000/T_fit[i,0]*np.sign(T_fit[i,1]) for i in range(n_pmts)]
    ax[1].plot(f_shift,'o-k',fillstyle = "none")
    ax[1].set_xlabel("PMT #")
    ax[1].set_ylabel("f_shift Hz")
    ax[1].grid(True)
    plt.show()
    print(f_shift)

    return f_shift, T_fit

def Calibrate_stark_shift_seg(rid, date_str = [], n_scans = 9, n_pmts = 13, T_estimate_div = 1.4, thresh = 0.25, ion_start = 0):

    def fit_exp_cos(x,y, p0):

        fun = lambda x, a, b, c, e , T:a*np.sin(2*np.pi*x/T + b)*np.exp(-x/e*0)+c
        popt, pcov = sp.optimize.curve_fit(fun,x,y,p0 = p0,maxfev=5000,bounds= ([-0.5,0,0.499,0,0],[0.5,np.pi,0.501,np.Inf,np.Inf]))
        fun_val = lambda x:fun(x,*popt)
        return popt,pcov,fun_val

    def guess_T(data, t_step = 1):
        data_fft = np.fft.fft(data)
        pow = np.abs(data_fft)**2
        ind = int(len(pow)/2)
        pow = pow[0:ind]
        ind_max = np.argwhere(pow == np.max(pow))
        ind_max[0]
        if ind_max >0:
            return len(data)/ind_max*t_step
        else:
            return len(data)*2*t_step

    # rids= ["294813"]

    # rids= ["294817"]
    rids= rid
    files = get_file_path_by_date(rid = rids, date_str = date_str)

    p,x =  get_population_from_file(files[0])
    n_tot, active_pmt = p.shape

    if n_tot/active_pmt == n_scans:
        ion_start = 0
    # n_scans = n_scans
    # n_pmts = 13
    T_fit = []

    n_row = np.max([int(np.sqrt(n_pmts)),2])
    n_col = np.max([int(np.floor(n_pmts/n_row))+1,2])

    fig, ax = plt.subplots(n_row,n_col)
    fig.set_size_inches((n_row*8,n_col*3))
    ax = [x for xs in ax for x in xs]

    plt.figure(figsize = (10,6))
    cmp = plt.get_cmap("viridis")
    n_estimate = T_estimate_div
    thresh_contrast = thresh


    for i in range(n_pmts - ion_start):
        x_data = x[n_scans*i:n_scans*(i+1)]
        y_data = np.squeeze(p[i+ion_start])
        y_data = y_data[n_scans*i:n_scans*(i+1)]
        ax[i].plot(x_data,y_data,'--o',color = cmp(i/n_pmts))
        ax[i].set_ylim([0,1])
        ax[i].grid(True)
        ax[i].set_title("PMT_{0}".format(i+ion_start))

        print(y_data.shape)
        print(p.shape)
        if y_data.shape[0] == 0:
            y_con = 0
        else:
            y_con = np.max(y_data)-np.min(y_data)
        if y_con <= thresh_contrast:
            T_fit.append([0,0,0])
            print("Low contrast, no fit")
            print([0,0,0])
            continue

        if y_data[1]>y_data[0]:
            a0 = np.max(y_data) - 0.5
            b0 = 0
            # print("fit_sign_positive")
        else:
            a0 = np.min(y_data) - 0.5
            b0 = 0
            # print("fit_sign_negative")

        t_step = x_data[1]-x_data[0]
        T0 = guess_T(y_data - np.mean(y_data),t_step)/n_estimate
        c0 = 0.5
        e0 = 1*T0
        p0 = [a0,b0,c0,e0,T0]
        param, pcov, fun = fit_exp_cos(x_data,y_data,p0)
        T_fit.append([param[-1],param[0], param[-2]])

        x_plot = np.linspace(x_data[0],np.max([x_data[-1],1*param[-1]]),100)
        y_plot = fun(x_plot)
        ax[i].plot(x_plot,y_plot,'-',color = cmp(i/n_pmts))

        print([param[-1],param[0], param[-2]])

    plt.show()

    fix, ax = plt.subplots(2,1,figsize=(10,10))
    T_fit = np.array(T_fit)
    ax[0].plot(T_fit[:,0],'o-k',fillstyle = "none")
    ax[0].set_xlabel("PMT #")
    ax[0].set_ylabel("T_2PI (ms)")
    ax[0].set_title(rid)
    ax[0].grid(True)
    f_shift = [1000/T_fit[i,0]*np.sign(T_fit[i,1]) for i in range(n_pmts-ion_start)]
    ax[1].plot(f_shift,'o-k',fillstyle = "none")
    ax[1].set_xlabel("PMT #")
    ax[1].set_ylabel("f_shift Hz")
    ax[1].grid(True)
    plt.show()
    print(f_shift)

    return f_shift, T_fit



# packed calibration for calibrating stark shift with one ion for different power:

def Calibrate_stark_shift_ramp(rid, date_str = [], n_scans = 9, amp = [],T_estimate_div = 1.2, thresh = 0.1, pmt = 0):

    def fit_exp_cos(x,y, p0):

        fun = lambda x, a, b, c, e , T:a*np.sin(2*np.pi*x/T + b)*np.exp(-x/e)+c
        popt, pcov = sp.optimize.curve_fit(fun,x,y,p0 = p0,maxfev=5000,bounds= ([-0.5,0,0.499,0,0],[0.5,np.pi,0.501,np.Inf,np.Inf]))
        fun_val = lambda x:fun(x,*popt)
        return popt,pcov,fun_val

    def guess_T(data, t_step = 1):
        data_fft = np.fft.fft(data)
        pow = np.abs(data_fft)**2
        ind = int(len(pow)/2)
        pow = pow[0:ind]
        ind_max = np.argwhere(pow == np.max(pow))
        ind_max[0]
        if ind_max >0:
            return len(data)/ind_max*t_step
        else:
            return len(data)*2*t_step

    # rids = ["294899"]
    files = get_file_path_by_date(rid = rid, date_str = date_str)

    p,x =  get_population_from_file(files[0])
    n_sample = len(amp)
    # amp = [0,0.25,0.5,0.75,1]
    T_fit = []

    n_row = int(np.sqrt(n_sample))
    n_col = int(np.floor(n_sample/n_row))+1

    fig, ax = plt.subplots(n_row,n_col)
    fig.set_size_inches((n_row*8,n_col*3))
    ax = [x for xs in ax for x in xs]

    plt.figure(figsize = (10,6))
    cmp = plt.get_cmap("viridis")
    n_estimate = T_estimate_div
    thresh_contrast = thresh

    for i in range(n_sample):
        x_data = x[n_scans*i:n_scans*(i+1)]
        y_data = np.squeeze(p[pmt])
        y_data = y_data[n_scans*i:n_scans*(i+1)]
        ax[i].plot(x_data,y_data,'--o',color = cmp(i/n_sample))
        ax[i].set_ylim([0,1])
        ax[i].grid(True)
        ax[i].set_title("amp = {0}".format(amp[i]))

        y_con = np.max(y_data)-np.min(y_data)
        if y_con <= thresh_contrast:
            T_fit.append([0,0,0])
            print("Low contrast, no fit")
            print([0,0,0])
            continue

        if y_data[1]>y_data[0]:
            ind = np.argwhere(y_data==np.max(y_data))
            ind = ind[0]
            a0 = np.max(y_data) - 0.5
            b0 = 0

            # print("fit_sign_positive")
        else:
            ind = np.argwhere(y_data==np.min(y_data))
            ind = ind[0]
            a0 = np.min(y_data) - 0.5
            b0 = 0
            # print("fit_sign_negative")

        t_step = x_data[1] - x_data[0]
        T0 = guess_T(y_data,t_step)/n_estimate
        # T0 = 0.5/np.abs(x_data[ind]-0.5)*n_estimate
        c0 = 0.5
        e0 = 1*T0
        p0 = [a0,b0,c0,e0,T0]
        param, pcov, fun = fit_exp_cos(x_data,y_data,p0)
        T_fit.append([param[-1],param[0], param[-2]])

        x_plot = np.linspace(x_data[0],np.max([x_data[-1],1*param[-1]]),100)
        y_plot = fun(x_plot)
        ax[i].plot(x_plot,y_plot,'-',color = cmp(i/n_sample))

        # print([param[-1],param[0], param[-2]])

    plt.show()

    fix, ax = plt.subplots(2,1,figsize=(10,10))
    T_fit = np.array(T_fit)
    ax[0].plot(amp, T_fit[:,0],'o-k',fillstyle = "none")
    ax[0].set_xlabel("Amp")
    ax[0].set_ylabel("T_2PI (ms)")
    ax[0].set_title(rid)
    ax[0].grid(True)
    f_shift = [1000/T_fit[i,0]*np.sign(T_fit[i,1]) for i in range(n_sample)]
    ax[1].plot(amp,f_shift,'o-k',fillstyle = "none")
    ax[1].set_xlabel("Amp")
    ax[1].set_ylabel("f_shift Hz")
    ax[1].grid(True)

    plt.show()
    print(f_shift)
    return f_shift



# packed calibration for SK1 Pi time scale factor
def Calibrate_Pi_factor_SK1(rid ,date_str = []):

    def fit_simple_cos(x,y, p0):

        fun = lambda x, a, b, c, T:a*np.cos(x/T + b)+c
        popt, pcov = sp.optimize.curve_fit(fun,x,y,p0 = p0,maxfev=5000,bounds= ([-0.5,0,0.499,0],[0.5,np.pi,0.501,np.Inf]))
        fun_val = lambda x:fun(x,*popt)
        return popt,pcov,fun_val

    def guess_T(data, t_step = 1):
        data_fft = np.fft.fft(data)
        pow = np.abs(data_fft)**2
        ind = int(len(pow)/2)
        pow = pow[0:ind]
        ind_max = np.argwhere(pow == np.max(pow))
        ind_max[0]
        if ind_max >0:
            return len(data)/ind_max*t_step
        else:
            return len(data)*2*t_step

    file_path =get_file_path_by_date(rid = rid, date_str = date_str)
    data,x = get_data_from_file(file_path=file_path[0])
    data = np.average(data,axis = 1)

    n_pmts, n_scans = data.shape

    p0 = [0.5,0.0,0.5,6]
    T = []
    fit_param = []
    cmp = plt.get_cmap('viridis')
    plt.figure(figsize=(10,5))
    for i in range(n_pmts):

        t_step = x[1]-x[0]
        y_data = np.squeeze(data[i]-np.mean(data[i]))
        T0 = guess_T(y_data,t_step)
        p0[-1] = T0/2/np.pi/1

        if y_data[2]>y_data[0]:
            p0[0] = -0.5
        else:
            p0[0] = 0.5

        popt,pcov,fun = fit_simple_cos(x,data[i],p0 = p0)
        # print(popt[-1])
        T.append(popt[-1])
        fit_param.append(popt)

        plt.plot(x,data[i],'o',color = cmp(i/n_pmts))

        x_p = np.linspace(x[0],x[-1],150)
        y_p = fun(x_p)
        plt.plot(x_p,y_p,"-",color = cmp(i/n_pmts))

    plt.show()

    plt.figure()
    fun = lambda x, a, b, c, T:a*np.cos(x/T + b)+c
    for i in range(1,n_pmts-1):
    # i = 1
        y_p = fun(x_p,*fit_param[i])
        plt.plot(np.array(x_p)/T[i],y_p,'-',color = cmp(i/n_pmts))
        plt.plot(np.array(x)/T[i],data[i],'o',color = cmp(i/n_pmts))


    plt.show()

    plt.figure()
    data_print = [x[0] for x in fit_param]
    plt.plot(data_print,'o-b')
    # print(data_print)
    return T

# packed calibrate Raman spectrum:

def calibrate_raman_spectrum(rid, date_str = [], aspect = 200):
    file = get_file_path_by_date(rid = [rid], date_str=date_str)
    data, x = get_population_from_file(file[0])
    data_max = np.max(np.max(data))

    n_pmts, n_scan = data.shape
    n_plots = n_pmts + 2
    n_row = int(np.sqrt(n_plots))
    n_col = int(np.floor(n_plots/n_row))

    if n_row*n_col < n_plots:
        if n_row<=n_col:
            n_row = n_row + 1
        else:
            n_col = n_col + 1


    fig, ax = plt.subplots(n_row,n_col)
    fig.set_size_inches((n_row*10,n_col*2))
    ax = [x for xs in ax for x in xs]

    cmp = plt.get_cmap('rainbow')
    for i in range(n_pmts):
        ax[i].plot(x,data[i],'.-',color = cmp(i/n_pmts))
        ax[i].set_xlim([x[0],x[-1]])
        ax[i].set_ylim([-0.01,data_max])
        ax[i].set_title("PMT#{0}".format(i))

    for i in range(n_pmts):
        ax[n_pmts].plot(x,data[i],'.-',color = cmp(i/n_pmts))
        ax[n_pmts].set_xlim([x[0],x[-1]])
        ax[n_pmts].set_title("PMT ALL")

    data_avg = np.mean(data,axis = 0)
    ax[n_pmts+1].plot(x,data_avg,'.-k')
    ax[n_pmts+1].set_xlim([x[0],x[-1]])
    ax[n_pmts+1].set_title("Average over pmts")

    plt.show()

    plt.figure(figsize=(20,6))
    plt.imshow(data, cmap= plt.get_cmap("viridis"), extent=[np.min(x),np.max(x),(n_pmts-1)/2,-(n_pmts-1)/2],aspect = aspect)
    plt.show()
    return data, x

def calculate_di_dj_correlation(data):
    state = data*2 - 1
    n_pmts, n_shots, n_scans = state.shape
    prob = np.repeat(np.mean(state,axis=1,keepdims=True),n_shots,axis = 1)
    state = state - prob

    x = [i for i in range(0,n_pmts)]

    corr = np.zeros((2*n_pmts-1,n_pmts,n_scans))
    for ix in x:
        for iy in range(ix - n_pmts+2, ix+0):
            counter = 0
            tmp_corr = 0
            for i in range(0,n_pmts):
                if (i+iy>=0 and i+iy <n_pmts) and (i+ix>=0 and i+ix <n_pmts):
                    tmp_corr = tmp_corr + np.mean(np.squeeze(state[i]*state[i+iy]*state[i+ix]),axis = 0)
                    counter = counter + 1
            corr[n_pmts-1+iy,ix] = tmp_corr/counter

    return corr
# packed calibration for correlation
def inspect_correlation(rid = "", date_str = [], ind_ion = 6, color_map = "plasma", color_range = [-0.4, 0.4],edge_ion = 2):

    c_max = np.max(color_range)
    c_min = np.min(color_range)

    files = get_file_path_by_date(rid = [rid],date_str=date_str)
    print(files)
    data, x = get_data_from_file(files[0])

    n_pmts, n_shots, n_scans = data.shape
    # data = (data>1).astype(int)

    print(data.shape)
    corr_1j = calculate_connected_1_j_corr(data,ind_ion)
    corr_dj = calculate_connected_delta_j_corr_offset(data,edge_ion)

    y_dim, x_dim = corr_1j.shape

    # plt.figure()
    # plt.plot(corr_1j[:,1],'o-b')

    plt.figure(figsize = (6,6))
    plt.imshow(corr_1j, extent = [0, x_dim, y_dim,0], cmap = color_map, vmax= c_max, vmin = c_min)
    plt.colorbar()
    plt.title("g_1j")
    plt.xlabel("scan")
    plt.ylabel("j")
    # plt.ylim([0,0.4])
    plt.show()

    plt.figure(figsize = (6,6))
    plt.imshow(corr_dj, extent = [0, x_dim, y_dim,0], cmap = color_map, vmax= c_max, vmin = c_min)
    plt.colorbar()
    plt.title("g_dj")
    plt.xlabel("scan")
    plt.ylabel("j")
    # plt.ylim([0,0.4])
    plt.show()

    cmp = plt.get_cmap('viridis')
    plt.figure(figsize = (8,4))

    for i in range(x_dim):
        plt.plot(corr_1j[:,i],'o-',color = cmp(i/x_dim), fillstyle = "none")
    plt.title("g_1j")
    plt.xlabel("j")
    plt.ylabel("g_1j")
    plt.grid(True)
    plt.ylim([-0.3,0.4])

    plt.show()



    corr_ij = calculate_connected_all_i_j_corr(data)
    d1, d2 , d3 = corr_ij.shape





    n_row = np.max([int(np.sqrt(d3)),2])
    n_col = np.max([int(np.floor(d3/n_row))+1,2])

    fig, ax = plt.subplots(n_row,n_col)
    fig.set_size_inches((n_row*8,n_col*3))
    ax = [x for xs in ax for x in xs]

    for i_scan in range(d3):

        im = ax[i_scan].imshow(corr_ij[:,:,i_scan], extent = [0, d2, d1, 0], cmap = color_map, vmax= c_max, vmin = c_min)
        ax[i_scan].set_title("g_ij at i_scan = {0} in {1} scans".format(i_scan,d3))
        ax[i_scan].set_xlabel("i")
        ax[i_scan].set_ylabel("j")
        fig.colorbar(im)


    # plt.colorbar()
    # fig.colorbar()
    plt.show()
    return corr_1j

def inspect_correlation_regular(rid = "", date_str = [], ind_ion = 6, color_map = "plasma", color_range = [-0.4, 0.4]):

    c_max = np.max(color_range)
    c_min = np.min(color_range)

    files = get_file_path_by_date(rid = [rid],date_str=date_str)
    print(files)
    data, x = get_data_from_file(files[0])

    n_pmts, n_shots, n_scans = data.shape
    # data = (data>1).astype(int)

    print(data.shape)
    corr_1j = calculate_1_j_corr(data,ind_ion)
    corr_dj = calculate_connected_delta_j_corr_offset(data,ind_ion)

    y_dim, x_dim = corr_1j.shape

    # plt.figure()
    # plt.plot(corr_1j[:,1],'o-b')

    plt.figure(figsize = (6,6))
    plt.imshow(corr_1j, extent = [0, x_dim, y_dim,0], cmap = color_map, vmax= c_max, vmin = c_min)
    plt.colorbar()
    plt.title("g_1j")
    plt.xlabel("scan")
    plt.ylabel("j")
    plt.show()


    plt.figure(figsize = (6,6))
    plt.imshow(corr_dj, extent = [0, x_dim, y_dim,0], cmap = color_map, vmax= c_max, vmin = c_min)
    plt.colorbar()
    plt.title("g_1j")
    plt.xlabel("scan")
    plt.ylabel("j")
    plt.show()

    cmp = plt.get_cmap('viridis')
    plt.figure(figsize = (8,4))

    for i in range(x_dim):
        plt.plot(corr_1j[:,i],'o-',color = cmp(i/x_dim), fillstyle = "none")
    plt.title("g_1j")
    plt.xlabel("j")
    plt.ylabel("g_1j")
    plt.grid(True)
    plt.show()



    corr_ij = calculate_all_i_j_corr(data)
    d1, d2 , d3 = corr_ij.shape

    n_row = np.max([int(np.sqrt(d3)),2])
    n_col = np.max([int(np.floor(d3/n_row))+1,2])

    fig, ax = plt.subplots(n_row,n_col)
    fig.set_size_inches((n_row*8,n_col*3))
    ax = [x for xs in ax for x in xs]

    for i_scan in range(d3):

        im = ax[i_scan].imshow(corr_ij[:,:,i_scan], extent = [0, d2, d1, 0], cmap = color_map, vmax= c_max, vmin = c_min)
        ax[i_scan].set_title("g_ij at i_scan = {0} in {1} scans".format(i_scan,d3))
        ax[i_scan].set_xlabel("i")
        ax[i_scan].set_ylabel("j")
        fig.colorbar(im)


    # plt.colorbar()
    # fig.colorbar()
    plt.show()



def save_data_Matlab(rid = "", date_str = [], edge_ion = 0, appendix_str = "", L_sys = 23):

    # save_path = save_correlation(rid = rid, date_str = date_str, appendix_str = appendix_str, edge_ion = edge_ion)

    save_path = save_correlation_reg(rid = rid, date_str = date_str, appendix_str = appendix_str, edge_ion = edge_ion)

    L = L_sys-2*edge_ion
    half_L = int((L-1)/2)
    center_L = int((L_sys-1)/2)
    mg_reg, mg_reg_err, x1 = get_mag_from_file(rid = rid, date_str = date_str, pmts = [center_L+i for i in range(-half_L,half_L+1)])
    mg_stg, mg_stg_err, x2 = get_mag_from_file(rid = rid, date_str = date_str, pmts = [center_L+i for i in range(-half_L,half_L+1)], stag_flag= True)

    mg_all = [mg_reg,mg_reg_err,mg_stg, mg_stg_err]
    np.savetxt(save_path +'/Magnetization.txt', mg_all)

def save_correlation_reg(rid = "", date_str = [], appendix_str = "", edge_ion = 0):

    save_home = "D:/Lei/dataprocess/Regular/"
    save_path = save_home + rid + appendix_str + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    files = get_file_path_by_date(rid = [rid],date_str=date_str)
    print(files)
    data, x = get_data_from_file(files[0])

    n_pmts, n_shots, n_scans = data.shape
    # data = (data>1).astype(int)

    print(data.shape)
    corr_dj = calculate_delta_j_corr_offset(data,edge_ion)
    f_name = save_path + "Corr_dj.txt"
    np.savetxt(f_name,corr_dj)

    f_name = save_path + "Scan_xalues.txt"
    np.savetxt(f_name,x)

    y_dim, x_dim = corr_dj.shape


    corr_ij = calculate_all_i_j_corr(data)
    d1, d2 , d3 = corr_ij.shape

    for i_scan in range(d3):
        tmp = np.squeeze(corr_ij[:,:,i_scan])
        f_name = save_path + "Corr_i_j_scan_{0}".format(i_scan)+".txt"
        np.savetxt(f_name,tmp)

    return save_path

def save_correlation(rid = "", date_str = [], appendix_str = "", edge_ion = 0):

    save_home = "D:/Lei/dataprocess/"
    save_path = save_home + rid + appendix_str + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    files = get_file_path_by_date(rid = [rid],date_str=date_str)
    print(files)
    data, x = get_data_from_file(files[0])

    n_pmts, n_shots, n_scans = data.shape
    # data = (data>1).astype(int)

    print(data.shape)
    corr_dj = calculate_connected_delta_j_corr_offset(data,edge_ion)
    f_name = save_path + "Corr_dj.txt"
    np.savetxt(f_name,corr_dj)

    f_name = save_path + "Scan_xalues.txt"
    np.savetxt(f_name,x)

    y_dim, x_dim = corr_dj.shape


    corr_ij = calculate_connected_all_i_j_corr(data)
    d1, d2 , d3 = corr_ij.shape

    for i_scan in range(d3):
        tmp = np.squeeze(corr_ij[:,:,i_scan])
        f_name = save_path + "Corr_i_j_scan_{0}".format(i_scan)+".txt"
        np.savetxt(f_name,tmp)

    return save_path



# packed calibtation of Jij to normalized diagnols



def calibrate_Jij(rid, date_str = [], n_scans = 9, n_pmts = 13, T_estimate_div = [1.4],thresh = 0.25, calib_value = 0.8, flg_update_Jig = False, T_relax = [6], J_mat = {}, save_path = ""):

    def fit_exp_cos(x,y, p0):

        fun = lambda x, a, b, c, e , T:a*np.cos(2*np.pi*x/T + b)*np.exp(-x/e)+c
        popt, pcov = sp.optimize.curve_fit(fun,x,y,p0 = p0,maxfev=5000,bounds= ([-1,0,-0.1,0,0],[1,np.pi,0.1,np.Inf,np.Inf]))
        fun_val = lambda x:fun(x,*popt)
        return popt,pcov,fun_val

    def fit_exp_cos_no_a(x,y, p0):

        fun = lambda x, c, e , T:0.98*np.cos(2*np.pi*(x+80)/T)*np.exp(-(x+80)/e)+c
        popt, pcov = sp.optimize.curve_fit(fun,x,y,p0 = p0,maxfev=5000,bounds= ([-0.1,0,0],[0.1,np.Inf,np.Inf]))
        fun_val = lambda x:fun(x,*popt)
        return popt,pcov,fun_val


    def fit_simple_cos(x,y, p0):

        fun = lambda x, a, b, c, T:a*np.cos(x/T + b)+c
        popt, pcov = sp.optimize.curve_fit(fun,x,y,p0 = p0,maxfev=5000,bounds= ([-1,0,-0.05,0],[1,np.pi,0.05,np.Inf]))
        fun_val = lambda x:fun(x,*popt)
        return popt,pcov,fun_val


    def guess_T(data, t_step = 1):
        data_fft = np.fft.fft(data)
        pow = np.abs(data_fft)**2
        ind = int(len(pow)/2)
        pow = pow[0:ind]
        ind_max = np.argwhere(pow == np.max(pow))
        ind_max[0]
        if ind_max >0:
            return len(data)/ind_max*t_step
        else:
            return len(data)*2*t_step

    # rids= ["294813"]

    # rids= ["294817"]
    files = get_file_path_by_date(rid = [rid], date_str = date_str)

    p,x =  get_population_from_file(files[0])
    n_ions, n_points = p.shape
    n_segments = int(n_points/n_scans)

    print(n_points)

    if len(T_estimate_div) == 1:
        T_div = np.ones(n_segments)*T_estimate_div[0]
    else:
        T_div = np.array(T_estimate_div)


    if len(T_relax) == 1:
        Tau = np.ones(n_segments)*T_relax[0]
    else:
        Tau = np.array(T_relax)


    T_fit = []

    n_row = int(np.sqrt(n_segments))

    if n_row == 1:
        n_row = 2

    n_col = int(np.floor(n_segments/n_row))+1

    if n_col == 1:
        n_col = 2

    fig, ax = plt.subplots(n_row,n_col)
    fig.set_size_inches((n_row*8,n_col*3))
    ax = [x for xs in ax for x in xs]

    plt.figure(figsize = (10,6))
    cmp = plt.get_cmap("plasma")
    # n_estimate = T_estimate_div
    thresh_contrast = thresh


    for i in range(n_segments):
        n_estimate = T_div[i]

        x_data = x[n_scans*i:n_scans*(i+1)]
        p_seg = p[:,n_scans*i:n_scans*(i+1)]
        ind_col = np.argmax(p_seg,axis =1)
        ind_row = np.argwhere([p_seg[j][ind_col[j]]>thresh for j in range(n_pmts)])
        ind_row = np.squeeze(ind_row)

        if len(ind_row) != 2:
            print("Error, too many or little pmts available in the calibration:{0}".format(ind_row))
            break

        y_data1 =  np.squeeze(p_seg[ind_row[0]])
        y_data2 =  np.squeeze(p_seg[ind_row[1]])
        y_data = y_data1 - y_data2
        # y_data = 2*y_data2-1
        # y_data =-(y_data1 + y_data2)+1
        # y_data =-2*y_data1+1

        ax[i].plot(x_data,y_data,'--o',color = cmp(i/n_segments))

        ax[i].set_ylim([-1,1])
        ax[i].grid(True)
        ax[i].set_title("PMT_{0}".format(ind_row))

        if y_data[1]>y_data[0]:
            a0 = 1.0
            b0 = 0
        else:
            a0 = -1
            b0 = 0

        t_step = x_data[1]-x_data[0]
        T0 = guess_T(y_data,t_step)/n_estimate
        c0 = 0.0
        # e0 = 1*T0
        e0 = Tau[i]
        # p0 = [a0,b0,c0,e0,T0]
        # param, pcov, fun = fit_exp_cos(x_data,y_data,p0)
        p0 = [c0,e0,T0]
        param, pcov, fun = fit_exp_cos_no_a(x_data,y_data,p0)
        # p0 = [a0,b0,c0,1]
        # param, pcov, fun = fit_simple_cos(x_data,y_data,p0)

        T_fit.append([param[-1],param[0], param[-2]])


        # x_plot = np.linspace(x_data[0],np.max([x_data[-1],1*param[-1]]),100)
        x_plot = np.linspace(x_data[0],1.2*x_data[-1],100)

        y_plot = fun(x_plot)
        ax[i].plot(x_plot,y_plot,'-',color = cmp(i/n_segments))

        # print([param[-1],param[0], param[-2]])
        sigma = np.sqrt(np.diagonal(pcov))
        # print('one cycle time')
        # print([param[-1],sigma[-1]])
        # print('Gamma')
        # print([param[-2],sigma[-2]])
        # print('amplitude')
        # print([param[0],sigma[0]])
        print(param)
        print(sigma)
        if flg_update_Jig:
            J_mat[(ind_row[0],ind_row[1])] = [param[-1],param[-2],sigma[-1],sigma[-2]]
            J_mat[(ind_row[1],ind_row[0])] = [param[-1],param[-2],sigma[-1],sigma[-2]]
        # print(J_mat)
        else:
            print(rid + ": calibrated correction ratio: {0}".format(param[-1]/calib_value))
            if save_path:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                data_save = np.array([x_data,y_data1,y_data2])
                np.savetxt(save_path+"PMT_{0}.txt".format(ind_row),data_save)
    return J_mat
    plt.show()


def Calibrate_stark_shift_simple(rid, date_str = [], n_scans = 9, amp = [],T_estimate_div = 1.2, thresh = 0.1, pmt = 0):

    def fit_exp_cos(x,y, p0):

        fun = lambda x, a, b, c, e , T:a*np.sin(2*np.pi*x/T + b)*np.exp(-x/e)+c
        popt, pcov = sp.optimize.curve_fit(fun,x,y,p0 = p0,maxfev=5000,bounds= ([-0.5,0,0.499,0,0],[0.5,np.pi,0.501,np.Inf,np.Inf]))
        fun_val = lambda x:fun(x,*popt)
        return popt,pcov,fun_val

    def guess_T(data, t_step = 1):
        data_fft = np.fft.fft(data)
        pow = np.abs(data_fft)**2
        ind = int(len(pow)/2)
        pow = pow[0:ind]
        ind_max = np.argwhere(pow == np.max(pow))
        ind_max[0]
        if ind_max >0:
            return len(data)/ind_max*t_step
        else:
            return len(data)*2*t_step

    # rids = ["294899"]
    files = get_file_path_by_date(rid = rid, date_str = date_str)

    p,x =  get_population_from_file(files[0])
    n_sample = len(amp)
    # amp = [0,0.25,0.5,0.75,1]
    T_fit = []

    n_row = np.max([int(np.sqrt(n_sample)),2])
    n_col = np.max([int(np.floor(n_sample/n_row))+1,2])

    fig, ax = plt.subplots(n_row,n_col)
    fig.set_size_inches((n_row*8,n_col*3))
    ax = [x for xs in ax for x in xs]

    plt.figure(figsize = (10,6))
    cmp = plt.get_cmap("viridis")
    n_estimate = T_estimate_div
    thresh_contrast = thresh

    for i in range(n_sample):
        x_data = x[n_scans*i:n_scans*(i+1)]


        y_data = np.squeeze(p[pmt])
        y_data = y_data[n_scans*i:n_scans*(i+1)]


        p_seg = p[:,n_scans*i:n_scans*(i+1)]
        ind_col = np.argmax(p_seg,axis =1)
        ind_row = np.argwhere([p_seg[j][ind_col[j]]>0.3 for j in range(n_pmts)])
        ind_row = np.squeeze(ind_row)

        # if len(ind_row) != 1:
        #     print("Error, too many or little pmts available in the calibration:{0}".format(ind_row))
        #     break

        y_data =  np.squeeze(p_seg[ind_row])


        ax[i].plot(x_data,y_data,'--o',color = cmp(i/n_sample))
        ax[i].set_ylim([0,1])
        ax[i].grid(True)
        ax[i].set_title("amp = {0}".format(amp[i]))

        y_con = np.max(y_data)-np.min(y_data)
        if y_con <= thresh_contrast:
            T_fit.append([0,0,0])
            print("Low contrast, no fit")
            print([0,0,0])
            continue

        if y_data[1]>y_data[0]:
            ind = np.argwhere(y_data==np.max(y_data))
            ind = ind[0]
            a0 = np.max(y_data) - 0.5
            b0 = 0

            # print("fit_sign_positive")
        else:
            ind = np.argwhere(y_data==np.min(y_data))
            ind = ind[0]
            a0 = np.min(y_data) - 0.5
            b0 = 0
            # print("fit_sign_negative")

        t_step = x_data[1] - x_data[0]
        T0 = guess_T(y_data,t_step)/n_estimate
        # T0 = 0.5/np.abs(x_data[ind]-0.5)*n_estimate
        c0 = 0.5
        e0 = 1*T0
        p0 = [a0,b0,c0,e0,T0]
        param, pcov, fun = fit_exp_cos(x_data,y_data,p0)
        T_fit.append([param[-1],param[0], param[-2]])

        x_plot = np.linspace(x_data[0],np.max([x_data[-1],1*param[-1]]),100)
        y_plot = fun(x_plot)
        ax[i].plot(x_plot,y_plot,'-',color = cmp(i/n_sample))

        # print([param[-1],param[0], param[-2]])

    plt.show()

    # fix, ax = plt.subplots(2,1,figsize=(10,10))
    T_fit = np.array(T_fit)
    # ax[0].plot(amp, T_fit[:,0],'o-k',fillstyle = "none")
    # ax[0].set_xlabel("Amp")
    # ax[0].set_ylabel("T_2PI (ms)")
    # ax[0].set_title(rid)
    # ax[0].grid(True)
    f_shift = [1000/T_fit[i,0]*np.sign(T_fit[i,1]) for i in range(n_sample)]
    # ax[1].plot(amp,f_shift,'o-k',fillstyle = "none")
    # ax[1].set_xlabel("Amp")
    # ax[1].set_ylabel("f_shift Hz")
    # ax[1].grid(True)

    # plt.show()
    print(f_shift)
    return f_shift


def list_exp_params(rid, date_str = []):
    f_path = get_file_path_by_date(rid = [rid],date_str = date_str)
    file = h5.File(f_path[0])
    print("Keys in datasets: ---------------------->>")
    print(file["datasets"].keys())

    print("Keys in archive: ---------------------->>")
    print(file["archive"].keys())

    print("Experiment specific data is in key expid: ---------------------->>")

def get_exp_params(rid, date_str = [], param_key1 = ["datasets"],param_key2 = [""]):
    f_path = get_file_path_by_date(rid = [rid],date_str = date_str)
    file = h5.File(f_path[0])
    data = {}

    for k1 in param_key1:

        if k1 == "expid":
            data[k1] = str(file[k1][...])
        else:
            for k2 in param_key2:
                data[k1+"."+k2] = file[k1][k2][...]

    return data

def get_thres_counts(rid,date):
    file_name=get_file_path_by_date(rid=rid,date_str=date)
    f=h5.File(file_name[0])
    tmp=[k for k in f['datasets'].keys()]
    data_name=dict([(x.split('.')[-1],x) for x in tmp])
    y_name=data_name['raw_counts']
    x_name=data_name['x_values']
    raw_counts=f['datasets'][y_name][...]
    ## taking data only for the last time point
    raw_counts=raw_counts[:,:,-1]
    print(raw_counts[0,:])
    x_vals=f['datasets'][x_name][...]
    thresh_counts=raw_counts.copy()
    thresh_counts[raw_counts > 1] = 1
    thresh_counts[raw_counts <= 1] = 0
    return thresh_counts

def get_exp_hist(rid, date, str_len):
    thresh_counts=get_thres_counts(rid,date)
        # Turn the (nq, nshot, nbatch) into (nq, nshot*nbatch, 1) dimension
    bin_data = thresh_counts.reshape(thresh_counts.shape[0], thresh_counts.shape[1])
    npmt,nshots=bin_data.shape
    str_len=str_len
    qubit_to_ion_map=np.arange(int((-str_len)/2),int((str_len)/2)+1)
    nq = len(qubit_to_ion_map)
    ion_index_to_pmt_vector_index = dict(list(zip(range(-int(npmt/2),int(npmt/2)+1),range(npmt))))
    filtered_bin_data = np.zeros((nq,nshots))
    bit_flip_vec=np.zeros(nshots)
    dec_state_vec=np.zeros(nshots)

    for iq in range(nq):
        filtered_bin_data[iq,:] = bin_data[ion_index_to_pmt_vector_index[qubit_to_ion_map[iq]],:]
    for shot in range(nshots):
        bit_flip_vec[shot]=np.sum(filtered_bin_data[:,shot])
        for ion in range(nq):
            dec_state_vec[shot]=dec_state_vec[shot]+filtered_bin_data[ion,shot]*2**(nq-1-ion)
    bit_flips, counts = np.unique(bit_flip_vec, return_counts=True)
    dec_states,num_counts=np.unique(dec_state_vec,return_counts=True)
    counts = np.array(counts)/len(bit_flip_vec)
    num_counts=np.array(num_counts)/len(dec_state_vec)
    # #         #convert to native number types for yaml dump
    bit_flips = [int(bt) for bt in bit_flips]
    probs = [float(ct) for ct in counts]

    dec_states = [int(ds) for ds in dec_states]
    probs_dec = [float(ct) for ct in num_counts]

    hist1 = dict(zip(bit_flips, probs))
    hist2 = dict(zip(dec_states, probs_dec))
    return hist1, hist2

def get_thres_counts(rid,date):
    file_name=get_file_path_by_date(rid=rid,date_str=date)
    f=h5.File(file_name[0])
    tmp=[k for k in f['datasets'].keys()]
    data_name=dict([(x.split('.')[-1],x) for x in tmp])
    y_name=data_name['raw_counts']
    x_name=data_name['x_values']
    raw_counts=f['datasets'][y_name][...]
    ## taking data only for the last time point
    # print(raw_counts.shape)
    # raw_counts=raw_counts[:,:,-1]
    x_vals=f['datasets'][x_name][...]
    thresh_counts=raw_counts.copy()
    thresh_counts[raw_counts > 1] = 1
    thresh_counts[raw_counts <= 1] = 0
    return thresh_counts

def get_thres_counts_byStrLen(rid, date, str_len):
    thresh_counts=get_thres_counts(rid,date)
        # Turn the (nq, nshot, nbatch) into (nq, nshot*nbatch, 1) dimension
    bin_data = thresh_counts.reshape(thresh_counts.shape[0], thresh_counts.shape[1])
    npmt,nshots=bin_data.shape
    qubit_to_ion_map=np.arange(int((-str_len)/2),int((str_len)/2)+1)

    nq = len(qubit_to_ion_map)
    ion_index_to_pmt_vector_index = dict(list(zip(range(-int(npmt/2),int(npmt/2)+1),range(npmt))))
    filtered_bin_data = np.zeros((nq,nshots))
    bit_flip_vec=np.zeros(nshots)
    dec_state_vec=np.zeros(nshots)

    for iq in range(nq):
        filtered_bin_data[iq,:] = bin_data[ion_index_to_pmt_vector_index[qubit_to_ion_map[iq]],:]

    return filtered_bin_data

def histToBinned(str_len, hist):
    bitflips = np.array([np.array([int(Str) for Str in list(np.binary_repr(i))]).sum() for i in range(2**str_len)])
    flip_probs = np.zeros(str_len+1)
    for i in range(2**str_len):
        flip_probs[bitflips[i]] += hist[i]
    return flip_probs

def kinkNumberfromDecState(dec_states,str_len):
    dec2bin=[list(f"{state:b}") for state in dec_states]
    string_w_edge=[]
    for state in dec2bin:
        while len(state)<str_len:
            state=['0']*(str_len-len(state))+state
        state=['0']+state+['0']
        string_w_edge.append(state)
    all_states=np.zeros((2**str_len,str_len+2))
    N,nq=np.shape(string_w_edge)
    tot_kink=[]
    kink_density=[]
    for i in range(N):
        all_states[i:]=[int(x) for x in string_w_edge[i]]
        kink=0
        for j in range(nq-1):
            if all_states[i,j]!=all_states[i,j+1]:
                kink+=1
        tot_kink.append(kink)
        kink_density.append(kink/(str_len+1))
    return tot_kink,kink_density


def calibrate_Bx_freq_with_amp(Bx_amp,meas_frequencies,target_frequency):
    num_ions=np.shape(meas_frequencies)[1]
    cmp = plt.get_cmap('viridis')
    Bx_calib_w_amp=lambda x, a,b: a*x+b
    slope=[]
    intercept=[]
    for i in range(num_ions):
        plt.plot(Bx_amp,meas_frequencies[:,i],'o',color=cmp(i/num_ions))
        popt, pcov = sp.optimize.curve_fit(Bx_calib_w_amp,Bx_amp,meas_frequencies[:,i])
        x_fit=np.linspace(0,Bx_amp,100)
        y_fit = Bx_calib_w_amp(x_fit, *popt)
        plt.plot(x_fit,y_fit,'--',color=cmp(i/num_ions))
        slope.append(popt[0])
        intercept.append(popt[1])
    plt.xlabel('Bx_amp (e-3)')
    plt.ylabel('frequency (kHz)')
    fig,ax=plt.subplots(1,2)
    fig.set_size_inches(12,5)
    ax[0].plot(slope,'o-r')
    ax[1].plot(intercept,'o-b')
    ax[0].legend(['slope'])
    ax[1].legend(['intercept'])
    print('slope',slope)
    print('intercept',intercept)
    amps=[]
    phases=np.zeros(num_ions)
    Bx_amp_from_freq= lambda x,intercept,slope: (x-intercept)/slope
    for i in range(num_ions):
        amp_val=Bx_amp_from_freq(np.abs(target_frequency[i]),intercept[i],slope[i])
        if amp_val<0.5:
            amps.append(0)
        else:
            amps.append(Bx_amp_from_freq(np.abs(target_frequency[i]),intercept[i],slope[i]))
        if target_frequency[i]<0:
            phases[i]=1
    print('amplitudes',list(np.around(amps,decimals=3)))
    print('phases',list(phases))
    plt.figure()
    plt.plot(target_frequency,'o-')
    plt.xlabel('ion number')
    plt.ylabel('freq(kHz)')
    plt.legend(['Theory targets','Expt. measured'])

    return slope, intercept
