# import all the useful lib in python
import h5py as h5
from matplotlib import pyplot as plt
from matplotlib import artist as art
import numpy as np
import scipy as sp
#ring as AgCluster
#import pandas as pd
import pathlib
import os
import datetime as dt
import time
import json
import io
from IPython.display import clear_output

from h5analysis import *


def fit_gaussian_new(x,y, p0 = None, maxfev = 5000):
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

    popt, pcov = sp.optimize.curve_fit(offset_gaussian, x,y, p0 = [a0,x0,b0,sigma], maxfev = maxfev)

    fun_evaluate = lambda x: offset_gaussian(x,*popt)
    return popt, pcov, fun_evaluate

def get_sscooling_data(rid,date_str):

    file = get_file_path_by_date([rid],date_str)
    file = h5.File(file[0])

    tmp = [k for k in file["datasets"].keys()]

    data_name = dict([(x.split(".")[-1],x) for x in tmp])
    # y_name = data_name["raw_counts"]
    x_name = data_name["x_values"]
    ssc_name = data_name["sscooling"]

    ssc_data = file["datasets"][ssc_name][...]
    x_vals = file["datasets"][x_name][...]

    return ssc_data, x_vals

def save_data_h5(rid,date_str,B_field, config_num, measure_ax,folder):

    file = get_file_path_by_date([rid],date_str)
    data,x = get_data_from_file(file[0])

    tunit = 10.769*10**-3
    tunit = tunit/8

    dx, = x.shape
    t = np.linspace(0,tunit,dx)

    file_name = folder + rid + "_data.h5"

    hf = h5.File(file_name,'w')

    hf.create_dataset("Data",data = data)
    hf.create_dataset("X_val",data = t)
    hf.create_dataset("B_field",data = B_field)
    hf.create_dataset("Configuration",data = config_num)
    hf.create_dataset("Measurement basis",data = measure_ax)

    hf.close()

    print("Compressed data for B = {0} and config = {1} saved to ".format(B_field,config_num) + file_name)
def calibrate_Jij_new(rid, date_str = [], n_scans = 9, n_pmts = 13, T_estimate_div = [1.4],thresh = 0.25, calib_value = 0.8, flg_update_Jig = False, T_relax = [6], J_mat = {}, save_path = ""):

    def fit_exp_cos(x,y, p0):

        fun = lambda x, a, b, c, e , T:a*np.cos(2*np.pi*x/T + b)*np.exp(-x/e)+c
        popt, pcov = sp.optimize.curve_fit(fun,x,y,p0 = p0,maxfev=5000,bounds= ([-1,0,-0.1,0,0],[1,np.pi,0.1,np.Inf,np.Inf]))
        fun_val = lambda x:fun(x,*popt)
        return popt,pcov,fun_val

    def fit_exp_cos_no_a(x,y, p0):

        # fun = lambda x, c, e , T:0.98*np.cos(2*np.pi*(x)/T)*np.exp(-(x)/e)+c
        fun = lambda x, c, e , T:1*np.cos(2*np.pi*(x)/T)*np.exp(-(x)/e)+c
        popt, pcov = sp.optimize.curve_fit(fun,x,y,p0 = p0,maxfev=5000,bounds= ([-0.1,0,0],[0.1,np.Inf,np.Inf]))
        fun_val = lambda x:fun(x,*popt)
        return popt,pcov,fun_val

    def fit_simp_cos_no_a(x,y, p0):

        # fun = lambda x, c, e , T:0.98*np.cos(2*np.pi*(x)/T)*np.exp(-(x)/e)+c
        fun = lambda x, c, e , T:0.98*np.cos(2*np.pi*(x)/T)*np.exp(-(x)/e*0)+c
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
        ind_max = ind_max[0]
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

    print('number of scan points',n_points)

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
    # if n_row == 1:
    #     n_row = 2

    # n_col = int(np.floor(n_segments/n_row))+1
    n_col = int(np.floor(n_segments/n_row))


    # if n_col == 1:
    #     n_col = 2
    print('row', n_row)
    print('col',n_col)
    fig, ax = plt.subplots(n_row,n_col)
    print('ax',ax)
    fig.set_size_inches((n_row*8,n_col*3))
    try:
        ax = [x for xs in ax for x in xs]
    except:
        ax=[ax]
    # ax = [xs for xs in ax]

    # print(ax)


    plt.figure(figsize = (10,6))
    cmp = plt.get_cmap("plasma")
    # n_estimate = T_estimate_div
    thresh_contrast = thresh


    for i in range(n_segments):
        n_estimate = T_div[i]

        x_data = x[n_scans*i:n_scans*(i+1)]

        x_data = x_data - x_data[0]
        p_seg = p[:,n_scans*i:n_scans*(i+1)]
        p_seg_mean = np.repeat(np.mean(p_seg,axis = 1,keepdims=True),p_seg.shape[1],axis = 1)
        # print(p_seg.shape)
        # print(p_seg_mean.shape)
        p_seg_fluct = np.abs(p_seg - p_seg_mean)


        ind_col = np.argmax(p_seg_fluct,axis =1)
        ind_row = np.argwhere([p_seg_fluct[j][ind_col[j]]>thresh for j in range(n_pmts)])
        ind_row = np.squeeze(ind_row)
        print(ind_row)

        if len(ind_row) != 2:
            print("Error, too many or little pmts available in the calibration:{0}".format(ind_row))
            break

        y_data1 =  np.squeeze(p_seg[ind_row[0]])
        y_data2 =  np.squeeze(p_seg[ind_row[1]])
        y_data = y_data1 - y_data2 ##if ion 0 starts from 1 and ion 1 starts from 0

        # y_data = 1-(y_data1 + y_data2) ###If both the ions start from 0
        # y_data = -y_data+1

        print('ydata1',y_data1)
        print('ydata2',y_data2)
        print('ydata',y_data)
        # if y_data[0]<0:
        #     y_data = -y_data

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
        T0 = [6]
        # T0 = T0[0][0]

        # print("Guess T0 is {0}".format(T0))
        c0 = 0.0
        # e0 = 1*T0
        e0 = Tau[i]
        # p0 = [a0,b0,c0,e0,T0]
        # param, pcov, fun = fit_exp_cos(x_
        # data,y_data,p0)
        p0 = [c0,e0,T0[0]]
        # print("Fitting")
        # print(x_data.shape, y_data.shape, p0)
        param, pcov, fun = fit_exp_cos_no_a(x_data,y_data,p0)
        # param, pcov, fun = fit_simp_cos_no_a(x_data,y_data,p0)
        # p0 = [a0,b0,c0,1]
        # param, pcov, fun = fit_simple_cos(x_data,y_data,p0)

        T_fit.append([param[-1],param[0], param[-2]])

        # x_plot = np.linspace(x_data[0],np.max([x_data[-1],1*param[-1]]),100)
        x_plot = np.linspace(x_data[0],1.2*x_data[-1],100)

        y_plot = fun(x_plot)
        ax[i].plot(x_plot,y_plot,'-',color = cmp(i/n_segments))

        # print([param[-1],param[0], param[-2]])
        sigma = np.sqrt(np.diagonal(pcov))
        print('one cycle time [val,sigma]')
        print([param[-1],sigma[-1]])
        print('Gamma [val,sigma]')
        print([param[-2],sigma[-2]])
        print('(1/T)*tau',param[-2]/param[-1])
        # print('amplitude')
        # print([param[0],sigma[0]])
        # print(param)
        # print(sigma)
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
