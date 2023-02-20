# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:05:25 2020

@author: zhenh
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import gaussian_process
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, RationalQuadratic, ConstantKernel as C_ker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import myfun as my
import sys
import pickle
import pyDOE
import random

# %% Load aircraft load history data
def load_obj(name ):
    with open('obj/' + name, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
Load_test=load_obj('New_load.pkl')
True_crack=load_obj('Crack_true.pkl')
Strain_obs=load_obj('Strain_obs.pkl')
Load_para=load_obj('Lag_opt.pkl')
GP_load=load_obj('GP_load.pkl')
# %% ================== Diagnostics and Prognostics =========================
c_mu=1.2e-3 # Mean of crack growth model parameter c
c_std=1e-4 # Standard deviation (std) of crack growth model parameter c
m_mu=1.7 # Mean of crack growth model parameter m
m_std=0.15 # Standard deviation (std) of crack growth model parameter m
a0_mu=0.14 # Mean of initial crack length
a0_std=0.01 # Std of initial crack length
theta_mu=0.88 # Mean of strain model parameter
theta_std=0.02 # Std of strain model parameter
obs_std=0.0001 # Standard deviation of observation noises
################## Parameters for diagnostics ##################
nprior=2000 # Number of prior samples
Nt_prognostics=500 # Number of time steps for prognostics
a_threhold=8 # Failure threshold of crack length
Failure_true=np.argmax(True_crack>a_threhold)
RUL_true=np.arange(Failure_true,0,-1)
####### Samples of model parameters which are not updated over time ###########
c_sample=np.random.normal(c_mu,c_std,(nprior,1))
m_sample=np.random.normal(m_mu,m_std,(nprior,1))
a0_sample=np.random.normal(a0_mu,a0_std,(nprior,1))
theta_sample=np.random.normal(theta_mu,theta_std,(nprior,1))
###############################################################################
RUL_time_step=10 # Perform RUL prediction every RUL_time_step steps
a_CI=np.zeros(shape=(1,3))
RUL_CI=np.zeros(shape=(1,3))
RUL_true_time=np.zeros(shape=(1,2))
for ite in range(len(Strain_obs[:,0])):
    print (ite)
    sys.stdout.flush()
    Strain_temp=Strain_obs[ite,:] # Strain observation at current time step
    load_temp=Load_test[ite,:] # Measured load at current time step
    ############################## Diagnostics ################################
    a_post=my.Diagnostics(a0_sample,c_sample,m_sample,load_temp,Strain_temp,theta_sample,obs_std)
    # Update a0 samples
    a0_sample=np.reshape(a_post,(len(a_post),1))
    ####### Save the confidence interval of the posteior distribution #########
    a_CI_temp=np.percentile(a_post,(5,50,95))
    if ite==0:
        a_CI=a_CI_temp
    else:
        a_CI=np.vstack((a_CI,a_CI_temp))
    ################################# Prognostics #############################
    if ite>1 and np.mod(ite,RUL_time_step)==0: # Perform RUL analysis
        Load_init_temp=Load_test[ite-int(Load_para['Lag']):ite,0]
        RUL_sample,a_prog=my.Prognostics(a0_sample,c_sample,m_sample,GP_load,\
                                  Load_init_temp,Load_para,a_threhold, nprior,Nt_prognostics)
        if ite==3*RUL_time_step or ite==6*RUL_time_step:
            time_plot=np.linspace(0,ite+1,ite+1) # Time indices for plotting
            time_plot2=np.linspace(ite,ite+Nt_prognostics-1,Nt_prognostics)
            time_plot2=np.reshape(time_plot2,(len(time_plot2),1))
            time_plot_prog=time_plot2*np.ones((1,nprior))
            # Obtain PDF of the RUL sample
            kde = stats.gaussian_kde(RUL_sample)
            RUL_values = np.linspace(RUL_sample.min(), RUL_sample.max(), 100)
            RUL_PDF=kde(RUL_values)
            # Plot curves
            plt.figure()
            plt.plot(time_plot,True_crack[:len(time_plot)],'k-', time_plot,a_CI[:,1],'r-')
            plt.plot(time_plot,a_CI[:,0],'g--', time_plot,a_CI[:,2],'g--')
            plt.plot(time_plot_prog, a_prog,'g-',alpha=0.1)
            plt.plot([1,1.5*Failure_true],[a_threhold,a_threhold],'r-')
            plt.ylim([0,1.3*a_threhold])
            plt.xlim([0,1.5*Failure_true])
            plt.plot(RUL_values+ite,a_threhold+35*RUL_PDF,'m-')
            plt.xlabel('Time')
            plt.ylabel('Crack length (inches)')
        # Obtain RUL confidence interval
        RUL_CI_temp=np.percentile(RUL_sample,(5,50,95))
        RUL_CI=np.vstack((RUL_CI,RUL_CI_temp))
        # Store the true RUL
        RUL_true_time_temp=np.asarray([ite,Failure_true-ite])
        RUL_true_time=np.vstack((RUL_true_time,RUL_true_time_temp))
        if np.sum(RUL_sample)==0:
            break
RUL_true_time=RUL_true_time[1:,:]
RUL_CI=RUL_CI[1:,:]
# %%###########################################################################
#==================================== Plot results============================= 
time_plot=np.linspace(1,len(a_CI[:,0]),len(a_CI[:,0])) # Time indices for plotting
plt.figure()
plt.plot(time_plot,True_crack[:len(time_plot)],'k-', time_plot,a_CI[:,1],'r-')
plt.plot(time_plot,a_CI[:,0],'g--', time_plot,a_CI[:,2],'g--')
plt.legend(['True crack length','Predicted mean','90% confidence interval'])
plt.xlabel('Time')
plt.ylabel('Crack length (inches)')
plt.grid()


plt.figure()
plt.plot(RUL_true_time[:,0],RUL_true_time[:,1],'k-', RUL_true_time[:,0],RUL_CI[:,1],'r-')
plt.plot(RUL_true_time[:,0],RUL_CI[:,0],'g--', RUL_true_time[:,0],RUL_CI[:,2],'g--')
plt.legend(['True RUL','Predicted RUL mean','90% confidence interval'])
plt.ylim([0,1.2*np.max(RUL_true_time[:,1])])
plt.xlabel('Time')
plt.ylabel('RUL')
plt.grid()  
 