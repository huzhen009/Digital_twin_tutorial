# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:48:32 2020

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
from scipy.stats import norm
import copy

def GP_training(Xtrain,Y_train, L0,K_option):
    """
    Training of a GPR model using data
    Parameters
    ----------
    Xtrain : Training data of inputs
    Y_train : Training data of output
    L0 : Initial point for GPR length scale
    K_option : Option of GPR kernel:0-->RBF kernel; 1-->Matern kernel; Others-->Rational_quadratic

    Returns
    -------
    Object of trained GPR model

    """
    nInputs=len(Xtrain[0,:]) # Number of input variables
    L_bounds=(1e-5,1e3)
    if K_option==0:
        RBF_length0=[L0]
        RBF_length_bounds=[L_bounds]
        for i in range(nInputs-1):
            RBF_length0.append(L0)
            RBF_length_bounds.append(L_bounds)
        kernel_RBF = RBF(RBF_length0, RBF_length_bounds)
        kernel_Con=C_ker(L0, L_bounds)
        kernel_noise=WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        kernal_prod=kernel_Con*kernel_RBF+kernel_noise
    elif K_option==1:
        ############### Constant Kernel##############
        kernel_Con=C_ker(L0, L_bounds)
        ############### Matern Kernel##############
        Matern_length0=[L0]
        Matern_length_bounds=[L_bounds]
        for i in range(nInputs-1):
            Matern_length0.append(L0)
            Matern_length_bounds.append(L_bounds)
        kernel_Matern = Matern(Matern_length0, Matern_length_bounds, nu=1.5)
        kernel_noise=WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        kernal_prod=kernel_Con*kernel_Matern+kernel_noise
    else:
        ############### RationalQuadratic  Kernel##############
        Rational_length0=[L0]
        Rational_length_bounds=[L_bounds]
        for i in range(nInputs-1):
            Rational_length0.append(L0)
            Rational_length_bounds.append(L_bounds)
        kernel_noise=WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        kernel_Rational = RationalQuadratic(Rational_length0, Rational_length_bounds,alpha=1.5)
        kernal_prod=kernel_Rational+kernel_noise
    gp_GPML = GaussianProcessRegressor(kernel=kernal_prod, alpha=0,n_restarts_optimizer=2,normalize_y=True)
    gp_GPML.fit(Xtrain, Y_train) 
    return gp_GPML

def Training_data_process(Y_all_data,N_steps):
    """
    Process load history data into format that can be used by the GPR model
    ----------
    Parameters
    ----------
    Y_all_data : Load history data
    N_steps : Number of lags in the NARX model

    Returns
    -------
    X_train : Processed data of the inputs
    Y_train : Processed data of the output
    """
    Ne=len(Y_all_data)
    X_train=[]
    Y_train=[]
    for i in range(Ne):
        Y_temp=Y_all_data[i]
        for j in range(len(Y_temp)-N_steps-1):
            X_train_temp=Y_temp[j:j+N_steps]
            X_train_temp=np.reshape(X_train_temp,(len(X_train_temp),))
            Y_train_temp=Y_temp[j+N_steps]
            X_train.append(X_train_temp)
            Y_train.append(Y_train_temp)
    X_train=np.asarray(X_train)
    Y_train=np.asarray(Y_train)
    return X_train,Y_train

# Prediction using GP-NARX
def GP_dynamic(GP_model,Y0,e_std, N_MCS, N):
    """
    Perform dynamic load condition prediction for given intial 
    condition using GPR model
    Parameters
    ----------
    GP_model : GPR model to model the load dynamics
    Y0 : Initial condition
    e_std: standard deviation of process noise
    N : Number of time steps to predict
    Returns
    -------
    Y_all_mean : predicted mean value
    Y_all_5 : predicted 5-th percentile value
    Y_all_95 : predicted 95-th percentile value
    Y_pre_sample: predicted load samples
    """
    Y_simu_mean=[]
    Y_simu_5=[]
    Y_simu_95=[]
    Y0_temp=copy.copy(Y0)
    Y0_temp=np.reshape(Y0_temp,(1,len(Y0_temp)))
    Y_previous=np.dot(np.ones(shape=(N_MCS,1)),Y0_temp)
    Y_pre_sample=np.zeros(shape=(N_MCS,N))
    for i in range(N):
        NARX_prediction,NARX_std=GP_model.predict(Y_previous,return_std=True)
        y_temp=np.random.normal(NARX_prediction[:,0],np.sqrt(NARX_std**2+e_std**2)) # Account for uncertainty
        Y_pre_sample[:,i]=y_temp
        # Update Y0
        Y_previous_new=copy.copy(Y_previous)
        Y_previous_new[:,0:-1]=Y_previous[:,1:] 
        Y_previous_new[:,-1]=y_temp
        Y_previous=copy.copy(Y_previous_new)
        # Obtain the percentile values
        Y_simu_mean.append(np.mean(y_temp))
        Y_simu_5.append(np.percentile(y_temp,5))
        Y_simu_95.append(np.percentile(y_temp,95))
    Y_simu_mean=np.asarray(Y_simu_mean)
    Y_simu_mean=np.reshape(Y_simu_mean,(len(Y_simu_mean),1))
    # Y0=np.reshape(Y0,(len(Y0),1))
    # Y_all_mean=np.vstack((Y0,Y_simu_mean)) # Put all data together
    
    Y_simu_5=np.asarray(Y_simu_5)
    Y_simu_5=np.reshape(Y_simu_5,(len(Y_simu_5),1))
    # Y_all_5=np.vstack((Y0,Y_simu_5)) # Put all data together
    
    
    Y_simu_95=np.asarray(Y_simu_95)
    Y_simu_95=np.reshape(Y_simu_95,(len(Y_simu_95),1))
    # Y_all_95=np.vstack((Y0,Y_simu_95)) # Put all data together
    return Y_simu_mean,Y_simu_5,Y_simu_95,Y_pre_sample

def Crack_growth(a0,c,m,load):
    """
    Hypothetical crack growth model. This model needs to be replaced 
    with real crack growth model developed following certain crack growth laws
    ----------
    Parameters
    ----------
    a0 : Initial crack length
    c : Crack model parameter 1
    m : Crack model parameter 2
    load : External load
    Returns
    -------
    a : Predicted crack growth at the next time step

    """
    n_sample=a0.size
    a=a0+c*(4*np.sin((load/100)**2))*m+np.abs(np.random.normal(0,0.05,(n_sample,1)))
    return a

def Strain_model(load,a,theta):
    """
    Hypothetical strain analysis model. This model needs to be replaced with 
    stress model constructed from finite element analysis. If the FEA model is 
    computationally expensive, this model needs to be substituted with a surrogate
    model.
    ---------
    Parameters
    ----------
    load : Load on the structure
    a : Crack length
    theta : Parameter of the structural analysis model
    Returns
    -------
    Stress : Predicted strain response
    """
    Stress=0.1*np.sin(a*theta)*np.cos(load)
    return Stress

def Diagnostics(a0_sample,c_sample,m_sample,load_temp,Strain_temp,theta_sample,obs_std):
    """
    Estimate the posterior distribution of the damage state based on 
    observed strain and load conditions.
    Parameters
    ----------
    a0_sample : sample of a0
    c_sample : sample of c
    m_sample : sample of m
    load_temp : measured load data
    Strain_temp : measured strain data
    theta_sample : sample of theta
    obs_std : standard deviation of observation noise
    
    Returns
    -------
    a_post : Posterior distribution of damage state

    """
    nprior=len(c_sample)
    a_prior=Crack_growth(a0_sample,c_sample,m_sample,load_temp) # Prior sample of a at current time step
    Strain_prior=Strain_model(load_temp,a_prior,theta_sample) # Prior sample of strain at current time step
    # Compute likelihood function of the particles
    Likelihood_prior=norm.pdf(Strain_temp, loc=Strain_prior, scale=obs_std) # Likelihood of the prior samples
    
    weight_Bayes=Likelihood_prior/sum(Likelihood_prior)
    ########################## Resampling #####################
    cumulative_sum = np.cumsum(weight_Bayes)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes_post = np.searchsorted(cumulative_sum, np.random.uniform(0,1,nprior))
    a_post=a_prior[indexes_post,0]
    
    return a_post

def Prognostics(a0_sample,c_sample,m_sample,GP_load,Load_init_temp,Load_para,G_threhold, nprior,Nt_prognostics):
    """
    Perform failure prognostics based on load forcasting and degradation modeling.
    Parameters
    ----------
    a0_sample : Samples of damage state
    c_sample : Samples of degradation model parameter c
    m_sample : Samples of degradation model parameter m
    GP_load : GPR model trained for load forecasting
    Load_init_temp : Initial condition for load forecasting
    Load_para : Parameters for load forecasting
    G_threhold : Failure threshold for prognostics
    nprior : Number of prior samples
    Nt_prognostics : Number of time steps for failure forecasting
    Returns
    -------
    RUL : Predicted RUL
    a_simu : Samples of forecasted damage state

    """
    
    _, _, _,load_samples=GP_dynamic(GP_load,Load_init_temp,Load_para['Noise_std'],nprior,Nt_prognostics)
    a_prog=np.zeros(shape=(nprior,1))
    for i in range(Nt_prognostics):
        load_temp=np.reshape(load_samples[:,i],(len(load_samples[:,i]),1)) # Load at current time step
        a_prog_temp=Crack_growth(a0_sample,c_sample,m_sample,load_temp)
        if i==0:
           a_prog=a_prog_temp
        else:
           a_prog=np.hstack((a_prog,a_prog_temp))
        a0_sample=a_prog_temp
    ############################# Obtain RUL estimate #########################
    a_simu=np.transpose(a_prog) # Simulated crack length
    # Obtain the possible failure time from the samples;
    m,n=np.shape(a_simu)
    Failure_time=[]
    for isample in range(n):
        Failure_time_temp=np.argmax(a_simu[:,isample]>G_threhold)
        if np.max(a_simu[:,isample])<G_threhold: # No failure
            Failure_time.append(Nt_prognostics) # No failure
        else:
            Failure_time.append(Failure_time_temp) # Has failure
    Failure_time=np.asarray(Failure_time)
    RUL=Failure_time
    return RUL,a_simu
    