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
Load_all_data=load_obj('Load_history.pkl')
# %% ==================== Processing Training Data ============================
N_step_max=15 # Maximum number of lags for the NARX model
# Obtain training data
X_train_all,Y_train_all=my.Training_data_process(Load_all_data,N_step_max)
# Split the data into training and test
X_train, X_test, y_train, y_test = train_test_split(X_train_all, Y_train_all, test_size=0.33, random_state=42)
# %%  ========================= Training of GP ================================
K_option=1 # Use Matern kernel
All_lags=np.linspace(1,N_step_max,N_step_max) # All possible lags-->MCS method is used to optimize the number of lags
MSE_y=np.zeros(shape=(len(All_lags),)) # Initialize MSE
for i in range(len(All_lags)):
    print('Training GPR model for lag scenario '+str(i)+'/'+str(len(All_lags)))
    N_step=int(All_lags[i])
    X_train_temp=X_train[:,-N_step:]
    X_test_temp=X_test[:,-N_step:]
    GP_model=my.GP_training(X_train_temp,y_train,2,K_option)
    # Check accuracy using test data
    y_pre=GP_model.predict(X_test_temp)
    MSE_y[i]=mean_squared_error(y_test,y_pre) # Compute MSE of the test data
# Determine the opitmal number of lags
Lag_opt=All_lags[np.argmin(MSE_y)] # Optimal number of lags
plt.figure()
plt.plot(All_lags,MSE_y,'-o')
plt.yscale('log')
plt.grid()
plt.xlabel('Number of lags')
plt.ylabel('MSE')
########################## Obtain the final GPR model #########################
X_train_opt=X_train[:,-int(Lag_opt):]
GP_load=my.GP_training(X_train_opt,y_train,2,K_option)
y_pre_load=GP_load.predict(X_train_opt)
Noise_std_est=np.std(y_pre_load-y_train) # Estimate the process noise standard deviation
Noise_mean_est=np.mean(y_pre_load-y_train) # Estimate the process noise mean
# %% ======================= Predict for the new load condition ===============
Load_test=load_obj('New_load.pkl')
Time_current=150 # Current prediction time
N_mcs=1500
Load_init=Load_test[Time_current-int(Lag_opt):Time_current,0]
Y_GP_mean,Y_GP_5,Y_GP_95,Y_samples=my.GP_dynamic(GP_load,Load_init,Noise_std_est,N_mcs,len(Load_test)-Time_current)

fig, ax = plt.subplots() 
time_x=np.linspace(Time_current,len(Load_test),len(Load_test)-Time_current)
ax.plot(time_x, Load_test[Time_current:], 'k-')
ax.plot(time_x, Y_GP_mean, 'r--')

ax.fill_between(time_x, Y_GP_5[:,0], Y_GP_95[:,0], color='blue', alpha=0.3)
ax.legend(['True mean response','Mean prediction', 'Uncertainty bounds'])
ax.set_xlabel('Time')
ax.set_ylabel('Load')
# Save Load Modeling Results
save_obj(GP_load, 'GP_load')
save_obj({'Lag':Lag_opt,'Noise_std':Noise_std_est}, 'Lag_opt')
    
    
    



    
    
 