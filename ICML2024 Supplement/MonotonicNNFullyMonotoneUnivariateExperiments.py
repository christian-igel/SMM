# # Smooth Monotonic Networks: Experiments on fully monotonic functions
# The results are stored in files, which are read by ``MonotonicNNPaperEvaluate.ipynb``.
# ## General definitions
# Among others, we compare against XGBoost, which can be installed via `pip install xgboost`, and 
# the Hierarchical Lattice Layer, which can be installed via 
# `pip install pmlayer`.

import numpy as np
import random

import torch 
import torch.nn as nn

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
from tqdm.notebook import tnrange

from xgboost import XGBRegressor

from pmlayer.torch.layers import HLattice

from MonotonicNN import SmoothMonotonicNN, MonotonicNN, MonotonicNNAlt
from MonotonicNNPaperUtils import Progress, total_params, fit_torch

from monotonenorm import GroupSort, direct_norm, SigmaNet

prefix = "./"  # prefix for filenames of result files


# ## Univariate experiments 
# Section 4.1 in the manuscript.

T = 21  # number of trials, odd number for having a "median trial"
ls = 75  # lattice points (k in original paper)
ls_small = 35
K = 6  # number of SMM groups, we always use H_k = K
N_train = 100  # number of examples in training data set
N_test = 1000 # number of examples in test data set
sigma = 0.01  # noise level, feel free to vary 
width_small = K
width = K+2

def generate1D(function_name, sigma=0., random=False, xrange=1., N=50):
    if random:
        x = np.random.rand(N) * xrange
        x = np.sort(x, axis=0)
    else:
        xstep = xrange / N
        x = np.arange(0, xrange, xstep)
    match function_name:
        case 'sigmoid10':
            y = 1. /(1. + np.exp(-(x-xrange/2.) * 10.))
        case 'sq':
            y = x**2
        case 'sqrt':
            y = np.sqrt(x)
    y = y + sigma*np.random.normal(0, 1., N)
    return x.reshape(N, 1), y


tasks = ['sq', 'sqrt', 'sigmoid10']
methods = ['monotonic', 'smooth', 'xgboost', 'xgboost_val', 'iso', 'hll', 'lip_small', 'lip']
N_tasks = len(tasks)
N_methods = len(methods)

MSE_train = np.zeros((N_tasks, N_methods, T))
MSE_test = np.zeros((N_tasks, N_methods, T))
MSE_clip = np.zeros((N_tasks, N_methods, T))
R2_train = np.zeros((N_tasks, N_methods, T))
R2_test = np.zeros((N_tasks, N_methods, T))
X_train = np.zeros((N_tasks, T, N_train))
Y_train = np.zeros((N_tasks, T, N_train))
X_test = np.zeros((N_tasks, T, N_test))
Y_test = np.zeros((N_tasks, T, N_test))
O_test = np.zeros((N_tasks, N_methods, T, N_test))
no_params=np.zeros(N_methods)

for trial in tnrange(T):
    for task_id, task in enumerate(tasks):
        seed = task_id + trial*N_tasks
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        x_train, y_train = generate1D(task, sigma=sigma, random=True, N=N_train)
        x_test, y_test   = generate1D(task, sigma=0., random=False, N=N_test)
        X_test[task_id, trial] = x_test.reshape(-1)
        Y_test[task_id, trial] = y_test
        X_train[task_id, trial] = x_train.reshape(-1)
        Y_train[task_id, trial] = y_train
        x_train_torch = torch.from_numpy(x_train.astype(np.float32)).clone()
        y_train_torch = torch.from_numpy(y_train.astype(np.float32)).clone()
        x_test_torch = torch.from_numpy(x_test.astype(np.float32)).clone()
        y_test_torch = torch.from_numpy(y_test.astype(np.float32)).clone()

        for method_id, method in enumerate(methods):
            match method:
                case 'xgboost':             
                    model = XGBRegressor(monotone_constraints=(1,), n_estimators=ls)
                    model.fit(x_train, y_train)
                    y_pred_train = model.predict(x_train)
                    y_pred_test = model.predict(x_test)
                case 'xgboost_val':             
                    x_train_small, x_val, y_train_small, y_val = train_test_split(x_train, y_train, test_size=.25, random_state=42)
                    model = XGBRegressor(monotone_constraints=(1,), n_estimators=ls, early_stopping_rounds=(ls // 10), verbosity=0)
                    model.fit(x_train_small, y_train_small, eval_set=[(x_train_small, y_train_small), (x_val, y_val)], verbose=0)
                    y_pred_train = model.predict(x_train)
                    y_pred_test = model.predict(x_test)
                case 'iso':
                    model = IsotonicRegression(y_min=0, y_max=1., out_of_bounds='clip')
                    model.fit(x_train, y_train)
                    y_pred_train = model.predict(x_train)
                    y_pred_test = model.predict(x_test)
                case 'lattice':
                    model = HLattice(1,torch.tensor([ls], dtype=torch.long), [0])
                    if(trial+task_id==0):
                        no_params[method_id] = total_params(model)
                        print(method, total_params(model), "parameters")
                    fit_torch(model, x_train_torch, y_train_torch.reshape(-1,1))
                    y_pred_train = model(x_train_torch).detach().numpy()
                    y_pred_test = model(x_test_torch).detach().numpy()
                case 'monotonic':
                    model = MonotonicNN(1, K, K)
                    if(trial+task_id==0):
                        no_params[method_id] = total_params(model)
                        print(method, total_params(model), "parameters")
                    fit_torch(model, x_train_torch, y_train_torch)
                    y_pred_train = model(x_train_torch).detach().numpy()
                    y_pred_test = model(x_test_torch).detach().numpy()            
                case 'smooth':
                    model = SmoothMonotonicNN(1, K, K, beta=-1.)
                    if(trial+task_id==0):
                        no_params[method_id] = total_params(model)
                        print(method, total_params(model), "parameters")
                    fit_torch(model, x_train_torch, y_train_torch)
                    y_pred_train = model(x_train_torch).detach().numpy()
                    y_pred_test = model(x_test_torch).detach().numpy()
                case 'smooth_sq':
                    model = SmoothMonotonicNN(1, K, K, beta=-1., transform='sq')
                    if(trial+task_id==0):
                        no_params[method_id] = total_params(model)
                        print(method, total_params(model), "parameters")
                    fit_torch(model, x_train_torch, y_train_torch)
                    y_pred_train = model(x_train_torch).detach().numpy()
                    y_pred_test = model(x_test_torch).detach().numpy()
                case 'xgboost_small':             
                    model = XGBRegressor(monotone_constraints=(1,), n_estimators=ls_small)
                    model.fit(x_train, y_train)
                    y_pred_train = model.predict(x_train)
                    y_pred_test = model.predict(x_test)
                case 'lip_small':
                    dim = 1
                    model = torch.nn.Sequential(
                        direct_norm(torch.nn.Linear(dim, width_small), kind="one-inf"),
                        GroupSort(width_small//2),
                        direct_norm(torch.nn.Linear(width_small, width_small), kind="inf"),
                        GroupSort(width_small//2),
                        direct_norm(torch.nn.Linear(width_small, 1), kind="inf"),
                    )
                    model = SigmaNet(model, sigma=1, monotone_constraints=(1,))
                    if(trial+task_id==0):
                        no_params[method_id] = total_params(model)
                        print(method, total_params(model), "parameters")
                    fit_torch(model, x_train_torch, y_train_torch.reshape(-1,1))
                    y_pred_train = model(x_train_torch).detach().numpy()
                    y_pred_test = model(x_test_torch).detach().numpy()
                case 'lip':
                    dim = 1
                    model = torch.nn.Sequential(
                        direct_norm(torch.nn.Linear(dim, width), kind="one-inf"),
                        GroupSort(width//2),
                        direct_norm(torch.nn.Linear(width, width), kind="inf"),
                        GroupSort(width//2),
                        direct_norm(torch.nn.Linear(width, 1), kind="inf"),
                    )
                    model = SigmaNet(model, sigma=1, monotone_constraints=(1,))
                    if(trial+task_id==0):
                        no_params[method_id] = total_params(model)
                        print(method, total_params(model), "parameters")
                    fit_torch(model, x_train_torch, y_train_torch.reshape(-1,1))
                    y_pred_train = model(x_train_torch).detach().numpy()
                    y_pred_test = model(x_test_torch).detach().numpy()

            MSE_train[task_id, method_id, trial] = mse(y_train, y_pred_train)
            MSE_test[task_id, method_id, trial] = mse(y_test, y_pred_test)
            MSE_clip[task_id, method_id, trial] = mse(y_test, np.clip(y_pred_test, 0., 1.))
            R2_train[task_id, method_id, trial] = r2(y_train, y_pred_train)
            R2_test[task_id, method_id, trial] = r2(y_test, y_pred_test)
            O_test[task_id, method_id, trial] = y_pred_test.reshape(-1)

fn = prefix + "univariate.npz"
np.savez(fn, MSE_train=MSE_train, MSE_test=MSE_test, MSE_clip=MSE_clip, R2_train=R2_train, R2_test=R2_test, 
     X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, O_test=O_test, no_params=no_params)


