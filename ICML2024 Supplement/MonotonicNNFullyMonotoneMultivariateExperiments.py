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

def generatePoly(dim=2, degree=2, sigma_train=0., sigma_test=0, N_train=100, N_test=100):
    x_train = np.random.rand(N_train, dim)
    x_test = np.random.rand(N_test, dim)
    poly = PolynomialFeatures(degree)  # includes bias
    x_poly_train = poly.fit_transform(x_train)
    x_poly_test = poly.fit_transform(x_test)
    w = np.random.rand(x_poly_train.shape[1])
    w_sum = w.sum()
    y_train = np.sum(x_poly_train * w, axis=1)/w_sum + sigma_train * np.random.normal(0, 1., N_train)
    y_test = np.sum(x_poly_test * w, axis=1)/w_sum + sigma_test * np.random.normal(0, 1., N_test)
    return x_train, y_train, x_test, y_test

N_train = 500
N_test = 1000
T = 21
sigma = 0.01  # noise level
trees = 100
trees2 = 200
methods = ['smooth', 'xgboost', 'xgboost_val', 'xgboost2', 'xgboost2_val','lattice', 'lattice_plus', 'lip_small', 'lip']
dims = [2, 4, 6]
degree = 2
ls = [10, 3, 2]
K = 6

N_methods = len(methods)
N_tasks = len(dims)
MSE_train = np.zeros((N_tasks, N_methods, T))
MSE_test = np.zeros((N_tasks, N_methods, T))
MSE_clip = np.zeros((N_tasks, N_methods, T))
R2_train = np.zeros((N_tasks, N_methods, T))
R2_test = np.zeros((N_tasks, N_methods, T))
O_test = np.zeros((N_tasks, N_methods, T, N_test))
no_params=np.zeros(N_methods)
for task_id, dim in enumerate(dims):
    for trial in tnrange(T):
        seed = task_id + trial*N_tasks
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        lattice_sizes = list(np.ones(dim)*ls[task_id])
        lattice_sizes_plus = list(np.ones(dim)*(ls[task_id] + 1))
        lattice_sizes_tensor = torch.tensor(lattice_sizes, dtype=torch.long)
        lattice_sizes_tensor_plus = torch.tensor(lattice_sizes_plus, dtype=torch.long)
        increasing_hll = list(range(dim))
        increasing = [1] * dim

        x_train, y_train, x_test, y_test = generatePoly(dim, degree=degree, sigma_train=sigma, sigma_test=0., N_train=N_train, N_test=N_test)
        x_train_torch = torch.from_numpy(x_train.astype(np.float32)).clone()
        y_train_torch = torch.from_numpy(y_train.astype(np.float32)).clone()
        x_test_torch = torch.from_numpy(x_test.astype(np.float32)).clone()
        y_test_torch = torch.from_numpy(y_test.astype(np.float32)).clone()

        for method_id, method in enumerate(methods):
            match method:
                case 'xgboost':             
                    model = XGBRegressor(monotone_constraints=tuple(increasing), n_estimators=trees)
                    model.fit(x_train, y_train)
                    y_pred_train = model.predict(x_train)
                    y_pred_test = model.predict(x_test)
                case 'xgboost_val':             
                    x_train_small, x_val, y_train_small, y_val = train_test_split(x_train, y_train, test_size=.25, random_state=42)
                    model = XGBRegressor(monotone_constraints=tuple(increasing), n_estimators=trees, 
                                         early_stopping_rounds=(trees // 10), verbosity=0)
                    model.fit(x_train_small, y_train_small, eval_set=[(x_train_small, y_train_small), (x_val, y_val)], verbose=0)
                    y_pred_train = model.predict(x_train)
                    y_pred_test = model.predict(x_test)
                case 'xgboost2':             
                    model = XGBRegressor(monotone_constraints=tuple(increasing), n_estimators=trees2)
                    model.fit(x_train, y_train)
                    y_pred_train = model.predict(x_train)
                    y_pred_test = model.predict(x_test)
                case 'xgboost2_val':             
                    x_train_small, x_val, y_train_small, y_val = train_test_split(x_train, y_train, test_size=.25, random_state=42)
                    model = XGBRegressor(monotone_constraints=tuple(increasing), n_estimators=trees2, 
                                         early_stopping_rounds=(trees2 // 10), verbosity=0)
                    model.fit(x_train_small, y_train_small, eval_set=[(x_train_small, y_train_small), (x_val, y_val)], verbose=0)
                    y_pred_train = model.predict(x_train)
                    y_pred_test = model.predict(x_test)
                case 'lattice':
                    model = HLattice(dim, lattice_sizes_tensor, increasing_hll)
                    if(trial==0):
                        print(method, total_params(model), "parameters")
                        no_params[method_id] = total_params(model)
                    fit_torch(model, x_train_torch, y_train_torch.reshape(-1,1))
                    y_pred_train = model(x_train_torch).detach().numpy()
                    y_pred_test = model(x_test_torch).detach().numpy()      
                case 'lattice_plus':
                    model = HLattice(dim, lattice_sizes_tensor_plus, increasing_hll)
                    if(trial==0):
                        no_params[method_id] = total_params(model)
                        print(method, total_params(model), "parameters")
                    fit_torch(model, x_train_torch, y_train_torch.reshape(-1,1))
                    y_pred_train = model(x_train_torch).detach().numpy()
                    y_pred_test = model(x_test_torch).detach().numpy()      
                case 'smooth':
                    model = SmoothMonotonicNN(dim, K, K, beta=-1.)
                    if(trial==0):
                        no_params[method_id] = total_params(model)
                        print(method, total_params(model), "parameters")
                    fit_torch(model, x_train_torch, y_train_torch)
                    y_pred_train = model(x_train_torch).detach().numpy()
                    y_pred_test = model(x_test_torch).detach().numpy()
                case 'lip_small':
                    width_small = int(dim + K)
                    model = torch.nn.Sequential(
                        direct_norm(torch.nn.Linear(dim, width_small), kind="one-inf"),
                        GroupSort(width_small//2),
                        direct_norm(torch.nn.Linear(width_small, width_small), kind="inf"),
                        GroupSort(width_small//2),
                        direct_norm(torch.nn.Linear(width_small, 1), kind="inf"),
                    )
                    model = SigmaNet(model, sigma=1, monotone_constraints=increasing)
                    if(trial==0):
                        no_params[method_id] = total_params(model)
                        print(method, total_params(model), "parameters")
                    fit_torch(model, x_train_torch, y_train_torch.reshape(-1,1))
                    y_pred_train = model(x_train_torch).detach().numpy()
                    y_pred_test = model(x_test_torch).detach().numpy()
                case 'lip':
                    width = int(dim + K) + 2
                    model = torch.nn.Sequential(
                        direct_norm(torch.nn.Linear(dim, width), kind="one-inf"),
                        GroupSort(width//2),
                        direct_norm(torch.nn.Linear(width, width), kind="inf"),
                        GroupSort(width//2),
                        direct_norm(torch.nn.Linear(width, 1), kind="inf"),
                    )
                    model = SigmaNet(model, sigma=1, monotone_constraints=increasing)
                    fit_torch(model, x_train_torch, y_train_torch.reshape(-1,1))
                    y_pred_train = model(x_train_torch).detach().numpy()
                    y_pred_test = model(x_test_torch).detach().numpy()
                    if(trial==0):
                        no_params[method_id] = total_params(model)
                        print(method, total_params(model), "parameters")

            MSE_train[task_id, method_id, trial] = mse(y_train, y_pred_train)
            MSE_test[task_id, method_id, trial] = mse(y_test, y_pred_test)
            MSE_clip[task_id, method_id, trial] = mse(y_test, np.clip(y_pred_test, 0., 1.))
            R2_train[task_id, method_id, trial] = r2(y_train, y_pred_train)
            R2_test[task_id, method_id, trial] = r2(y_test, y_pred_test)
            O_test[task_id, method_id, trial] = y_pred_test.reshape(-1)
fn = prefix + "multivariate.npz"
np.savez(fn, MSE_train=MSE_train, MSE_test=MSE_test, MSE_clip=MSE_clip, R2_train=R2_train, R2_test=R2_test, O_test=O_test, no_params=no_params)
