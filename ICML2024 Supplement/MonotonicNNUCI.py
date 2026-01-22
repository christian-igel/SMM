# # Smooth Monotonic Networks: Experiments on UCI bechmark functions
# 
# See Section 4.3 in the manuscript.
# 
# ## General definitions
# 
# Among others, we compare against XGBoost, which can be installed via `pip install xgboost`, and 
# the Hierarchical Lattice Layer, which can be installed via 
# `pip install pmlayer`.


import numpy as np
import pandas as pd

import random

import torch 
import torch.nn as nn

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from scipy.stats import wilcoxon

import matplotlib.pyplot as plt
from tqdm.notebook import tnrange

from xgboost import XGBRegressor

from pmlayer.torch.layers import HLattice

from MonotonicNN import SmoothMonotonicNN, SmoothMonotonicNNAlt, MonotonicNN, SMM_MLP
from MonotonicNNPaperUtils import Progress, MLP, total_params, fit_torch, fit_torch_val

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from monotonenorm import GroupSort, direct_norm, SigmaNet


n_folds = 5  # number of splits in cross-validation
n_trees = 100  # number of trees for XGBoost
K = 6  # number of SMM groups and neurons per group
k = 3  # lattice points per dimension with constraint
transform = "exp"  # way positivity is ensured in SMM
mlp_neurons = 64  # size of auxiliary networks for HLattice and SMM_MLP
maxiter = 100000  # upper bound on epochs, never reached for this value

prefix = "./"


# ## Cross-validation
# The function doing the cross-validation, relying on global variables for all hyperparameters.

def do_the_cv():
    seed = 0
    clip = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    no_methods = 6
    kf = KFold(n_splits=n_folds)
    mse_train=np.zeros((no_methods, n_folds))
    mse_test=np.zeros((no_methods, n_folds))
    no_params=np.zeros(no_methods)
    
    for split, (train, test) in enumerate(kf.split(x)):
        x_train, x_val, y_train, y_val = train_test_split(x[train], y[train], test_size=0.25)
        x_test = x[test]
        y_test = y[test]
        
        
        m = 0
        print("split:", split)
        x_train_torch = torch.from_numpy(x_train.astype(np.float32)).clone()
        x_test_torch = torch.from_numpy(x_test.astype(np.float32)).clone()
        x_val_torch = torch.from_numpy(x_val.astype(np.float32)).clone()
        y_train_torch = torch.from_numpy(y_train.astype(np.float32)).clone()
        y_test_torch = torch.from_numpy(y_test.astype(np.float32)).clone()
        y_val_torch = torch.from_numpy(y_val.astype(np.float32)).clone()
        
        print("SMM-MLP: ", end='')
        model = SMM_MLP(dim, increasing, mlp_neurons, K, transform)
        model_best = SMM_MLP(dim, increasing, mlp_neurons, K, transform)
        fit_torch_val(model, model_best, 
                      x_train_torch, y_train_torch, 
                      x_val_torch, y_val_torch, 
                      max_iterations=maxiter)
        y_smm_train = model_best(x_train_torch).detach().numpy().reshape(-1,1)
        y_smm_test = model_best(x_test_torch).detach().numpy().reshape(-1,1)
        mse_train[m, split] = mse(y_train, y_smm_train)
        if not clip:
            mse_test[m, split] = mse(y_test, y_smm_test)
        else:
            mse_test[m, split] = mse(y_test, np.clip(y_smm_test, 0., 1.))
        print(mse_test[m, split])
        if(split == 0):
            no_params[m] = total_params(model)
        m=m+1
        
        print("SMM: ", end='')
        model = SmoothMonotonicNNAlt(dim, K, K, beta=-1., mask=mask, transform=transform)
        model_best = SmoothMonotonicNNAlt(dim, K, K, beta=-1., mask=mask, transform=transform)
        fit_torch_val(model, model_best, 
                      x_train_torch, y_train_torch, 
                      x_val_torch, y_val_torch, 
                      max_iterations=maxiter)
        y_smm_train = model_best(x_train_torch).detach().numpy().reshape(-1,1)
        y_smm_test = model_best(x_test_torch).detach().numpy().reshape(-1,1)
        mse_train[m, split] = mse(y_train, y_smm_train)
        if not clip:
            mse_test[m, split] = mse(y_test, y_smm_test)
        else:
            mse_test[m, split] = mse(y_test, np.clip(y_smm_test, 0., 1.))
        print(mse_test[m, split])
        if(split == 0):
            no_params[m] = total_params(model)
        m=m+1  

        print("XG: ", end='')
        model = XGBRegressor(monotone_constraints=xg_constraints, n_estimators=n_trees, 
                             early_stopping_rounds=(n_trees // 10), verbosity=0)
        model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], verbose=0)           
        y_xg_train = model.predict(x_train)
        y_xg_test = model.predict(x_test)
        mse_train[m, split] = mse(y_train, y_xg_train)
        if not clip:
            mse_test[m, split] = mse(y_test, y_xg_test)
        else:
            mse_test[m, split] = mse(y_test, np.clip(y_xg_test, 0., 1.))
        print(mse_test[m, split])
        if(split == 0):
            no_params[m] = n_trees
        m=m+1

        print("HLL: ", end='')
        input_len = dim - len(increasing)
        output_len = torch.prod(lattice_sizes_tensor).item()
        ann = MLP(input_len, output_len, mlp_neurons)
        model = HLattice(dim, lattice_sizes_tensor, increasing, ann)
        ann_best = MLP(input_len, output_len, mlp_neurons)
        model_best = HLattice(dim, lattice_sizes_tensor, increasing, ann_best)
        fit_torch_val(model, model_best, 
                      x_train_torch, y_train_torch, 
                      x_val_torch, y_val_torch, 
                      max_iterations=maxiter)
        y_hll_train = model_best(x_train_torch).detach().numpy().reshape(-1,1)
        y_hll_test = model_best(x_test_torch).detach().numpy().reshape(-1,1)
        mse_train[m, split] = mse(y_train, y_hll_train)
        if not clip:
            mse_test[m, split] = mse(y_test, y_hll_test)
        else:
            mse_test[m, split] = mse(y_test, np.clip(y_hll_test, 0., 1.))                           
        print(mse_test[m, split])
        if(split == 0):
            no_params[m] = total_params(model)
        m=m+1
        
        print("Lip_small: ", end='')
        model = torch.nn.Sequential(
            direct_norm(torch.nn.Linear(dim, width_small), kind="one-inf"),
            GroupSort(width_small//2),
            direct_norm(torch.nn.Linear(width_small, width_small), kind="inf"),
            GroupSort(width_small//2),
            direct_norm(torch.nn.Linear(width_small, 1), kind="inf"),
        )
        model = SigmaNet(model, sigma=1, monotone_constraints=xg_constraints)
        model_best = torch.nn.Sequential(
            direct_norm(torch.nn.Linear(dim, width_small), kind="one-inf"),
            GroupSort(width_small//2),
            direct_norm(torch.nn.Linear(width_small, width_small), kind="inf"),
            GroupSort(width_small//2),
            direct_norm(torch.nn.Linear(width_small, 1), kind="inf"),
        )
        model_best = SigmaNet(model_best, sigma=1, monotone_constraints=xg_constraints)
        
        fit_torch_val(model, model_best, 
                      x_train_torch, y_train_torch, 
                      x_val_torch, y_val_torch, 
                      max_iterations=maxiter)
        y_lip_train = model_best(x_train_torch).detach().numpy().reshape(-1,1)
        y_lip_test = model_best(x_test_torch).detach().numpy().reshape(-1,1)
        mse_train[m, split] = mse(y_train, y_lip_train)
        if not clip:
            mse_test[m, split] = mse(y_test, y_lip_test)
        else:
            mse_test[m, split] = mse(y_test, np.clip(y_lip_test, 0., 1.))
        print(mse_test[m, split])
        if(split == 0):
            no_params[m] = total_params(model)
        m=m+1
        
        print("Lip: ", end='')
        model = torch.nn.Sequential(
            direct_norm(torch.nn.Linear(dim, width), kind="one-inf"),
            GroupSort(width//2),
            direct_norm(torch.nn.Linear(width, width), kind="inf"),
            GroupSort(width//2),
            direct_norm(torch.nn.Linear(width, 1), kind="inf"),
        )
        model = SigmaNet(model, sigma=1, monotone_constraints=xg_constraints)
        model_best = torch.nn.Sequential(
            direct_norm(torch.nn.Linear(dim, width), kind="one-inf"),
            GroupSort(width//2),
            direct_norm(torch.nn.Linear(width, width), kind="inf"),
            GroupSort(width//2),
            direct_norm(torch.nn.Linear(width, 1), kind="inf"),
        )
        model_best = SigmaNet(model_best, sigma=1, monotone_constraints=xg_constraints)
        
        fit_torch_val(model, model_best, 
                      x_train_torch, y_train_torch, 
                      x_val_torch, y_val_torch, 
                      max_iterations=maxiter)
        y_lip_train = model_best(x_train_torch).detach().numpy().reshape(-1,1)
        y_lip_test = model_best(x_test_torch).detach().numpy().reshape(-1,1)
        mse_train[m, split] = mse(y_train, y_lip_train)
        if not clip:
            mse_test[m, split] = mse(y_test, y_lip_test)
        else:
            mse_test[m, split] = mse(y_test, np.clip(y_lip_test, 0., 1.))
        print(mse_test[m, split])
        if(split == 0):
            no_params[m] = total_params(model)
        m=m+1
        
    return mse_train, mse_test, no_params
        


# ## Energy

df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx')

# Some of the methods - not the SMM - require sependent and indepedent variables to be in [0,1]. To ensure that all test data points are in that interval, we normalize the data before we split into training and test sets. In a real-world setting, one of course does not have access to the test data for computing the normalization.


dim = 8  # number of input variables
x = df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values
y = df[["Y1"]]
scaler = preprocessing.MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y)

increasing = (2,4,6)
mask = np.array([0,0,1,0,1,0,1,0])
xg_constraints=(0,0,1,0,1,0,1,0)
lattice_sizes = list(np.ones(len(increasing))*k)
lattice_sizes_tensor = torch.tensor(lattice_sizes, dtype=torch.long)

width_small = 22 #dim + K
width = 24 #dim + K + 2

MSE_train, MSE_test, no_params = do_the_cv()

fn = prefix + "energy-y1-results-val.npz"
np.savez(fn, MSE_train=MSE_train, MSE_test=MSE_test, no_params= no_params)


y = df[["Y2"]]
y = scaler.fit_transform(y)

width_small = dim + K
width = dim + K + 2
width_small = 22 #dim + K
width = 24 #dim + K + 2

MSE_train, MSE_test, no_params = do_the_cv()

fn = prefix + "energy-y2-results-val.npz"
np.savez(fn, MSE_train=MSE_train, MSE_test=MSE_test, no_params=no_params)


# ## Qsar

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv', sep=';', header=None)
x = df[df.columns[0:6]].values
y = df[df.columns[6]].values
x = scaler.fit_transform(x)
y = scaler.fit_transform(y.reshape(-1, 1))

dim = 6
increasing = (1,5)
mask = np.array([0,1,0,0,0,1])
xg_constraints = (0,1,0,0,0,1)
lattice_sizes = list(np.ones(len(increasing))*k)
lattice_sizes_tensor = torch.tensor(lattice_sizes, dtype=torch.long)

width_small = dim + K + 8
width = dim + K + 10

MSE_train, MSE_test, no_params = do_the_cv()

fn = prefix + "qsar-results-val.npz"
np.savez(fn, MSE_train=MSE_train, MSE_test=MSE_test, no_params=no_params)


# ## Concrete

df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls")
print(df.columns[0:8])
print(df.columns[8])

k = 3
dim = 8
x = df[df.columns[0:8]].values
y = df[df.columns[8]].values
scaler = preprocessing.MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y.reshape(-1, 1))

increasing = (3,)
mask = np.array([0,0,0,1,0,0,0,0])
xg_constraints=(0,0,0,1,0,0,0,0)
lattice_sizes = list(np.ones(len(increasing))*k)
lattice_sizes_tensor = torch.tensor(lattice_sizes, dtype=torch.long)

width_small = dim + K + 10
width = dim + K + 12

MSE_train, MSE_test, no_params = do_the_cv()

fn = prefix + "concrete-results-val.npz"
np.savez(fn, MSE_train=MSE_train, MSE_test=MSE_test, no_params=no_params)
