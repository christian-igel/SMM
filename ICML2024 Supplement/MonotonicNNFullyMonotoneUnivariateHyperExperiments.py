# # Smooth Monotonic Networks: Experiments on fully monotonic functions
# The results are stored in files, which are read by ``MonotonicNNPaperEvaluate.ipynb``.

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

class NewSmoothMonotonicNN(nn.Module):
    def __init__(self, n, K, h_K, mask=None, b_z = 1., b_t = 1., beta=-1., transform="exp", scale_beta=False):
        super(NewSmoothMonotonicNN, self).__init__()
        self.K = K
        self.beta_init = beta
        if(scale_beta):
            self.b_z = b_z * np.exp(self.beta_init)
            self.b_t = b_t * np.exp(self.beta_init)
        else:
            self.b_z = b_z
            self.b_t = b_t

        self.gamma = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.z = nn.ParameterList([nn.Parameter(torch.ones(h_K, n), requires_grad=True) for i in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.ones(h_K), requires_grad=True) for i in range(K)])
        self.softmax = nn.Softmax(dim=1)
        if mask is None:
            self.mask = None
        else:
            self.mask = torch.BoolTensor(mask)
            assert mask.shape == (n, )
            self.mask_inv = ~self.mask
            
        self.transform = transform
        self.reset_parameters()
    
    def reset_parameters(self):
        for i in range(self.K):
            torch.nn.init.trunc_normal_(self.z[i], std=self.b_z)
            torch.nn.init.trunc_normal_(self.t[i], std=self.b_t)
        torch.nn.init.constant_(self.beta, self.beta_init)
    
    def soft_max(self, a):
        return torch.logsumexp(a, dim=1)
    
    def soft_min(self, a):
        return -torch.logsumexp(-a, dim=1)
    
    def forward(self, x):
        for i in range(self.K):  # loop over groups
            # hidden layer
            if(self.transform == 'exp'):
                w = torch.exp(self.z[i])  # positive weights
            if(self.transform == 'abs'):
                w = torch.abs(self.z[i])  # positive weights
            if(self.transform == 'explin'):
                w = torch.where(self.z[i] > 1., self.z[i], torch.exp(self.z[i]-1.))  # positive weights
            else:
                w = self.z[i] * self.z[i]
            if self.mask is not None:
                w = self.mask * w + self.mask_inv * self.z[i]  # restore non-constrained
            a = torch.matmul(x, w.t()) + self.t[i]
            g = self.soft_max(a)
            
            # output layer
            g = torch.unsqueeze(g, dim=1)
            if i==0:
                y = g
            else:
                y = torch.cat([y, g], dim=1)
        y =  self.soft_min(y) / torch.exp(self.beta) + self.gamma
        return y
        
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


T = 11  # number of trials, odd number for having a "median trial"
tasks = ['sq', 'sqrt', 'sigmoid10']
K_values = (2, 4, 6, 8)
beta_values = (-3., -2., -1., 0., 1.)


N_tasks = len(tasks)
N_K = len(K_values)
N_beta = len(beta_values)


MSE_train = np.zeros((N_tasks, N_K, N_beta, T))
MSE_test = np.zeros((N_tasks, N_K, N_beta, T))
MSE_clip = np.zeros((N_tasks, N_K, N_beta, T))
no_params = np.zeros(N_K)

for K_id, K in enumerate(K_values):
    for beta_id, beta in enumerate(beta_values):
        #print("K:", K, "beta:", beta)
        for trial in range(T):
            for task_id, task in enumerate(tasks):
                print("K:", K, "beta:", beta, task, trial)
                seed = task_id + trial*N_tasks
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                x_train, y_train = generate1D(task, sigma=sigma, random=True, N=N_train)
                x_test, y_test   = generate1D(task, sigma=0., random=False, N=N_test)
                x_train_torch = torch.from_numpy(x_train.astype(np.float32)).clone()
                y_train_torch = torch.from_numpy(y_train.astype(np.float32)).clone()
                x_test_torch = torch.from_numpy(x_test.astype(np.float32)).clone()
                y_test_torch = torch.from_numpy(y_test.astype(np.float32)).clone()


                model = NewSmoothMonotonicNN(1, K, K, beta=beta)
                if(trial+task_id==0):
                    no_params[K_id] = total_params(model)
                fit_torch(model, x_train_torch, y_train_torch)
                y_pred_train = model(x_train_torch).detach().numpy()
                y_pred_test = model(x_test_torch).detach().numpy()

                MSE_train[task_id, K_id, beta_id, trial] = mse(y_train, y_pred_train)
                MSE_test[task_id, K_id, beta_id, trial] = mse(y_test, y_pred_test)
                MSE_clip[task_id, K_id, beta_id, trial] = mse(y_test, np.clip(y_pred_test, 0., 1.))


fn = prefix + "hyper.npz"
np.savez(fn, MSE_train=MSE_train, MSE_test=MSE_test, MSE_clip=MSE_clip, no_params=no_params)


