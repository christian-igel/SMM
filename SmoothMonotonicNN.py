import torch
import torch.nn as nn
import numpy as np
        
"""
Smooth Monotonic Neural Network.

This implementation is not optimized for speed.

Args:
    n (int): number of inputs
    K (int): number of groups
    h_K (int): number of neurons per group
    mask: (np.array): Boolean mask indicating the variables with monotonicity constraint
    b_z (float): sdv. for Gaussian init., interval width for uniform init.
    b_t (float): sdv. for Gaussian init., interval width for uniform init. of bias
    beta (float): scaling parameter for LSE
    transform (string): type of transformation for ensuring positivity ('exp', 'abs', 'explin', 'sqr')
    scale_beta (bool): indicating whether initial weights should be scaled by beta
"""
class SmoothMonotonicNN(nn.Module):
    def __init__(self, n, K, h_K, mask=None, b_z = 1., b_t = 1., beta=-1., transform="exp", scale_beta=False):
        super(SmoothMonotonicNN, self).__init__()
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
