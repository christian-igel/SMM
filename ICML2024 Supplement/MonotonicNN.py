import torch
import torch.nn as nn
import numpy as np


"""
Smooth Monotonic Neural Network.

The implementation is not optimized for speed.

Args:
    n (int): number of inputs
    K (int): number of groups
    h_K (int): number of neurons per group
    b_z (float): sdv. for Gaussian init., interval width for uniform init.
    b_t (float): sdv. for Gaussian init., interval width for uniform init. of bias
    beta (float): scaling parameter for LSE
    transform (string): type of transformation for ensuring positivity ('exp', 'abs', 'explin', 'sqr')
"""
class SmoothMonotonicNN(nn.Module):
    def __init__(self, n, K, h_K, b_z = 1., b_t = 1., beta=-1., transform='sqr'):
        super(SmoothMonotonicNN, self).__init__()
        self.K = K
        self.beta_init = beta
        self.b_z = b_z
        self.b_t = b_t
        self.beta = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.z = nn.ParameterList([nn.Parameter(torch.ones(h_K, n), requires_grad=True) for i in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.ones(h_K), requires_grad=True) for i in range(K)])
        self.softmax = nn.Softmax(dim=1)
        self.transform = transform
        self.reset_parameters()
        
    def check_grad(self):
        r = 0
        for i in range(self.K):
            r += torch.sum((self.z[i].grad == 0).int()).item()
            r += torch.sum((self.t[i].grad == 0).int()).item()
        return r
        
    def check_grad_neuron(self):
        zero = 0
        for i in range(self.K):
            biases_zero = (self.t[i].grad == 0).int()
            weights_zero = (self.z[i].grad == 0).int()
            all_zero = torch.prod(weights_zero, dim=1) * biases_zero
            zero += torch.sum( all_zero )
        return zero

    
    def reset_parameters(self):
        for i in range(self.K):
            torch.nn.init.trunc_normal_(self.z[i], std=self.b_z)
            torch.nn.init.trunc_normal_(self.t[i], std=self.b_t)
        torch.nn.init.constant_(self.beta, self.beta_init)
    
    def soft_max(self, a, beta):
        return torch.logsumexp(beta * a, dim=1)/beta
    
    def soft_min(self, a, beta):
        return -torch.logsumexp(-beta * a, dim=1)/beta
    
    def forward(self, x):
        for i in range(self.K):  # loop over groups
            # hidden layer
            if(self.transform == 'exp'):
                w = torch.exp(self.z[i])  # positive weights
            elif(self.transform == 'abs'):
                w = torch.abs(self.z[i])  # positive weights
            elif(self.transform == 'explin'):
                w = torch.where(self.z[i] > 1., self.z[i], torch.exp(self.z[i]-1.))  # positive weights
            else:
                w = self.z[i] * self.z[i]
            a = torch.matmul(x, w.t()) + self.t[i]
            g = self.soft_max(a, beta=torch.exp(self.beta))
            
            # output layer
            g = torch.unsqueeze(g, dim=1)
            if i==0:
                y = g
            else:
                y = torch.cat([y, g], dim=1)
        y = self.soft_min(y, beta=torch.exp(self.beta))
        return y
        
"""
Smooth Monotonic Neural Network.

This alternative implementation supports a mask indicating the variables with monotonicity constraint.
The implementation is not optimized for speed.

Args:
    n (int): number of inputs
    K (int): number of groups
    h_K (int): number of neurons per group
    mask: (np.array): Boolean mask indicating the variables with monotonicity constraint
    b_z (float): sdv. for Gaussian init., interval width for uniform init.
    b_t (float): sdv. for Gaussian init., interval width for uniform init. of bias
    beta (float): scaling parameter for LSE
    transform (string): type of transformation for ensuring positivity ('exp', 'abs', 'explin', 'sqr')
"""
class SmoothMonotonicNNAlt(nn.Module):
    def __init__(self, n, K, h_K, mask, b_z = 1., b_t = 1., beta=-1., transform="sqr"):
        super(SmoothMonotonicNNAlt, self).__init__()
        self.K = K
        self.beta_init = beta
        self.b_z = b_z
        self.b_t = b_t
        self.beta = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.z = nn.ParameterList([nn.Parameter(torch.ones(h_K, n), requires_grad=True) for i in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.ones(h_K), requires_grad=True) for i in range(K)])
        self.softmax = nn.Softmax(dim=1)
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
    
    def soft_max(self, a, beta):
        return torch.logsumexp(beta * a, dim=1)/beta
    
    def soft_min(self, a, beta):
        return -torch.logsumexp(-beta * a, dim=1)/beta
    
    def forward(self, x):
        for i in range(self.K):  # loop over groups
            # hidden layer
            if(self.transform == 'exp'):
                w = torch.exp(self.z[i])  # positive weights
            elif(self.transform == 'abs'):
                w = torch.abs(self.z[i])  # positive weights
            elif(self.transform == 'explin'):
                w = torch.where(self.z[i] > 1., self.z[i], torch.exp(self.z[i]-1.))  # positive weights
            else:
                w = self.z[i] * self.z[i]
            w = self.mask * w + self.mask_inv * self.z[i]  # restore non-constrained
            a = torch.matmul(x, w.t()) + self.t[i]
            g = self.soft_max(a, beta=torch.exp(self.beta))
            
            # output layer
            g = torch.unsqueeze(g, dim=1)
            if i==0:
                y = g
            else:
                y = torch.cat([y, g], dim=1)
        y = self.soft_min(y, beta=torch.exp(self.beta))
        return y



"""
Monotonic Neural Network.

Implementation folows:
Joseph Sill. Monotonic Networks. Advances in Neural Information Processing Systems 10, 1997.

Args:
    n (int): number of inputs
    K (int): number of groups
    h_K (int): number of neurons per group
    b_z (float): sdv. for Gaussian init., interval width for uniform init.
    b_t (float): sdv. for Gaussian init., interval width for uniform init. of bias
"""
class MonotonicNN(nn.Module):
    def __init__(self, n, K, h_K, b_z = 1., b_t = 1.):
        super(MonotonicNN, self).__init__()
        self.K = K
        self.h_K = h_K
        self.b_z = b_z
        self.b_t = b_t
        self.beta = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.z = nn.ParameterList([nn.Parameter(torch.ones(h_K, n), requires_grad=True) for i in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.ones(h_K), requires_grad=True) for i in range(K)])
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()
    
    def reset_parameters(self):
        for i in range(self.K):
            torch.nn.init.trunc_normal_(self.z[i], std=self.b_z)
            torch.nn.init.trunc_normal_(self.t[i], std=self.b_t)

    def forward(self, x):
        for i in range(self.K):  # loop over groups
            # hidden layer
            w = torch.exp(self.z[i])  # positive weights
            g = torch.matmul(x, w.t()) + self.t[i]
            g = torch.max(g, axis=1)
            # output layer
            if i==0:
                y = g.values
            else:
                y = torch.minimum(y, g.values)
        return y


"""
Monotonic Neural Network (alternative version).

Implementation folows:
Joseph Sill. Monotonic Networks. Advances in Neural Information Processing Systems 10, 1997.

This alternative implementation keeps track of alive/dead neurons.

Args:
    n (int): number of inputs
    K (int): number of groups
    h_K (int): number of neurons per group
    b_z (float): sdv. for Gaussian init., interval width for uniform init.
    b_t (float): sdv. for Gaussian init., interval width for uniform init. of bias
"""
class MonotonicNNAlt(nn.Module):
    def __init__(self, n, K, h_K, b_z = 1., b_t = 1.):
        super(MonotonicNNAlt, self).__init__()
        self.K = K
        self.h_K = h_K
        self.b_z = b_z
        self.b_t = b_t
        self.A = torch.zeros(K, h_K)
        self.beta = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.z = nn.ParameterList([nn.Parameter(torch.ones(h_K, n), requires_grad=True) for i in range(K)])
        self.t = nn.ParameterList([nn.Parameter(torch.ones(h_K), requires_grad=True) for i in range(K)])
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()
    
    def reset_parameters(self):
        for i in range(self.K):
            torch.nn.init.trunc_normal_(self.z[i], std=self.b_z)
            torch.nn.init.trunc_normal_(self.t[i], std=self.b_t)

    def forward(self, x):
        max_indices = []
        N = x.shape[0]  # number of data points
        for i in range(self.K):  # loop over groups
            # hidden layer
            w = torch.exp(self.z[i])  # positive weights
            g = torch.matmul(x, w.t()) + self.t[i]
            g = torch.max(g, axis=1)
            max_indices.append(g.indices)
            # output layer
            g = torch.unsqueeze(g.values, dim=1)
            if i==0:
                y = g
            else:
                y = torch.cat([y, g], dim=1)
        y = torch.min(y, axis=1)
        min_indices = y.indices
        for n in range(N):
            self.A[min_indices[n], max_indices[min_indices[n]][n]]+=1
            
        return y.values
    
    def reset_active_max(self):
        self.A.zero_()
        
    def active_max(self):
        return torch.count_nonzero(self.A).item(), self.A.numpy()
        
        
        
"""
Smooth Monotonic Neural Network combined with standard MLP

Args:
    dim (int): number of inputs
    increasing: (list): list of integers indicating the variables with monotonicity constraint
    num_neuron (int): number of neurons in auxiliary MLP
    K (int): number of groups, it is assumed that H_k=K
    transform (string): type of transformation for ensuring positivity ('exp', 'abs', 'explin', 'sqr')
"""
class SMM_MLP(nn.Module):
    def __init__(self, dim, increasing, num_neuron, K=6, transform="sqr", last_linear=False):
        super().__init__()
        
        # Mc: project to constrained features
        # Mu: project to unconstrained features
        Mcnp = np.zeros((dim, len(increasing)))
        Munp = np.zeros((dim, dim-len(increasing)))
        c_counter = 0
        u_counter = 0
        for i in range(dim):
            if (i in increasing):
                Mcnp[i, c_counter]=1
                c_counter+=1
            else:
                Munp[i, u_counter]=1
                u_counter+=1
        
        self.Mc = torch.from_numpy(Mcnp.astype(np.float32))  # not used
        self.Mu = torch.from_numpy(Munp.astype(np.float32))

        mask = np.zeros(dim)
        np.put(mask, increasing, 1)
        self.smm = SmoothMonotonicNNAlt(dim, K, K, mask=mask, beta=-1., transform=transform)
        
        self.fc1 = nn.Linear(dim - len(increasing), num_neuron)
        self.fc2 = nn.Linear(num_neuron, 1)
        
        self.last_linear = last_linear

    def forward(self, x):
        ysmm = self.smm(x).reshape(-1,1)

        xu = torch.matmul(x, self.Mu)
        xu = self.fc1(xu)
        xu = nn.functional.sigmoid(xu)
        yu = self.fc2(xu)

        if self.last_linear:
            y = ysmm + yu
        else:
            y = nn.functional.sigmoid(ysmm + yu)
    
        return y

