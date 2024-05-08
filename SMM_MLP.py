import torch
import torch.nn as nn
import numpy as np
        
"""
Smooth Monotonic Neural Network combined with simple MLP

An example of how a Smooth Monotonic Neural Network can be combined with a simple MLP with a single non-linear layer

Args:
    dim (int): number of inputs
    increasing: (list): list of integers indicating the variables with monotonicity constraint
    num_neuron (int): number of neurons in auxiliary MLP
    K (int): number of groups, it is assumed that H_k=K
    transform (string): type of transformation for ensuring positivity ('exp', 'abs', 'explin', 'sqr')
    last_linear (bool): indicating whether a sigmoid should be applied to the output
"""
class SMM_MLP(nn.Module):
    def __init__(self, dim, increasing, num_neuron, K=6, transform="exp", last_linear=False):
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
        self.smm = SmoothMonotonicNN(dim, K, K, mask=mask, beta=-1., transform=transform)
        
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
