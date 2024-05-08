import numpy as np
import torch
import torch.nn as nn

"""
Training Progress.

Implementation folows:
Prechelt, Lutz. Early Stopping—But When? Neural Networks: Tricks of the Trade (2012): 53-67.

Args:
    strip (int): length of training strip observed
    threshold (float): lower thresholf on progress

"""
class Progress():
    def __init__(self, strip=5, threshold=0.01):
        self.strip = strip
        self.E = np.ones(strip)
        self.t = 0
        self.valid = False
        self.threshold = threshold
        
    def progress(self):
        return 1000*((self.E.mean() / self.E.min()) - 1.)
    
    def stop(self):
        if self.valid==False:
            return False
        r = (self.progress() < self.threshold)
        return r

    def update(self, e):
        self.E[np.mod(self.t, self.strip)] = e
        self.t += 1
        if self.t>=self.strip:
            self.valid=True
        return self.stop()


"""
Compute number of parameters of a PyTorch model.

Args:
    model: the model

"""
def total_params(model):
    r = sum(
        param.numel() for param in model.parameters()
    )
    return r


"""
Gradent based fitting of PyTorch models using RProp.

Args:
    model: the model
    x: inputs
    y: labels
    threshold: threshold on learning progress
    max_iterations: maximum number of iterations

"""
def fit_torch(model, x, y, threshold=1e-3, max_iterations=100000):
    P = Progress(5, threshold=threshold)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    for epoch in range(max_iterations):
        pred_y = model(x)
        loss = loss_function(pred_y, y)
        stop = P.update(loss.item())
        if(stop):
            return
        loss.backward()
        optimizer.step()
        model.zero_grad()
        
"""
Gradent based fitting of PyTorch models using RProp using validation data.

Args:
    model: the model
    best_model: the model with minimum validation error
    x: inputs training data
    y: labels training data
    x: inputs validation data
    y: labels validation data
    threshold: threshold on learning progress
    max_iterations: maximum number of iterations
    verbose: if not 0, report if training was stopped due to lack of training progress
"""
def fit_torch_val(model, best_model, x, y, x_val, y_val, threshold=100, max_iterations=100000, verbose=1):
    best_val = 0
    best_epoch = 0
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    for epoch in range(max_iterations):
        pred_y = model(x).reshape(-1,1)
        loss = loss_function(pred_y, y)
        loss.backward()
        optimizer.step()
        model.zero_grad()
        pred_y = model(x_val).reshape(-1,1)
        loss_val = loss_function(pred_y, y_val)
        if(epoch == 0):
            best_val = loss_val
            best_model.load_state_dict(model.state_dict())
        elif (loss_val<best_val):
            best_val = loss_val
            best_model.load_state_dict(model.state_dict())
            best_epoch = epoch
        if(epoch-best_epoch > threshold):
            if(verbose):
                print("("+str(epoch)+") ", end='')
            break
            
            
"""
MLP for HLL.
"""
class MLP(nn.Module):
    def __init__(self, input_len, output_len, num_neuron):
        super().__init__()
        self.fc1 = nn.Linear(input_len, num_neuron)
        self.fc2 = nn.Linear(num_neuron, output_len)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
