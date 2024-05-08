import pandas as pd
import torch
import tqdm
from monotonenorm import GroupSort, direct_norm, SigmaNet

import numpy as np
from MonotonicNN import SmoothMonotonicNN, SmoothMonotonicNNAlt, MonotonicNN, SMM_MLP
from MonotonicNNPaperUtils import Progress, MLP, total_params, fit_torch, fit_torch_val


df = pd.read_csv('data/auto-mpg.csv')
df = df[df.horsepower != "?"]

monotone_constraints=[0,-1,-1,-1,0,0,0]
flipper = torch.tensor(2. * np.array(monotone_constraints) + 1., dtype=torch.float32)

# mpg is regression target
# cylinders, displacement, horsepower, weight, acceleration, model year, origin are features
X = df.drop(columns=['mpg', 'car name']).values
Y = df['mpg'].values
#X = torch.tensor(X.astype(float)*flipper, dtype=torch.float32)
X = torch.tensor(X.astype(float), dtype=torch.float32)*flipper
Y = torch.tensor(Y.astype(float), dtype=torch.float32).view(-1, 1)
X = (X - X.mean(0)) / X.std(0)
Ymean = Y.mean(0)
Ystd = Y.std(0)
Y = (Y - Ymean) / Ystd

rmses = []
for seed in range(3):
  torch.manual_seed(seed)
  
  
    
  # split in train and test
  randperm = torch.randperm(X.shape[0])
  X = X[randperm]
  Y = Y[randperm]
  split = int(0.8 * X.shape[0])
  Xtr = X[:split]
  Ytr = Y[:split]
  Xts = X[split:]
  Yts = Y[split:]


  print(type(monotone_constraints), monotone_constraints, len(monotone_constraints))
  increasing = []
  for i, b in enumerate(monotone_constraints):
    if(b):
      increasing.append(i)
  print(increasing)

  dim = Xtr.shape[1]
  K = 6
  transform = "exp"
  mlp_neurons = 64
  mask = np.array(monotone_constraints)
  #smm = SmoothMonotonicNNAlt(dim, K, K, beta=-1., mask=mask, transform=transform)
  smm = SMM_MLP(dim, increasing, mlp_neurons, K, transform, last_linear=True)

  model = smm

  # number of elements
  print(sum(p.numel() for p in model.parameters()))


  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  epochs = 2000

  mse = float('inf')
  bar = tqdm.tqdm(range(epochs))
  for epoch in bar:
    batch_size = 64
    for i in range(0, Xtr.shape[0], batch_size):
      optimizer.zero_grad()
      yhat = model(Xtr[i:i+batch_size])
      loss = torch.nn.functional.mse_loss(yhat, Ytr[i:i+batch_size])
      loss.backward()
      optimizer.step()
    with torch.no_grad():
      yhat = model(Xts)
      # unscaled mse
      new_mse = torch.nn.functional.mse_loss(yhat * Ystd, Yts * Ystd)
      mse = min(mse, new_mse.item())
      bar.set_description(f"mse: {new_mse:.1f}, best: {mse:.1f}")
  rmses.append(mse)

print(f"mean: {torch.tensor(rmses).mean():.5f}, std: {torch.tensor(rmses).std():.5f}")
