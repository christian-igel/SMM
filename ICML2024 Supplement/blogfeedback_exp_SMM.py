import torch
from monotonenorm import SigmaNet, GroupSort, direct_norm
from tqdm import tqdm

from MonotonicNN import SmoothMonotonicNN, SmoothMonotonicNNAlt, MonotonicNN, SMM_MLP
from MonotonicNNPaperUtils import Progress, MLP, total_params, fit_torch, fit_torch_val

import numpy as np

def run_exp(
    Xtr, Ytr, Xts, Yts, monotone_constraints,
    max_lr, expwidth, depth, Lip, batchsize, seed
):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xtrt = torch.tensor(Xtr, dtype=torch.float32).to(device)
    Ytrt = torch.tensor(Ytr, dtype=torch.float32).view(-1, 1).to(device)
    Xtst = torch.tensor(Xts, dtype=torch.float32).to(device)
    Ytst = torch.tensor(Yts, dtype=torch.float32).view(-1, 1).to(device)

    # normalize training data
    mean = Xtrt.mean(0)
    std = Xtrt.std(0)
    Xtrt = (Xtrt - mean) / std
    Xtst = (Xtst - mean) / std
    
    print(Yts.min(), Yts.max(), Yts.mean())

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtrt, Ytrt), batch_size=batchsize, shuffle=True
    )

    per_layer_lip = Lip ** (1 / depth)
    width = 2 ** expwidth

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
    #max_lr=2e-5
    #batchsize=2 ** 8
    mask = np.array(monotone_constraints)
    #smm = SmoothMonotonicNNAlt(dim, K, K, beta=-1., mask=mask, transform=transform)
    smm = SMM_MLP(dim, increasing, mlp_neurons, K, transform, last_linear=False)

    # number of elements
    print(sum(p.numel() for p in smm.parameters()))
    model = smm

    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    EPOCHS = 1000

    print("params:", sum(p.numel() for p in model.parameters()))
    bar = tqdm(range(EPOCHS))
    best_rmse = 1
    for _ in bar:
        model.train()
        for Xi, yi in dataloader:
            y_pred = model(Xi)
            #y_pred = torch.nn.functional.sigmoid(model(Xi))
            #y_pred = torch.clamp(y_pred, 0., 1.)
            losstr = torch.nn.functional.mse_loss(y_pred, yi)
            optimizer.zero_grad()
            losstr.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_predts = model(Xtst)
            #y_predts = torch.nn.functional.sigmoid(model(Xtst))
            #y_predts = torch.clamp(y_predts, 0., 1.)
            lossts = torch.nn.functional.mse_loss(y_predts, Ytst)
            tsrmse = lossts.item() ** 0.5
            trrmse = losstr.item() ** 0.5
            best_rmse = min(best_rmse, tsrmse)
            bar.set_description(
                f"train rmse: {trrmse:.5f} test rmse: {tsrmse:.5f}, best: {best_rmse:.5f}, lr: {optimizer.param_groups[0]['lr']:.5f}"
            )
    return best_rmse
