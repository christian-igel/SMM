import torch
from tqdm import tqdm
from monotonenorm import GroupSort, direct_norm, SigmaNet
import numpy as np
from sklearn.metrics import accuracy_score

from MonotonicNN import SmoothMonotonicNN, SmoothMonotonicNNAlt, MonotonicNN, SMM_MLP
from MonotonicNNPaperUtils import Progress, MLP, total_params, fit_torch, fit_torch_val


def run_exp(Xtr, Ytr, Xts, Yts, monotone_constraints, width, depth, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

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
    smm = SMM_MLP(dim, increasing, mlp_neurons, K, transform, last_linear=False)

    # number of elements
    print(sum(p.numel() for p in smm.parameters()))
    model = smm
    model = model.to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    print("params:", sum(p.numel() for p in model.parameters()))

    Xtrt = torch.tensor(Xtr, dtype=torch.float32).to(device)
    Ytrt = torch.tensor(Ytr, dtype=torch.float32).view(-1, 1).to(device)
    Xtst = torch.tensor(Xts, dtype=torch.float32).to(device)
    Ytst = torch.tensor(Yts, dtype=torch.float32).view(-1, 1).to(device)

    mean = Xtrt.mean(0)
    std = Xtrt.std(0)
    Xtrt = (Xtrt - mean) / std
    Xtst = (Xtst - mean) / std

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtrt, Ytrt), batch_size=int(2 ** 9), shuffle=True
    )
    max_acc = 0

    bar = tqdm(range(100))
    for i in bar:
        for Xi, yi in dataloader:
            y_pred = model(Xi)
            losstr = torch.nn.functional.binary_cross_entropy(y_pred, yi)
            optimizer.zero_grad()
            losstr.backward()
            optimizer.step()

        with torch.no_grad():
            y_predts = model(Xtst)
            lossts = torch.nn.functional.binary_cross_entropy(y_predts, Ytst)
            acc = 0
            for i in np.linspace(0, 1, 100):
                acc = max(
                    acc, accuracy_score(Ytst.cpu().numpy(), y_predts.cpu().numpy() > i),
                )

            max_acc = max(max_acc, acc)
            bar.set_description(
                f"Loss: {losstr.item():.4f} {lossts.item():.4f}, acc: {acc.item():.4f}, max_acc: {max_acc:.4f}"
            )
    return max_acc
