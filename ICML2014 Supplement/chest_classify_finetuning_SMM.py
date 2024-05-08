import os
import torch
from tqdm import tqdm
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from monotonenorm import GroupSort, SigmaNet, direct_norm
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from chest_config import basepath


from MonotonicNN import SmoothMonotonicNN, SmoothMonotonicNNAlt, MonotonicNN, SMM_MLP
from MonotonicNNPaperUtils import Progress, MLP, total_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

XIMG = torch.load(os.path.join(basepath, "XIMG.pt"))
XTAB = torch.load(os.path.join(basepath, "XTAB.pt"))
Y = torch.tensor(torch.load(os.path.join(basepath, "Y.pt"))).float()


class ResNet18Mono(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        resnet = resnet18(pretrained=True).requires_grad_(True)
        monotone_constraints = [1, 1, 0, 0] + [0] * resnet.fc.in_features
        resnet.fc = torch.nn.Identity()
        self.resnet = resnet
        
        increasing = []
        for i, b in enumerate(monotone_constraints):
            if(b):
                increasing.append(i)

        dim = len(monotone_constraints)
        K = 6
        transform = "exp"
        mlp_neurons = 64
        mask = np.array(monotone_constraints)
        smm = SMM_MLP(dim, increasing, mlp_neurons, K, transform, last_linear=True)
        
        self.monotonic = smm.to(device)

        self.monotonic.load_state_dict(state_dict)

    def forward(self, ximg, xtab):
        ximg = self.resnet(ximg)
        x = torch.hstack([xtab, ximg])
        return self.monotonic(x)


accs = []
for i in range(3):
    torch.manual_seed(i)
    XIMG_train, XIMG_test, XTAB_train, XTAB_test, y_train, y_test = train_test_split(
        XIMG, XTAB, Y, test_size=0.2, random_state=i
    )

    XIMG_train = XIMG_train.float().to(device)
    XIMG_test = XIMG_test.float().to(device)
    XTAB_train = XTAB_train.float().to(device)
    XTAB_test = XTAB_test.float().to(device)
    y_train = y_train.float().unsqueeze(1).to(device)
    y_test = y_test.float().unsqueeze(1).to(device)
    train_loader = DataLoader(
        list(zip(XIMG_train, XTAB_train, y_train)), batch_size=2 ** 10, shuffle=True
    )

    state_dict = torch.load(f"models/chest_classify_nn_{i}.pt")

    model = ResNet18Mono(state_dict=state_dict).to(device)
    print(f"params: {sum(p.numel() for p in model.monotonic.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    EPOCHS = 20
    bar = tqdm(range(EPOCHS))
    acc = 0
    for i in bar:
        for ximg_, xtab_, y_ in train_loader:
            optimizer.zero_grad()
            y_pred = model(ximg_, xtab_)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = model(XIMG_test, XTAB_test)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, y_test)
            preds = torch.sigmoid(preds).cpu().numpy()
            acci = 0
            for cut in np.linspace(0, 1, 50):
                acci = max(acci, accuracy_score(y_test.cpu().numpy(), preds > cut))
            acc = max(acc, acci)
            bar.set_description(f"test loss: {loss:.5f}, current acc: {acci:.5f}, best acc: {acc:.5f}")
    accs.append(acc)

print(f"mean accuracy: {np.mean(accs):.5f}, std accuracy: {np.std(accs):.5f}")
