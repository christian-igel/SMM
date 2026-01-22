import numpy as np

from loan_exp_SMM import run_exp
from loaders.loan_loader import load_data, mono_list


Xtr, Ytr, Xts, Yts = load_data(get_categorical_info=False,constant=0.)
monotonic_constraints = np.array([int(i in mono_list) for i in range(Xtr.shape[1])])

accs = [
    run_exp(Xtr, Ytr, Xts, Yts, monotonic_constraints, 16, 3, i) for i in range(3)
]  # 3 seeds
# print mean and std of the 3 runs
print(f"mean: {np.mean(accs):.4f}, std: {np.std(accs):.4f}")
