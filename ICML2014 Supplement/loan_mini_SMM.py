import numpy as np
from sklearn.linear_model import Ridge

from loan_exp_SMM import run_exp
from loaders.loan_loader import load_data, mono_list


Xtr, Ytr, Xts, Yts = load_data(get_categorical_info=False)
monotonic_constraints = np.array([int(i in mono_list) for i in range(Xtr.shape[1])])

# Ridge regression to find the best features
ridge = Ridge()
ridge.fit(Xtr, Ytr)
top_features = np.argsort(np.abs(ridge.coef_))[-15:]


Xtr = Xtr[:, top_features]
Xts = Xts[:, top_features]
monotonic_constraints = monotonic_constraints[top_features]

accs = [
    run_exp(Xtr, Ytr, Xts, Yts, monotonic_constraints, 4, 2, i) for i in range(3)
]  # 3 seeds
# print mean and std of the 3 runs
print(f"mean: {np.mean(accs):.4f}, std: {np.std(accs):.8f}")
