from loaders.blog_loader import load_data, mono_list
import numpy as np
from sklearn.linear_model import Ridge

from blogfeedback_exp_SMM import run_exp


Xtr, Ytr, Xts, Yts = load_data(get_categorical_info=False)
monotone_constraints = np.array(
    [1 if i in mono_list else 0 for i in range(Xtr.shape[1])]
)


model = Ridge()
model.fit(
    Xtr, Ytr,
)
rmse = np.sqrt(np.mean((model.predict(Xts) - Yts) ** 2))
important_feature_idxs = np.argsort(model.coef_)[::-1][:20]

Xtr = Xtr[:, important_feature_idxs]
Xts = Xts[:, important_feature_idxs]
monotone_constraints = monotone_constraints[important_feature_idxs]

rmses = [
    run_exp(
        Xtr,
        Ytr,
        Xts,
        Yts,
        monotone_constraints,
        max_lr=5e-4,
        expwidth=3,
        depth=2,
        batchsize=2 ** 8,
        seed=i,
        Lip=1,
    )
    for i in range(3)
]
print(f"mean: {np.mean(rmses):.5f}, std: {np.std(rmses):.5f}")
