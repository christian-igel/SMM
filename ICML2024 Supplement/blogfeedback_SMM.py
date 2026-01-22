from loaders.blog_loader import load_data, mono_list
import numpy as np

from blogfeedback_exp_SMM import run_exp

Xtr, Ytr, Xts, Yts = load_data(get_categorical_info=False)
monotone_constraints = np.array(
    [1 if i in mono_list else 0 for i in range(Xtr.shape[1])]
)


rmses = [
    run_exp(
        Xtr,
        Ytr,
        Xts,
        Yts,
        monotone_constraints,
        max_lr=2e-4,
        expwidth=3,
        depth=2,
        batchsize=2 ** 8,
        seed=i,
        Lip=1.5,
    )
    for i in range(3)
]
print(f"mean: {np.mean(rmses):.5f}, std: {np.std(rmses):.5f}")
