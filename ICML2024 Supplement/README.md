# Smooth Monotonic Networks: Supplementary material

This supplement contains the source code for reproducing the experiments described in the paper and to perform new experiments, which is highly encouraged. Our original code will be distributed under the permissive **The MIT License**, which will be added later because the copyright holder name would break anonymity. It would be nice if you cite the paper if you use the code. 

The code will be cleaned up and moved to a proper public github repository incorporating feedback from the review process.

Feel free to conduct further experiments (e.g., change hyperparameters, other data sets, different auxiliary networks, different architectures using the SMM module, ...).

### Files
The original files are:
 - `MonotonicNN.py`: Implementation of monotonic neural networks
 - `MonotonicNNPaperUtils.py`: Utility functions for the experiments in the paper (e.g., the gradient-based training procedures)
 - `MonotonicNNFullyMonotoneUnivariateExperiments.py`: Experiments on the univariate fully monotone benchmark functions
 - `MonotonicNNFullyMonotoneMultivariateExperiments.py`: Experiments on the multivariate fully monotone benchmark functions
 - `MonotonicNNUCI.py`: Experiments on UCI data
 - `MonotonicNNEvaluate.ipynb`: For completeness, the notebook producing the statistical test results, figures and tables
 - `MonotonicNNFullyMonotoneExperimentsTimingICML.ipynb`: Timing experiments
 - `MonotonicNNFullyMonotoneUnivariateHyperExperiments.py`: SMM hyperparameter study (posthoc)
 - `MonotonicNNFullyMonotoneExperimentsActiveICML.ipynb`: Counting silent neurons
 
Adapted files to evaluate SMM on the experiemnts presented by [Nolte et al.](https://openreview.net/forum?id=w2P7fMy_RH)

 - `chest_classify_SMM.py`, `chest_classify_finetuning_SMM.py`, `chest_config.py`: chest xray experiments, execute the first two 
 - `compas_SMM.py`: compass experiments                                     
 - `loan_SMM.py`, `loan_mini_SMM.py`, `loan_exp_SMM.py`: loan experiments, execute the first two  
 - `auto_mpg_SMM.py`: Auto MPG experiments    
 - `blogfeedback_SMM.py`, `blogfeedback_mini_SMM.py` and `blogfeedback_exp_SMM.py`: blogfeedback experiments, execute the first two                                  
 - `heart_disease_SMM.py`and `heart_disease_SMM_sigmoid.py`: two different SMM architectures, adjust number of epochs in `heart_disease_SMM_sigmoid.py`

Directories with data (and models) for the experiemts from [Nolte et al.](https://openreview.net/forum?id=w2P7fMy_RH):

 - `loaders`: Helper functions for loading the data                                              
 - `data`: Tabular data files
 - `models`: Trained neural networks for the chest xray task
 - `xrays`: Directory for the chest xray data, which has to be downloaded and unzipped from [here](https://www.kaggle.com/datasets/nih-chest-xrays/sample).                             
 
### Dependencies
Among others, we compare against XGBoost, which can be installed via `pip install xgboost`, and the Hierarchical Lattice Layer, which can be installed via `pip install pmlayer`. The hompage is [here](https://ibm.github.io/pmlayer). We also consider [Lipschitz Monotonic Networks](https://github.com/niklasnolte/MonotonicNetworks) and the code for reproducing the experiments by Nolte et al. available [here](https://github.com/niklasnolte/monotonic_tests). For the chest xray experiments, the dataset has to be downloaded and unzipped from [here](https://www.kaggle.com/datasets/nih-chest-xrays/sample).


### SMM Models
The file `MonotonicNN.py` implements several models. `MonotonicNN` and `MonotonicNNAlt` implement the standard min-max (MM) architecture. The former is a simple implementation, the latter supports counting silent neurons. ``SmoothMonotonicNN`` and ``SmoothMonotonicNN`` implement the new SMM architecture. The first version is a simple implementation, the second supports partially monotone functions.  `SMM_MLP` implements the architecture with auxiliary network used for the UCI experiments. These implementations are not optimized for speed, a faster version of SMM can be found in `MonotonicNNFullyMonotoneExperimentsTiming.ipynb`.
