# Smooth Min-Max Monotonic Networks

Monotonicity constraints are powerful regularizers in statistical modelling. They can support fairness in computer-aided decision making and increase plausibility in data-driven scientific models. The seminal min-max (MM, [Still, 1997](https://papers.nips.cc/paper_files/paper/1997/hash/83adc9225e4deb67d7ce42d58fe5157c-Abstract.html)) neural network architecture ensures monotonicity, but often gets stuck in undesired local optima during training because of partial derivatives being zero when computing extrema. A simple modification of the MM network using strictly-increasing smooth minimum and maximum functions that alleviates this problem. The resulting smooth min-max (SMM) network module inherits the asymptotic approximation properties from the MM architecture. It can be used within larger deep learning systems trained end-to-end. The SMM module is conceptually simple and computationally less demanding than state-of-the-art neural networks for monotonic modelling. Experiments show that this does not come with a loss in generalization performance compared to alternative neural and non-neural approaches.

The directory [`ICML2014 Supplement`](ICML2014 Supplement) contains the code for reproducing the experiments presented in the ICML 2024 paper introducing SMM networks [[Igel, 2024](https://icml.cc/virtual/2024/poster/33186)].
The code is not very clean (e.g., containing several slightly different ways to implement the SMM).

A little bit cleaner (but not very efficient) are the implementations in:

 - [`SmoothMonotonicNN.py`](SmoothMonotonicNN.py): Implementation of SMM module, restricted to non-decreasing constraints and scalar output                                              
 - [`SMM_MLP.py`](SMM_MLP.py): Very simple example of how the SMM module can be combined with other layers

You can read about the approch here:

Christian Igel. [Smooth Min-Max Monotonic Networks](https://icml.cc/virtual/2024/poster/33186). *International Conference on Machine Learning*, 2024
