# Smooth Min-Max Monotonic Networks

Monotonicity constraints are powerful regularizers in statistical modelling. They can support fairness in computer-aided decision making and increase plausibility in data-driven scientific models. The seminal min-max (MM, [Still, 1997](https://papers.nips.cc/paper_files/paper/1997/hash/83adc9225e4deb67d7ce42d58fe5157c-Abstract.html)) neural network architecture ensures monotonicity, but often gets stuck in undesired local optima during training because of partial derivatives being zero when computing extrema. 

We [Igel, 2024](https://icml.cc/virtual/2024/poster/33186) propose a simple modification of the MM network using strictly-increasing smooth minimum and maximum functions that alleviates this problem. The resulting smooth min-max (SMM) network module inherits the asymptotic approximation properties from the MM architecture. It can be used within larger deep learning systems trained end-to-end. The SMM module is conceptually simple and computationally less demanding than state-of-the-art neural networks for monotonic modelling. Our experiments [Igel, 2024](https://icml.cc/virtual/2024/poster/33186) show that this does not come with a loss in generalization performance compared to alternative neural and non-neural approaches.

The directory [ICML2014 Supplement}(ICML2014 Supplement) contain the code for reproducing the experimnts.
Sorry, I did not have the time to implement 





Christian Igel. Smooth Min-Max Monotonic Networks. *International Conference on Machine Learning*, 2024
