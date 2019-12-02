## Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra

Numerous researchers recently applied empirical spectral analysis to the study of modern deep learning classifiers, observing spectral outliers and small but distinct bumps often seen beyond the edge of a "main bulk". This talk presents an important formal class/cross-class structure and shows how it lies at the origin of these visually striking patterns. The structure is shown to permeate the spectra of deepnet features, backpropagated errors, gradients, weights, Fisher Information matrix and Hessian, whether these are considered in the context of an individual layer or the concatenation of them all. The significance of the structure is illustrated by (i) proposing a correction to KFAC, a well known second-order optimization algorithm for training deepnets, and (ii) proving in the context of multinomial logistic regression that the ratio of outliers to bulk in the spectrum of the Fisher information matrix is predictive of misclassification.

## References

1. [The Full Spectrum of Deepnet Hessians at Scale: Dynamics with SGD Training and Sample Size](https://arxiv.org/abs/1811.07062), Vardan Papyan, 2018.

2. [Measurements of Three-Level Hierarchical Structure in the Outliers in the Spectrum of Deepnet Hessians](http://proceedings.mlr.press/v97/papyan19a.html), Vardan Papyan, ICML 2019.

![Vardan Papyan](/assets/img/VardanPapyan.png)  

Vardan Papyan received his B.Sc. degree in 2013 from the Computer Science Department at the Technion - Israel Institute of Technology, Haifa, Israel. He later received his M.Sc. and Ph.D. in 2017 from the same department, under the supervision of Prof. Michael Elad. He is currently a postdoctoral researcher, advised by Prof. David Donoho, at the Department of Statistics at Stanford University. His research interests include machine learning and more specifically deep learning, data science, signal and image processing, and high-dimensional statistics.

[Back](./)
