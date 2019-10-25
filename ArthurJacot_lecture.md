## Neural Tangent Kernel: Convergence and Generalization of DNNs

Modern deep learning has popularized the use of very large neural networks, but the theoretical tools to study such networks are still lacking. The Neural Tangent Kernel (NTK) describes how the output neutrons evolve during training. In the infinite width limit (when the number of hidden neutrons grows to infinity) the NTK converges to a deterministic and fixed limit, leading to a simple description of the dynamics of infinitely wide DNNs. The NTK is affected by the architecture of the network, and as such is helps understanding how architecture choices affect the convergence and generalization of DNNs.

As the depth of the network grows two regimes appear. A Freeze regime where the NTK is almost constant and convergence is slow and a Chaotic regime, where the NTK approaches a Kronecker delta, which speeds up training but may hurt generalization. Increasing the variance of the bias at initialization pushes the network towards the Freeze regime, while normalization methods such as Layer- and Batch-Normalization push the networks towards the Chaotic regime.

In GANs the Freeze regime leads to Mode Collapse, where the generator converge to a constant, and to checkerboard patterns, i.e. repeating patterns in images. Both problems are greatly reduced when the generator is chaotic, which may explain the importance of Batch Normalization in the training of GANs.

## References

1. Arora, S., Du, S. S., Hu, W., Li, Z., Salakhutdinov, R., and Wang, R. (2019). [On exact computation with an infinitely wide neural net](https://arxiv.org/abs/1904.11955). 

2. Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. (2014). [Generative Adversarial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), NeurIPS, 2014. 

3. Jacot, A., Gabriel, F., and Hongler, C. (2018). [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/pdf/1806.07572.pdf), NeurIPS, 2018.

4. Jacot, A., Gabriel, F., and Hongler, C. (2019). [The asymptotic spectrum of the hessian of DNN throughout training](https://arxiv.org/pdf/1910.02875.pdf).

5. Jacot, A., Gabriel, F., and Hongler, C. (2019). [Freeze and chaos for DNNs: an NTK view of batch normalization, checkerboard and boundary effects](https://arxiv.org/pdf/1907.05715.pdf), CoRR, 2019.

6. Poole, B., Lahiri, S., Raghu, M., Sohl-Dickstein, J., and Ganguli, S. (2016). [Exponential expressivity in deep neural networks through transient chaos](https://arxiv.org/abs/1606.05340), NeurIPS, 2016.

## Youtube Link

A nice [video](https://www.youtube.com/watch?v=raT2ECrvbag&t=131s) that explains the NTK.

![Arthur Jacot](/assets/img/ArthurJacot.png)  

Arthur Jacot is a PhD student in mathematics at the EPFL. After a Bachelor at the Freie Universitat Berlin, he finished his master at the EPFL. Since 2018, he works on the theory of deep neural networks at the Chair of Statistical Field Theory, supervised by Prof. Clement Hongler. He specializes in the study of infinitely wide networks, i.e. in the limit where the number of hidden neutrons grows to infinity. In this limit, the dynamics simplify and are described by a single object, the Neural Tangent Kernel, which was introduced by Arthur Jacot, Franck Gabriel and Clement Hongler in a 2018 NeurIPS paper.


[Back](./)
