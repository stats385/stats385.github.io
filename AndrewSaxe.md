## High-dimensional dynamics of training and generalization in neural networks: insights from the linear case

What is the effect of depth on learning dynamics in neural networks? What interplay of dynamics, architecture, and data make good generalization possible in overparameterized networks? A partial answer comes from studying a simple tractable case: deep linear neural networks. I will describe exact solutions to the dynamics of learning which specify how every weight in the network evolves over the course of training. The theory answers basic questions such as how learning speed scales with depth, and why unsupervised pretraining accelerates learning. Turning to generalization error, I use random matrix theory to analyze the "high-dimensional" regime, where the number of training examples is on the order or even less than the number of adjustable synapses. In this regime, good generalization is possible due to implicit regularization in the initialization and dynamics of gradient descent. Overtraining is worst at intermediate network sizes, when the effective number of free parameters equals the number of samples, and can be reduced by making a network smaller or larger. Finally, putting together results on training and generalization dynamics, I will describe a speed-accuracy trade-off between training speed and generalization performance in deep networks.

## References

1. Saxe, A. M., McClelland, J. L., & Ganguli, S. (2019) [A mathematical theory of semantic development in deep neural networks.](https://arxiv.org/abs/1810.10531) Proceedings of the National Academy of Sciences, 116(23), 11537?11546.
2. Advani*, M., & Saxe*, A. M (2017).  [High-dimensional dynamics of generalization error in neural networks.](https://arxiv.org/abs/1710.03667v1) *Equal contributions.
3. Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks.](https://arxiv.org/abs/1312.6120v3) ICLR, 2014.

![Jeffrey Pennington](/assets/img/AndrewSaxe.jpg)  

Dr. Andrew Saxe is a Sir Henry Dale Fellow in the Department of Experimental Psychology, University of Oxford. He was previously a Swartz Postdoctoral Fellow in Theoretical Neuroscience at Harvard University with Haim Sompolinsky, and he completed his PhD in Electrical Engineering at Stanford University, advised by Jay McClelland, Surya Ganguli, Andrew Ng, and Christoph Schreiner. His dissertation received the Robert J. Glushko Dissertation Prize from the Cognitive Science Society. His research focuses on the theory of deep learning and its applications to phenomena in neuroscience and psychology.

[back](./)
