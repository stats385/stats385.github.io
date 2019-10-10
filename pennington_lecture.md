## The Surprising Simplicity of Overparameterized Deep Neural Networks

In the physical sciences, it has often proved fruitful to study large, complex system by means of high-dimensional, mean-field, or other asymptotic approximations. For example, when the number of interacting particles in a thermodynamic system is large, exact microscopic descriptions are intractable, but macroscopic quantities such as temperature and pressure can nevertheless provide a useful characterization of its properties. Another modern example is the 1/N expansion in quantum field theory, in which the dimensionality N of an internal symmetry group is taken to be large, often leading to simplifications and unexpected connections such as AdS/CFT dualities.

In this talk, I will argue that one of the most promising directions in the development of a theory of deep learning is through a similar kind of approximation, namely one in which the number of parameters is taken to be large. In this limit, I will show that: (1) the prior over functions induced by common weight initialization schemes corresponds to a Gaussian process with a well-defined compositional kernel; (2) by tuning initialization hyperparameters, this kernel can be optimized for signal propagation, yielding networks that are trainable to enormous depths (10k+ layers); and (3) the learning dynamics of such overparameterized neural networks are governed by a linear model obtained from the first-order Taylor expansion of the network around its initial parameters.

![Jeffrey Pennington](/assets/img/pennington.jpg)  

Jeffrey Pennington is a Research Scientist at Google Brain, New York City. Prior to this, he was a postdoctoral fellow at Stanford University, as a member of the Stanford Artificial Intelligence Laboratory in the Natural Language Processing (NLP) group. He received his Ph.D. in theoretical particle physics from Stanford University while working at the SLAC National Accelerator Laboratory.

Jeffrey's research interests are multidisciplinary, ranging from the development of calculational techniques in perturbative quantum field theory to the vector representation of words and phrases in NLP to the study of trainability and expressivity in deep learning. Recently, his work has focused on building a set of theoretical tools with which to study deep neural networks. Leveraging techniques from random matrix theory and free probability, Jeffrey has investigated the geometry of neural network loss surfaces and the learning dynamics of very deep neural networks. He has also developed a new framework to begin harnessing the power of random matrix theory in applications with nonlinear dependencies, like deep learning.



[back](./)
