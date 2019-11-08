## Grossly Underdetermined Learning and Implicit Regularization

It is becoming increasingly clear that implicit regularization afforded by the optimization algorithms play a central role in machine learning, and especially so when using large, deep, neural networks. In this talk, I will present a view of deep learning which puts implicit regularization front and center: under this view, we consider deep learning as searching over the space of all functions, where the inductive bias is entirely controlled by the search geometry.  I will make this view concrete by discussing implicit regularization for matrix factorization,  linear convolutional networks, and two-layer ReLU networks, as well as a general bottom-up understanding on implicit regularization in terms of optimization geometry.  I will also use this view to explicitly contrast the power of deep learning with that of kernel methods, referring also to recent work on the role of the tangent kernel in deep learning.

## References

Here's a roadmap of the relevant papers:

An older paper that takes a higher level view of what might be going on and what we want to try to achieve is [https://arxiv.org/abs/1705.03071](https://arxiv.org/abs/1705.03071)

Gradient descent on logistic regression leads to max margin: [https://arxiv.org/abs/1710.10345](https://arxiv.org/abs/1710.10345)
(two very technical papers refine the exact rates and conditions are [https://arxiv.org/abs/1806.01796](https://arxiv.org/abs/1806.01796) and [https://arxiv.org/abs/1803.01905](https://arxiv.org/abs/1803.01905) --- I will not be discussing these results directly)

Implicit regularization in matrix factorization: [https://arxiv.org/abs/1705.09280](https://arxiv.org/abs/1705.09280) , and a follow-up paper: [https://arxiv.org/abs/1712.09203](https://arxiv.org/abs/1712.09203)
Relationship to NTK and elaboration of the techniques: [https://arxiv.org/abs/1906.05827](https://arxiv.org/abs/1906.05827)

General implicit reg framework and relation to optimization geometry: [https://arxiv.org/abs/1802.08246](https://arxiv.org/abs/1802.08246)

Implicit regularization in linear conv nets: [https://arxiv.org/abs/1806.00468](https://arxiv.org/abs/1806.00468)
Generalization of the above ideas: [https://arxiv.org/abs/1905.07325](https://arxiv.org/abs/1905.07325)

Inductive bias in infinite-width ReLU networks, in one dimension: [https://arxiv.org/abs/1902.05040](https://arxiv.org/abs/1902.05040)
In higher dimensions: [https://arxiv.org/abs/1910.01635](https://arxiv.org/abs/1910.01635)


![Nati Srebro](/assets/img/NatiSrebro.jpg)  

Nati (Nathan) Srebro is a professor at the Toyota Technological Institute at Chicago, with  cross-appointments at the University of Chicago Dept. of Computer Science and Committee on Computational and Applied Mathematics.  He obtained his PhD at the Massachusetts Institute of Technology (MIT) in 2004, and previously was a post-doctoral fellow at the University of Toronto, a Visiting Scientist at IBM, and an Associate Professor of the Technion.  Prof. Srebro’s research encompasses methodological, statistical and computational aspects of Machine Learning, as well as related problems in Optimization. Some of Prof. Srebro’s significant contributions include work on learning “wider” Markov networks; introducing the use of the nuclear norm for machine learning and matrix reconstruction; work on fast optimization techniques for machine learning, and on the relationship between learning and optimization.  His current interests include understanding deep learning through a detailed understanding of optimization; distributed and federated learning; algorithmic fairness and practical adaptive data analysis.

[Back](./)
