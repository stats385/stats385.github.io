<div class="abstract">   
    <strong>Analysis of Sufficiency, Minimality and Invariance of Learned Representations</strong>
    <p align="justify">
This lecture will be divided into 3 parts: 


In the first 15', I will describe desiderata for representations, i.e. functions of the data that are “useful” for a given task. Of all functions, we seek those that are sufficient (as “informative” as the data), invariant to nuisance factors, and minimal - in a sense to be properly defined. Such desiderata will be converted into a variational principle, that however cannot be optimized directly since we are interested in representations of future (test) data that we have no access to. This part of the lecture is not specific to deep learning. The reference reading are Sections 2 and 3 of [1].


In the second part (30'), I will give an overview of Deep Learning through the lens of PAC-Bayes and introduce the Information Lagrangian. This is a function of the given (training, or past) data that, if optimized, would yield a parametric model with generalization bound. The complexity term in the Information Lagrangian measures the Information in deep networks, a concept that has caused confusion among some researchers, since a network, once trained, is a deterministic function - this is resolved in [2]. I will introduce a general notion of Information in the Weights, and show that, depending on the choice of coding prior, it encompasses the Fisher Information of the weights (uninformative prior), the Shannon Information of the weights (prior = marginal over all training sets), or the Kolmogorov Complexity of the weights (prior = computable functions). While the arbitrary choice of prior may seem a shortcoming of the theory, it is critical since Shannon’s information — which is not computable in practice  — relates to generalization whereas the Fisher Information is optimized by stochastic gradient descent (SGD). We introduce two key results: The first shows that, when optimizing a deep network with SGD, the Fisher and the Shannon are tightly coupled. The second, known as the Emergence Bound, shows that minimizing the Fisher (and hence Shannon) information of the weights also  minimizes the Effective Information of the activations, which is shown to correspond to invariance of the activations of *future* data to nuisance variables. Thus, the Emergence bound links desirable properties of the representation of future data (activations of the test datum), to computable properties of the training set (empirical cross-entropy and Fisher Information of the weights). The reference reading for this part are Sections 3 and 4 of [2].


The first two parts address the question of optimal representations for a given task. However, it is customary in deep learning to train a model for a task (say, finding cats and dogs in ImageNet) and then use it for another (say, finding tumors in a mammogram) after fine-tuning. Remarkably, sometimes this transfer learning works, but we wish to predict whether it will work without actually having to try it. To address these questions, we need at least to formalize the notion of learning task, and define some sort of topology in the space of tasks, so we can reason about relations between tasks.


In the third part (15'), I will use the Information in the Weights, developed in the previous part, to define an asymmetric distance between tasks that correlates highly with ease of fine-tuning. Not only it is possible to determine how “far” two learning tasks are, but one can also predict how many computational resources will be needed to fine-tune from one task to another. This picture, however, has a wrinkle. There exist tasks that are very close to each other, yet it is not possible to fine-tune one from the other. I will show an empirical phenomenon whereby pre-training not only not beneficial, but detrimental. This  is observed widely in different architectures, optimizations, tasks, learning rates, and even in biological systems. It puts the emphasis on the role of the transient dynamics of differential learning. I will introduce a notion of Dynamic Distance between learning tasks, that accounts for the global geometry of the loss landscape, through an approximation of the probability over paths between any two points in the loss landscape, and a notion of Task Reachability.  The reference reading is [4].


The practical implications of the theory are (a) a system to perform model recommendation by computing the distance between a given task and a large number of pre-trained tasks, without actually running any fine-tuning experiments, by embedding each task in a linear space (Task2Vec, [6]), and (b) the discovery that regularization in deep networks does not work by smoothing the loss landscape, but by influencing the early stages of convergence [7], leaving many questions and opportunities for investigation open.
</p>
</div>

<div class="abstract">   
    <strong>References</strong>
    <p align="justify">
[1] A. Achille and S. Soatto, Emergence of Invariance and Disentanglement in Deep Representations, JMLR 2018, https://arxiv.org/pdf/1706.01350.pdf
        
        
[2] A. Achille and S. Soatto, Where is the Information in a Deep Neural Network? https://arxiv.org/pdf/1905.12213.pdf 


[3] (optional) A. Achille et al., The Information Complexity of Learning Tasks, their Structure and their Distance https://arxiv.org/pdf/1904.03292.pdf


[4] A. Achille, M. Rovere and S. Soatto, Critical Learning Periods in Deep Neural Networks, ICLR 2019, https://arxiv.org/pdf/1711.08856.pdf


[5] (optional) A. Achille, G. Mbeng an S. Soatto, Dynamics and Reachability of Learning Tasks, https://arxiv.org/abs/1810.02440


[6] A. Achille et al., Task2Vec, Task Embedding for Meta Learning, ICCV 2019, https://arxiv.org/pdf/1902.03545.pdf


[7] A. Golaktar et al., Time Matters in Regularizing Deep Networks: Weight Decay and Data Augmentation Affect Early Learning Dynamics, Matter Little Near Convergence, NeurIPS 2019, https://arxiv.org/pdf/1905.13277.pdf
</p>
</div>



![Stefano Soatto](/assets/img/StefanoSoatto.jpg)  

[Professor Stefano Soatto](www.cs.ucla.edu/~soatto) received his Ph.D. in Control and Dynamical Systems from the California Institute of Technology in 1996; he joined UCLA in 2000 after being Assistant and then Associate Professor of Electrical and Biomedical Engineering at Washington University, and Research Associate in Applied Sciences at Harvard University. Between 1995 and 1998 he was also Ricercatore in the Department of Mathematics and Computer Science at the University of Udine - Italy. He received his D.Ing. degree (highest honors) from the University of Padova- Italy in 1992.

His general research interests are in Computer Vision and Nonlinear Estimation and Control Theory. In particular, he is interested in ways for computers to use sensory information (e.g. vision, sound, touch) to interact with humans and the environment.  

Dr. Soatto is the recipient of the David Marr Prize (with Y. Ma, J. Kosecka and S. Sastry of U.C. Berkeley) for work on Euclidean reconstruction and reprojection up to subgroups. He also received the Siemens Prize with the Outstanding Paper Award from the IEEE Computer Society for his work on optimal structure from motion (with R. Brockett of Harvard). He received the National Science Foundation Career Award and the Okawa Foundation Grant. He is Associate Editor of the IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI) and a Member of the Editorial Board of the International Journal of Computer Vision (IJCV) and Foundations and Trends in Computer Graphics and Vision.



[back](./)
