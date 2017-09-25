---
layout: default
---

## Networks
<strong>Neocognitron</strong>
<p align="justify">
A hierarchical multi-layered neural network, proposed by Kunihiko Fukushima in 1982.
It has been used for handwritten character recognition and other pattern recognition tasks.
Since backpropagation had not yet been applied for training neural nets at the time, it was limited by the lack of a training algorithm.<br />
<a href="https://ml4a.github.io/ml4a/convnets/"> source </a>
</p>

<strong><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf"> LeNet-5 </a></strong>
<p align="justify">
A pioneering digit classification neural network by LeCun et. al.
It was applied by several banks to recognise hand-written numbers on checks.
The network was composed of three types layers: convolution, pooling and non-linearity.<br />
<a href="https://en.wikipedia.org/wiki/Convolutional_neural_network"> source </a><br />
<a href="https://www.youtube.com/watch?v=FwFduRA_L6Q"> link to a demo of LeNet from 1993 </a>
</p>

<strong><a href="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"> AlexNet </a></strong>
<p align="justify">
A convolutional neural network, which competed in the ImageNet Large Scale Visual Recognition Challenge in 2012 and achieved a top-5 error of 15.3%, more than 10.8% ahead of the runner up.
AlexNet was designed by Alex Krizhevsky, Geoffrey Hinton, and Ilya Sutskever.
The network consisted of five convolutional layers, some of which were followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax.
All the convolutional and fully connected layers were followed by the ReLU nonlinearity.<br />
<a href="https://en.wikipedia.org/wiki/AlexNet"> source </a>
</p>

<strong><a href="https://arxiv.org/pdf/1409.1556.pdf"> VGGNet </a></strong>
<p align="justify">
A 19 layer convolutional neural network from VGG group, Oxford, that was simpler and deeper than AlexNet.
All large-sized filters in AlexNet were replaced by cascades of 3x3 filters (with nonlinearity in between).
Max pooling was placed after two or three convolutions and after each pooling the number of filters was always doubled.
</p>

<strong><a href="https://arxiv.org/pdf/1512.03385.pdf"> ResNet </a></strong>
<p align="justify">
Developed by Microsoft Research, ResNet won first place in ILSVRC 2015 image classification using a 152-layer network -- 8 times deeper than the VGG.
The basic element in this architecture is the residual block, which	contains two paths between the input and the output, one of them being direct.
This forces the network to learn the features on top of already available input, and facilitates the optimization process.
</p>

## Network Architectures
<strong>Convolutional neural network (CNN)</strong>
<p align="justify">
A CNN is a multi-layer neural network constructed from convolutional, pooling and fully connected layers.
Convolutional layers apply a convolution operation to the input, passing the result to the next layer.
The weights in the convolutional layers are shared, which means that the same filter bank is used in all the spatial locations.<br />
<a href="https://en.wikipedia.org/wiki/Convolutional_neural_network"> source </a>
</p>

<strong>Deconvolutional networks</strong>
<p align="justify">
A generative network that is a special kind of convolutional network that uses transpose convolutions, also known as a deconvolutional layers.
</p>

<strong><a href="https://arxiv.org/pdf/1406.2661.pdf"> Generative Adversarial Networks (GAN) </a></strong>
<p align="justify">
A system of two neural networks, introduced by Ian Goodfellow et al. in 2014, contesting with each other in a zero-sum game framework.
The first is a deconvolutional network that generates signals.
While the second is a classifier that learns to discriminates between signals from the true data distribution and fake ones produced by the generator.
The generative network's goal is to increase the error rate of the discriminative network by fooling it with synthesized examples that appear to have come from the true data distribution.<br />
<a href="https://en.wikipedia.org/wiki/Generative_adversarial_network"> source </a>
</p>

<strong>Recurrent neural networks (RNN)</strong>
<p align="justify">
RNNs are built on the same computational unit as the feed forward neural net, but differ in the way these are connected.
Feed forward neural networks are organized in layers, where information flows in one direction -- from input units to output units -- and no cycles are allowed.
RNNs, on the other hand, do not have to be organized in layers and directed cycles are allowed.
This allows them to have internal memory and as a result to process sequential data.
</p>

## Architectural Components/Motifs
<strong>Depth of a network
<p align="justify">
The number of layers in the network.
</p>

<strong>Feature vector / representation / volume</strong>
<p align="justify">
A three dimensional tensor of size
<a href="http://www.codecogs.com/eqnedit.php?latex=W&space;\times&space;H&space;\times&space;D" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W&space;\times&space;H&space;\times&space;D" title="W \times H \times D" /></a>
obtained in a certain layer of a neural network.
W is the width, H is the height and D is the depth, i.e., the number of channels.
If there is more than one example, this becomes a four dimensional tensor of size
<a href="http://www.codecogs.com/eqnedit.php?latex=W&space;\times&space;H&space;\times&space;D&space;\times&space;B" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W&space;\times&space;H&space;\times&space;D&space;\times&space;B" title="W \times H \times D \times B" /></a>
, where
<a href="http://www.codecogs.com/eqnedit.php?latex=B" target="_blank"><img src="http://latex.codecogs.com/gif.latex?B" title="B" /></a>
is the batch size.
</p>

<strong>Filters and biases</strong>
<p align="justify">
Filters are a four dimensional tensor of size

<a href="http://www.codecogs.com/eqnedit.php?latex=F&space;\times&space;F&space;\times&space;D&space;\times&space;K" target="_blank"><img src="http://latex.codecogs.com/gif.latex?F&space;\times&space;F&space;\times&space;D&space;\times&space;K" title="F \times F \times D \times K" /></a>

and biases are a vector of length

<a href="http://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K" title="K" /></a>

.

<a href="http://www.codecogs.com/eqnedit.php?latex=F" target="_blank"><img src="http://latex.codecogs.com/gif.latex?F" title="F" /></a>

is the width and height of the filter,
 
<a href="http://www.codecogs.com/eqnedit.php?latex=D" target="_blank"><img src="http://latex.codecogs.com/gif.latex?D" title="D" /></a>

is the number of channels and 
 
<a href="http://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K" title="K" /></a>

 is the number of filters.
</p>

<strong>Convolutional (Conv) layer</strong>
<p align="justify">
Accepts as input:<br />
<ul style="list-style-type:circle">
	<li>feature vector of size
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=W_1&space;\times&space;H_1&space;\times&space;D_1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W_1&space;\times&space;H_1&space;\times&space;D_1" title="W_1 \times H_1 \times D_1" /></a>
	
	</li>
	<li>filters of size 
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=F&space;\times&space;F&space;\times&space;D_1&space;\times&space;D_2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?F&space;\times&space;F&space;\times&space;D_1&space;\times&space;D_2" title="F \times F \times D_1 \times D_2" /></a>
	
	</li>
	<li>biases of length 
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=D_2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?D_2" title="D_2" /></a>
	
	</li>
	<li>stride 
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=S" target="_blank"><img src="http://latex.codecogs.com/gif.latex?S" title="S" /></a>
	
	</li>
	<li>amount of zero padding 
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=P" target="_blank"><img src="http://latex.codecogs.com/gif.latex?P" title="P" /></a>
	
	</li>
</ul>  
Outputs another feature vector of size 

<a href="http://www.codecogs.com/eqnedit.php?latex=W_2&space;\times&space;H_2&space;\times&space;D_2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W_2&space;\times&space;H_2&space;\times&space;D_2" title="W_2 \times H_2 \times D_2" /></a>

, where
<ul style="list-style-type:circle">
	<li>
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=W_2&space;=&space;\frac{W_1-F&plus;2P}{S}&plus;1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W_2&space;=&space;\frac{W_1-F&plus;2P}{S}&plus;1" title="W_2 = \frac{W_1-F+2P}{S}+1" /></a>
	
	</li>
	<li>
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=H_2&space;=&space;\frac{H_1-F&plus;2P}{S}&plus;1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?H_2&space;=&space;\frac{H_1-F&plus;2P}{S}&plus;1" title="H_2 = \frac{H_1-F+2P}{S}+1" /></a>
	
	</li>
</ul>  
The d-th channel in the output feature vector is obtained by performing a valid convolution with stride

<a href="http://www.codecogs.com/eqnedit.php?latex=S" target="_blank"><img src="http://latex.codecogs.com/gif.latex?S" title="S" /></a>

of the d-th filter and the padded input.<br />
<a href="http://cs231n.github.io/convolutional-networks/"> source </a>
</p>

<strong>Stride</strong>
<p align="justify">
The amount by which a filter shifts spatially when convolving it with a feature vector.<br />
<a href="http://cs231n.github.io/convolutional-networks/"> source </a>
</p>

<strong>Dilation</strong>
<p align="justify">
A filter is dilated by a factor

<a href="http://www.codecogs.com/eqnedit.php?latex=Q" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Q" title="Q" /></a>

by inserting in every one of its channels independently

<a href="http://www.codecogs.com/eqnedit.php?latex=Q-1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Q-1" title="Q-1" /></a>

zeros between the filter elements.<br />
<a href="http://cs231n.github.io/convolutional-networks/"> source </a>
</p>

<strong>Fully connected (FC) layer</strong>
<p align="justify">
In practice, FC layers are implemented using a convolutional layer.
To see how this might be possible, note that when an input feature vector of size 

<a href="http://www.codecogs.com/eqnedit.php?latex=H&space;\times&space;W&space;\times&space;D_1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?H&space;\times&space;W&space;\times&space;D_1" title="H \times W \times D_1" /></a>

is convolved with a filter bank of size

<a href="http://www.codecogs.com/eqnedit.php?latex=H&space;\times&space;W&space;\times&space;D_1&space;\times&space;D_2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?H&space;\times&space;W&space;\times&space;D_1&space;\times&space;D_2" title="H \times W \times D_1 \times D_2" /></a>

, it results in an output feature vector of size

<a href="http://www.codecogs.com/eqnedit.php?latex=1&space;\times&space;1&space;\times&space;D_2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?1&space;\times&space;1&space;\times&space;D_2" title="1 \times 1 \times D_2" /></a>

.
Since the convolution is valid and the filter can not move spatially, the operation is equivalent to a fully connected one.
More over, when this feature vector of size 1x1xD_2 is convolved with another filter bank of size 

<a href="http://www.codecogs.com/eqnedit.php?latex=1&space;\times&space;1&space;\times&space;D_2&space;\times&space;D_3" target="_blank"><img src="http://latex.codecogs.com/gif.latex?1&space;\times&space;1&space;\times&space;D_2&space;\times&space;D_3" title="1 \times 1 \times D_2 \times D_3" /></a>

, the result is of size

<a href="http://www.codecogs.com/eqnedit.php?latex=1&space;\times&space;1&space;\times&space;D_3" target="_blank"><img src="http://latex.codecogs.com/gif.latex?1&space;\times&space;1&space;\times&space;D_3" title="1 \times 1 \times D_3" /></a>

.
In this case, again, the convolution is done over a single spatial location and therefore equivalent to a fully connected layer.<br />
<a href="http://cs231n.github.io/convolutional-networks/"> source </a>
</p>

<strong>Pooling layer</strong>
<p align="justify">
Accepts as input:
<ul style="list-style-type:circle">
	<li>feature vector of size
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=W_1&space;\times&space;H_1&space;\times&space;D_1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W_1&space;\times&space;H_1&space;\times&space;D_1" title="W_1 \times H_1 \times D_1" /></a>
	
	</li>
	<li>size of neighbourhood
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=F" target="_blank"><img src="http://latex.codecogs.com/gif.latex?F" title="F" /></a>
	
	</li>
	<li>stride S</li>
</ul>  
Outputs another feature vector of size

<a href="http://www.codecogs.com/eqnedit.php?latex=W_2&space;\times&space;H_2&space;\times&space;D_1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W_2&space;\times&space;H_2&space;\times&space;D_1" title="W_2 \times H_2 \times D_1" /></a>

, where
Accepts as input:
<ul style="list-style-type:circle">
	<li>
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=W_2&space;=&space;\frac{W_1&space;-&space;F}{S}&space;&plus;&space;1&plus;&space;1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W_2&space;=&space;\frac{W_1&space;-&space;F}{S}&space;&plus;&space;1&plus;&space;1" title="W_2 = \frac{W_1 - F}{S} + 1+ 1" /></a>
	
	</li>
	<li>
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=H_2&space;=&space;\frac{H_1&space;-&space;F}{S}&space;&plus;&space;1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?H_2&space;=&space;\frac{H_1&space;-&space;F}{S}&space;&plus;&space;1" title="H_2 = \frac{H_1 - F}{S} + 1" /></a>
	
	</li>
</ul>  
The pooling resizes independently every channel of the input feature vector by applying a certain function on neighbourhoods of size

<a href="http://www.codecogs.com/eqnedit.php?latex=F&space;\times&space;F" target="_blank"><img src="http://latex.codecogs.com/gif.latex?F&space;\times&space;F" title="F \times F" /></a>

, with a stride

<a href="http://www.codecogs.com/eqnedit.php?latex=S" target="_blank"><img src="http://latex.codecogs.com/gif.latex?S" title="S" /></a>

.<br />
<a href="http://cs231n.github.io/convolutional-networks/"> source </a>
</p>

<strong>Max pooling</strong>
<p align="justify">
Picks the maximal value from every neighbourhood.
</p>

<strong>Average pooling</strong>
<p align="justify">
Computes the average of every neighbourhood.
</p>

<strong>Linear classifier</strong>
<p align="justify">
This is implemented in practice by employing a fully connected layer of size

<a href="http://www.codecogs.com/eqnedit.php?latex=H&space;\times&space;W&space;\times&space;D&space;\times&space;C" target="_blank"><img src="http://latex.codecogs.com/gif.latex?H&space;\times&space;W&space;\times&space;D&space;\times&space;C" title="H \times W \times D \times C" /></a>

, where 

<a href="http://www.codecogs.com/eqnedit.php?latex=C" target="_blank"><img src="http://latex.codecogs.com/gif.latex?C" title="C" /></a>

is the number of classes.
Each one of the filters of size 

<a href="http://www.codecogs.com/eqnedit.php?latex=H&space;\times&space;W&space;\times&space;D" target="_blank"><img src="http://latex.codecogs.com/gif.latex?H&space;\times&space;W&space;\times&space;D" title="H \times W \times D" /></a>

corresponds to a certain class and there are 

<a href="http://www.codecogs.com/eqnedit.php?latex=C" target="_blank"><img src="http://latex.codecogs.com/gif.latex?C" title="C" /></a>

classifiers, one for each class.
</p>

<strong>Neighbourhood</strong>
<p align="justify">
A group of consecutive entries in a two-dimensional signal that has a rectangular or a square shape.
</p>

<strong>Spatial invariant feature vector</strong>
<p align="justify">
A feature vector that remains unchanged even if the input to the network is spatially translated.
</p>

<strong>Saturation of activation</strong>
<p align="justify">
An activation that has an almost zero gradient at certain regions.
This is an undesirable property since it results in slow learning.
</p>

<strong>Sigmoid</strong>
<p align="justify">
The sigmoid, defined as

<a href="http://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-x}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-x}}" title="f(x) = \frac{1}{1 + e^{-x}}" /></a>

,
is a non-linear function that suffers from saturation.
</p>

<strong>Tanh</strong>
<p align="justify">
This non-linearity squashes a real-valued number to the range

<a href="http://www.codecogs.com/eqnedit.php?latex=[-1,&space;1]" target="_blank"><img src="http://latex.codecogs.com/gif.latex?[-1,&space;1]" title="[-1, 1]" /></a>

.
Like the sigmoid neuron, its activations saturate, but unlike the sigmoid neuron its output is zero-centered.
</p>

<strong>ReLu</strong>
<p align="justify">
The most popular non-linearity in modern deep learning, partly due to its non-saturating nature, defined as

<a href="http://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\max(x,0)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f(x)&space;=&space;\max(x,0)" title="f(x) = \max(x,0)" /></a>

.
</p>

<strong>Dead filter</strong>
<p align="justify">
A filter which always results in negative values that are mapped by ReLU to zero, no matter what the input is.
This causes backpropagation to never update the filter and eventually, due to weight decay, it becomes zero and "dies".
</p>

<strong>Leaky ReLu</strong>
<p align="justify">
A possible fix to the dead filter problem is to define ReLU with a small slope in the negative part, i.e.,

<a href="http://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\left\{\begin{array}{lr}&space;ax,&space;&&space;\text{for&space;}&space;x<0\\&space;x,&space;&&space;x&space;\geq&space;0&space;\end{array}\right\}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f(x)&space;=&space;\left\{\begin{array}{lr}&space;ax,&space;&&space;\text{for&space;}&space;x<0\\&space;x,&space;&&space;x&space;\geq&space;0&space;\end{array}\right\}" title="f(x) = \left\{\begin{array}{lr} ax, & \text{for } x<0\\ x, & x \geq 0 \end{array}\right\}" /></a>

.
</p>

<strong>Batch normalization</strong>
<p align="justify">
Accepts as input:
<ul style="list-style-type:circle">
	<li>feature vector of size
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=W&space;\times&space;H&space;\times&space;D" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W&space;\times&space;H&space;\times&space;D" title="W \times H \times D" /></a>
	
	</li>
	<li>bias vector of size
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=D" target="_blank"><img src="http://latex.codecogs.com/gif.latex?D" title="D" /></a>
	
	</li>
	<li>gain vector of size
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=D" target="_blank"><img src="http://latex.codecogs.com/gif.latex?D" title="D" /></a>
	
	</li>
</ul>  
Outputs another feature vector of the same size.
This layer operates on each channel of the feature vector independently.
First, each channel is normalized to have a zero mean, unit variance and then it is multiplied by a gain and shifted by a bias.
The purpose of this layer is to ease the optimization process.
</p>

<strong>Softmax layer</strong>
<p align="justify">
Takes the output of the classifier, applies exponent on the score assigned to each class and then normalizes the result to unit sum.
The result can be interpreted as a vector of probabilities for the different classes.
</p>

<strong>One-hot vector</strong>
<p align="justify">
A vector containing one in a single entry and zero elsewhere.
</p>

<strong>Cross entropy</strong>
<p align="justify">
Commonly used to quantify the difference between two probability distributions.
In the case of neural networks, one of the distributions is the output of the softmax, while the other is a one-hot vector corresponding to the correct class.
</p>

<strong>Data preprocessing</strong>
<p align="justify">
The input to a neural network is often mean subtracted, contrast normalized and whitened.
</p>

<strong>Initialization of a network</strong>
<p align="justify">
Usually, the biases of a neural network are set to zero, while the weights are initialized with independent and identically distributed zero-mean Gaussian noise.
The variance of the noise is chosen in such a way that the magnitudes of input signals does not change drastically.<br />
<a href="https://arxiv.org/pdf/1502.01852.pdf"> source </a>
</p>

## Regularization in Neural Networks
<strong><a href="http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf"> Dropout </a></strong>
<p align="justify">
Accepts as input:
<ul style="list-style-type:circle">
	<li>feature vector of size
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=H&space;\times&space;W&space;\times&space;D" target="_blank"><img src="http://latex.codecogs.com/gif.latex?H&space;\times&space;W&space;\times&space;D" title="H \times W \times D" /></a>
	
	</li>
	<li>probability 
	
	<a href="http://www.codecogs.com/eqnedit.php?latex=p" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p" title="p" /></a>
	
	</li>
</ul>  
Outputs another feature vector of the same size.
At train time, every neuron in it is set to the value of the corresponding neuron in the input with probability 

<a href="http://www.codecogs.com/eqnedit.php?latex=p" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p" title="p" /></a>

, and zero otherwise.
At test time, the output feature vector is equal to the input one scaled by

<a href="http://www.codecogs.com/eqnedit.php?latex=p" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p" title="p" /></a>

.
</p>

<strong>Weight decay</strong>
<p align="justify">
Soft

<a href="http://www.codecogs.com/eqnedit.php?latex=L_2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?L_2" title="L_2" /></a>

constraint on the parameters of the network.
This is done by decreasing every parameter in each iteration of SGD by its value times a small constant, corresponding to the strength of the regularization.
</p>

<strong>Max norm constraints</strong>
<p align="justify">
Hard

<a href="http://www.codecogs.com/eqnedit.php?latex=L_2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?L_2" title="L_2" /></a>

constraint on the parameters of the network.
This is done by imposing an upper bound on the

<a href="http://www.codecogs.com/eqnedit.php?latex=L_2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?L_2" title="L_2" /></a>

norm of every filter and using projected gradient descent to enforce the constraint.<br />
<a href="http://cs231n.github.io/convolutional-networks/"> source </a>
</p>

<strong>Data augmentation</strong>
<p align="justify">
Creating additional training samples by perturbing existing ones.
In image classification this includes randomly flipping the input, cropping subsets from it, etc.
</p>

## Learning Ideas
<strong>Gradient descent</strong>
<p align="justify">
To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient of the function at the current point.<br />
<a href="https://en.wikipedia.org/wiki/Gradient_descent"> source </a>
</p>

<strong>Stochastic gradient descent (SGD)</strong>
<p align="justify">
A stochastic approximation of the gradient descent for minimizing an objective function that is a sum of functions.
The true gradient is approximated by the gradient of a randomly chosen single function.<br />
<a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent"> source </a>
</p>

<strong>Learning rate</strong>
<p align="justify">
The scalar by which the negative of the gradient is multiplied in gradient descent.
</p>

<strong>Backpropagation</strong>
<p align="justify">
An algorithm, relying on an iterative application of the chain rule, for computing efficiently the derivative of a neural network with respect to all of its parameters and feature vectors.<br />
<a href="https://en.wikipedia.org/wiki/Backpropagation"> source </a>
</p>

<strong>Goal function</strong>
<p align="justify">
The function being minimized in an optimization process, such as SGD.
</p>

<strong>Added noise</strong>
<p align="justify">
A perturbation added to the input of the network or one of the feature vectors it computes.
</p>

## Datasets
<strong><a href="http://yann.lecun.com/exdb/mnist/"> MNIST </a></strong>
<p align="justify">
The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
The digits have been size-normalized and centered in a fixed-size 28x28 image.<br />
</p>

<strong><a href="https://www.cs.toronto.edu/~kriz/cifar.html"> CIFAR-10 </a></strong>
<p align="justify">
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
There are 50000 training images and 10000 test images.<br />
</p>

<strong><a href="http://www.image-net.org/"> ImageNet </a></strong>
<p align="justify">
A large image database that has over ten million URLs of images that were hand-annotated using Amazon Mechanical Turk to indicate what objects are pictured.<br />
<a href="https://en.wikipedia.org/wiki/ImageNet"> source </a>
</p>

## Contests

<strong><a href="http://www.image-net.org/challenges/LSVRC/"> ImageNet Large Scale Visual Recognition Challenge (ILSVRC) </a></strong>
<p align="justify">
A competition in which teams compete to obtain the highest accuracy on several computer vision tasks, such as image classification.<br />
<a href="https://en.wikipedia.org/wiki/ImageNet"> source </a>
</p>

## Personalities
<strong><a href="http://www.cs.toronto.edu/~hinton/"> Geoff Hinton </a></strong>
<p align="justify">
A cognitive psychologist and computer scientist, most noted for his work on artificial neural networks.
He was one of the first researchers who demonstrated the use of backpropagation algorithm for training multi-layer neural networks.<br />
<a href="https://en.wikipedia.org/wiki/Geoffrey_Hinton"> source </a>
</p>

<strong><a href="http://yann.lecun.com/"> Yann Lecun </a></strong>
<p align="justify">
A computer scientist with contributions in machine learning, computer vision, mobile robotics and computational neuroscience.
He is well known for his work on optical character recognition and computer vision using convolutional neural networks, and is a founding father of convolutional nets.<br />
<a href="https://en.wikipedia.org/wiki/Yann_LeCun"> source </a>
</p>

<strong><a href="http://www.iro.umontreal.ca/~bengioy/yoshua_en/"> Yoshua Bengio </a></strong>
<p align="justify">
A computer scientist, most noted for his work on artificial neural networks and deep learning.
He has contributed to a wide spectrum of machine learning areas and is well known for his theoretical results on recurrent neural networks, kernel machines, distributed representations, depth of neural architectures, and the optimization challenge of deep learning.
His work was crucial in advancing how deep networks are trained, how neural networks can learn vector embeddings for words, how to perform machine translation with deep learning by taking advantage of an attention mechanism, and how to perform unsupervised learning with deep generative models.<br />
<a href="https://www.creativedestructionlab.com/people/yoshua-bengio/"> source </a>
</p>

<strong><a href="http://vision.stanford.edu/feifeili/"> Fei Fei Li </a></strong>
<p align="justify">
A computer scientist whose main research areas are in machine learning, deep learning, computer vision and cognitive and computational neuroscience.
She is the inventor of ImageNet and the ImageNet Challenge, a critical large-scale dataset and benchmarking effort that has contributed to the latest developments in deep learning and AI.<br />
<a href="http://www.itu.int/en/ITU-T/AI/Pages/li.fei-fei.aspx"> source </a>
</p>

<strong><a href="http://www.andrewng.org/"> Andrew Ng </a></strong>
<p align="justify">
A computer scientist, whose research is on machine learning and AI, with an emphasis on deep learning.
In 2011, Ng founded the Google Brain project at Google, which developed very large scale artificial neural networks using Google's distributed computer infrastructure.
Among its notable results was a neural network trained using deep learning algorithms on 16,000 CPU cores, that learned to recognize higher-level concepts, such as cats, after watching only YouTube videos, and without ever having been told what a "cat" is.<br />
<a href="https://en.wikipedia.org/wiki/Andrew_Ng"> source </a>
</p>

## Teams
<strong><a href="https://research.google.com/teams/brain/"> Google brain </a></strong>
<p align="justify">
A deep learning artificial intelligence research project at Google.
It combines open-ended machine learning research with system engineering and Google-scale computing resources.<br />
<a href="https://en.wikipedia.org/wiki/Google_Brain"> source </a>
</p>

<strong><a href="https://deepmind.com/"> Deep mind </a></strong>
<p align="justify">
A British artificial intelligence company founded in September 2010 and acquired by Google in 2014.
The company has created a neural network that learns how to play video games in a fashion similar to that of humans, as well as a Neural Turing Machine, or a neural network that may be able to access an external memory like a conventional Turing machine, resulting in a computer that mimics the short-term memory of the human brain.
The company made headlines in 2016 after its AlphaGo program beat a human professional Go player for the first time.<br />
<a href="https://en.wikipedia.org/wiki/DeepMind"> source </a>
</p>

<strong><a href="https://research.fb.com/category/facebook-ai-research-fair/"> Facebook AI research (FAIR) </a></strong>
<p align="justify">
An artificial intelligence research group in Facebook, led by Yann Lecun.
One of its notable projects is DeepFace -- a deep learning facial recognition system that identifies human faces in digital images.
It employs a nine-layer neural net with over 120 million connection weights, and was trained on four million images uploaded by Facebook users.
The system is said to be 97% accurate, compared to 85% for the FBI's Next Generation Identification system.<br />
<a href="https://en.wikipedia.org/wiki/DeepFace"> source </a>
</p>

<strong><a href="https://www.cifar.ca/"> Canadian Institute for Advanced Research (CIFAR)</a></strong>
<p align="justify">
Founded in 1982, is an institute of advanced study that creates and maintains global research networks working on complex areas of inquiry.
It is supported by individuals, foundations and corporations, as well as funding from the Government of Canada and the Provinces of Quebec, Ontario, British Columbia and Alberta.
Among other topics, its researchers work on the topic of learning in machines and brains.<br />
<a href="https://en.wikipedia.org/wiki/Canadian_Institute_for_Advanced_Research"> source </a>
</p>

## Tasks
<strong>Multiclass image classification</strong>
<p align="justify">
The task of determining what object does an image contain from a pre-specified list of possibilities, called classes.
Systems that tackle this problem usually order the classes, from the one that is most likely to appear in the image to the one that is least likely.
</p>

<strong>Top-1 error</strong>
<p align="justify">
Measures the percentage of test examples on which the target label was the class assigned with the highest score.
</p>

<strong>Top-5 error</strong>
<p align="justify">
Measures the percentage of test examples on which the target label was one in the top five scores.
</p>

<strong>Machine translation</strong>
<p align="justify">
A sub-field of computational linguistics that investigates the use of software to translate text or speech from one language to another.<br />
<a href="https://en.wikipedia.org/wiki/Machine_translation"> source </a>
</p>

## Events
<strong>International Conference on Learning Representations (ICRL)</strong>
<p align="justify">
The conference takes a broad view of the field of deep learning and includes various related topics.
The applications include vision, speech recognition, text understanding, gaming, music, etc.<br />
<a href="http://www.iclr.cc/doku.php?id=ICLR2018:main&redirect=1"> source </a>
</p>

<strong>Neural Information Processing Systems (NIPS)</strong>
<p align="justify">
A machine learning and computational neuroscience conference.
Other fields represented at NIPS include cognitive science, psychology, computer vision, statistical linguistics, and information theory.<br />
<a href="https://en.wikipedia.org/wiki/Conference_on_Neural_Information_Processing_Systems"> source </a>
</p>

<strong>International Conference on Machine Learning (ICML)</strong>
<p align="justify">
The leading international academic conference in machine learning.
Along with NIPS, it is one of the two primary conferences of high impact in Machine Learning and Artificial Intelligence research.
</p>

<strong>Conference on Computer Vision and Pattern Recognition (CVPR)</strong>
<p align="justify">
An annual conference on computer vision and pattern recognition, by several measures regarded as the top conference in computer vision.<br />
<a href="https://en.wikipedia.org/wiki/Conference_on_Computer_Vision_and_Pattern_Recognition"> source </a>
</p>

<strong>International Conference on Computer Vision (ICCV)</strong>
<p align="justify">
Considered, together with CVPR, the top level conference in computer vision.<br />
<a href="https://en.wikipedia.org/wiki/International_Conference_on_Computer_Vision"> source </a>
</p>

<strong>European Conference on Computer Vision (ECCV)</strong>
<p align="justify">
Similar to ICCV in scope and quality, it is held those years which ICCV is not.
Like ICCV and CVPR, it is considered an important conference in computer vision.<br />
<a href="https://en.wikipedia.org/wiki/European_Conference_on_Computer_Vision"> source </a>
</p>

## Resources
<strong><a href="http://cs231n.github.io/"> CS231n </a></strong>
<p align="justify">
A course on convolutional neural networks for visual recognition in Stanford's computer science department.
</p>

<strong><a href="http://www.deeplearningbook.org/"> The deep learning book </a></strong>
<p align="justify">
A book on deep learning by Ian Goodfellow, Yoshua Bengio and Aaron Courville.
</p>

## Systems
<strong><a href="http://www.vlfeat.org/matconvnet/"> Matconvnet </a></strong>
<p align="justify">
MatConvNet is a MATLAB toolbox implementing Convolutional Neural Networks (CNNs) for computer vision applications.
It is simple, efficient, and can run and learn state-of-the-art CNNs.
Many pre-trained CNNs for image classification, segmentation, face recognition, and text detection are available.<br />
<a href="http://www.vlfeat.org/matconvnet/"> source </a>
</p>

<strong><a href="https://www.tensorflow.org/"> TensorFlow </a></strong>
<p align="justify">
An open-source software library for machine learning across a range of tasks.
It is a system for building and training neural networks to detect and decipher patterns and correlations, analogous to (but not the same as) human learning and reasoning.
It is used for both research and production at Google.<br />
<a href="https://en.wikipedia.org/wiki/TensorFlow"> source </a>
</p>

<strong><a href="http://pytorch.org/"> PyTorch </a></strong>
<p align="justify">
An open source machine learning library, a scientific computing framework, and a script language based on the Lua programming language.
It provides a wide range of algorithms for deep machine learning, and uses the scripting language LuaJIT, and an underlying C implementation.<br />
<a href="https://en.wikipedia.org/wiki/Torch_(machine_learning)"> source </a>
</p>

<strong><a href="https://keras.io/"> Keras </a></strong>
<p align="justify">
Keras is an open source neural network library written in Python.
It is capable of running on top of MXNet, Deeplearning4j, Tensorflow, CNTK or Theano.
Designed to enable fast experimentation with deep neural networks, it focuses on being minimal, modular and extensible.
Its primary author and maintainer is François Chollet, a Google engineer.<br />
<a href="https://en.wikipedia.org/wiki/Keras"> source </a>
</p>