---
layout: default
---

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
