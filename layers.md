---
layout: default
---

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

[back](motifs)
