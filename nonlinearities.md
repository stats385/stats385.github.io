---
layout: default
---

<strong>Sigmoid</strong>
<p align="justify">
The sigmoid, defined as

<a href="http://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-x}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-x}}" title="f(x) = \frac{1}{1 + e^{-x}}" /></a>

,
is a non-linear function that suffers from saturation.
</p>

<strong>Saturation of activation</strong>
<p align="justify">
An activation that has an almost zero gradient at certain regions.
This is an undesirable property since it results in slow learning.
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

[back](motifs)
