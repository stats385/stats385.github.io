---
layout: default
---

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

[back](cheat_sheet)
