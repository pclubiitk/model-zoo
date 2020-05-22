# dcgan-in-pytorch

## dcgan implementation in pytorch on MNIST 

original paper :  [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

MNIST dataset: http://yann.lecun.com/exdb/mnist/

## Introduction
Generative Adversarial Networks (GANs) are one of the most popular (and coolest)
Machine Learning algorithms developed in recent times. They belong to a set of algorithms called generative models, which
are widely used for unupervised learning tasks which aim to learn the uderlying structure of the given data. As the name
suggests GANs allow you to generate new unseen data that mimic the actual given real data. However, GANs pose problems in
training and require carefullly tuned hyperparameters.This paper aims to solve this problem.

DCGAN is one of the most popular and succesful network design for GAN. It mainly composes of convolution layers 
without max pooling or fully connected layers. It uses strided convolutions and transposed convolutions 
for the downsampling and the upsampling respectively.

![architecture](images/architecture.png)

## Network Design of DCGAN:
* Replace all pooling layers with strided convolutions.
* Remove all fully connected layers.
* Use transposed convolutions for upsampling. 
* Use Batch Normalization after every layer except after the output layer of the generator and the input layer of the discriminator.
* Use ReLU non-linearity for each layer in the generator except for output layer use tanh.
* Use Leaky-ReLU non-linearity for each layer of the disciminator excpet for output layer use sigmoid.

## Hyperparameters for this Implementation
Hyperparameters are chosen as given in the paper.
* mini-batch size: 128
* learning rate: 0.0002
* momentum term beta1: 0.5
* slope of leak of LeakyReLU: 0.2
* For the optimizer Adam (with beta2 = 0.999) has been used instead of SGD as described in the paper.

* MNIST vs Generated images

<table align='center'>
<tr align='center'>
<td> MNIST </td>
<td> DCGAN after 10 epochs </td>
</tr>
<tr>
<td><img src = 'images/raw_MNIST.png'>
<td><img src = 'images/MNIST_DCGAN_10.png'>
</tr>
</table>

* Training loss

![Loss](images/loss.png)

## Contributed by:
* [Nakul Jindal](https://github.com/nakul-jindal)
