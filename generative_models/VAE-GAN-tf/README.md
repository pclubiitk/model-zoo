# vae-gan-tf


# Paper

* Title: Autoencoding beyond pixels using a learned similarity metric.

* Authors: Anders Boesen Lindbo Larsen,Søren Kaae Sønderby,Hugo Larochelle,Ole Winther.

* Link: https://arxiv.org/pdf/1512.09300.pdf.

* Tags: Neural Network, Generative Networks, GANs.

* Year: 2016.


# Summary

a variational autoencoder (VAE) is combined with a Generative Adversarial Network (GAN) in order to learn a higher level image similarity metric instead of the traditional element-wise metric. 

![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/model.png)

The encoder encodes the data sample **_x_** into a latent representation **_z_** while the decoder tries to reconstruct **_x_** from the latent vector. This reconstruction is fed to the discriminator of the GAN in order to learn the higher-level sample similarity.
![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/ganloss.png)


![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/vaeloss.png)

![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/loss_prior.png)

![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/trueloss.png)

The VAE and GAN is trained simultaneously using the loss function  which is composed of the prior regularization term from the encoder, the reconstruction error, and the style error from the GAN. However, this combined loss function is not applied on the entire network. A  the algorithm used in training the VAE/GAN network is shown below.

![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/algorithm.png)



