# VAEGAN_tensorflow


# Paper

* **Title**: Autoencoding beyond pixels using a learned similarity metric.

* **Authors**: Anders Boesen Lindbo Larsen,Søren Kaae Sønderby,Hugo Larochelle,Ole Winther.

* **Link**: https://arxiv.org/pdf/1512.09300.pdf.

* **Tags**: Neural Network, Generative Networks, GANs.

* **Year**: 2016.

# Contributed by
 [Umang Pandey](https://github.com/Umang-pandey)
# Summary

a variational autoencoder (VAE) is combined with a Generative Adversarial Network (GAN) in order to learn a higher level image similarity metric instead of the traditional element-wise metric.

## VAE
VAEs store latent attributes as probability distributions.Typical setup of a variational autoencoder is nothing other than a cleverly designed deep neural network, which consists of a pair of networks: the encoder and the decoder. The encoder can be better described as a variational inference network, which is responsible for the mapping of input x​​​ to posteriors distributions q​θ​​(z∣x). Likelihood p(x∣z) is then parametrized by the decoder, a generative network which takes latent variables z and parameters as inputs and projects them to data distributions p​ϕ​​(x∣z).A major drawback of VAEs is the blurry outputs that they generate. 

![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/VAE.png)

The loss in VAE is defined as:

![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/vaeloss.png)

## GANs
Just like VAEs, GANs belong to a class of generative algorithms that are used in unsupervised machine learning. Typical GANs consist of two neural networks, a generative neural network and a discriminative neural network. A generative neural network is responsible for taking noise as input and generating samples. The discriminative neural network is then asked to evaluate and distinguish the generated samples from training data. Much like VAEs, generative networks map latent variables and parameters to data distributions.

The major goal of generators is to generate data that increasingly “fools” the discriminative neural network, i.e.
increasing its error rate. This can be done by repeatedly generating samples that appear to be from the training data distribution.The loss in GANs is defined as:

![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/ganloss.png)
## VAEGAN
The term VAE-GAN is first introduced in the paper “Autoencoding beyond pixels using a learned similarity metric” by A. Larsen et. al. The authors suggested the combination of variational autoencoders and generative adversarial networks outperforms traditional VAEs.

![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/model.png)


The encoder encodes the data sample **_x_** into a latent representation **_z_** while the decoder tries to reconstruct **_x_** from the latent vector. This reconstruction is fed to the discriminator of the GAN in order to learn the higher-level sample similarity.The loss in VAEGAN is defined as:
.

![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/trueloss.png)

The VAE and GAN is trained simultaneously using the loss function  which is composed of the prior regularization term from the encoder, the reconstruction error, and the style error from the GAN. However, this combined loss function is not applied on the entire network. A  the algorithm used in training the VAE/GAN network is shown below.


![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/algorithm.png)

# Architecture
 
 ## Encoder
  ![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/generator.png)
 ## Discriminator
  ![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/dicriminator.png)
 ## Decoder
  ![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/decoder.png)
# Results

## Fashion MNIST 
### After 500 Epochs

![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/fashion_mgif.gif)


## MNIST
### After 500 Epochs
![](https://github.com/Umang-pandey/vae-gan-tf/blob/master/images/mnist_gif.gif)


