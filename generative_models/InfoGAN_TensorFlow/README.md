# TensorFlow Implementation of InfoGAN 
## Usage
```bash
$ python3 main.py --dataset CIFAR10 --noise_dim 64
```
> **_NOTE:_** on Colab Notebook use following command:
```python
!git clone link-to-repo
%run main.py --dataset CIFAR10 --noise_dim 64
```

## Help Log
```
usage: main.py [-h] [--dataset DATASET] [--epochs EPOCHS]
               [--noise_dim NOISE_DIM] [--continuous_weight CONTINUOUS_WEIGHT]
               [--batch_size BATCH_SIZE] [--outdir OUTDIR]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Name of dataset: MNIST (default) or CIFAR10
  --epochs EPOCHS       No of epochs: default 50 for MNIST, 150 for CIFAR10
  --noise_dim NOISE_DIM
                        No of latent Noise variables, default 62 for MNIST, 64
                        for CIFAR10
  --continuous_weight CONTINUOUS_WEIGHT
                        Weight given to continuous Latent codes in loss
                        calculation, default 0.5 for MNIST, 1 for CIFAR10
  --batch_size BATCH_SIZE
                        Batch size, default 256
  --outdir OUTDIR       Directory in which to store data, don't put '/' at the
                        end!
```

## Contributed by:
* [Atharv Singh Patlan](https://github.com/AthaSSiN)

## References

* **Title**: InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets
* **Authors**: Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel
* **Link**: https://arxiv.org/pdf/1606.03657.pdf
* **Tags**: Neural Network, Generative Networks, GANs
* **Year**: 2016

# Summary 

## Introduction

Generative adversarial nets were recently introduced as a novel way to train a generative model.
They consist of two ‘adversarial’ models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.

However, the above specified GAN, termed as VanillaGAN, is not good in classifying the inputs provided to it, and hence generate an image as per our specifications. In order to do this, we need to tune the noise provided in the input provided to the GAN, and hence define a way so that the GAN learns to classify an image as belong to a given class, and also determine if it is real or fake. 

Enter InfoGAN!

## InfoGAN

The idea is to provide a latent code, which has meaningful and consistent effects on the output. For instance, consider the MNIST dataset, where we have 10 digits. It would be helpful if we could use the property of the dataset having 10 classes and be able to assign a given digit with a particular value. This can be done by assigning part of the input to a 10-state discrete variable. The hope is that if you keep the code the same and randomly change the noise, you get variations of the same digit.

The way InfoGAN approaches this problem is by splitting the Generator input into two parts: the traditional noise vector and a new “latent code” vector. The codes are then made meaningful by maximizing the __Mutual Information__ between the code and the generator output.

![Eqn1](https://miro.medium.com/max/552/1*rSZXfx4_xcC-5z4LirNDRQ.png)

Here *V(D,G)* is the standard Vanilla Gan loss, and *I(c;G(z,c))* is the mutual information loss, with Lambda being sort of a regularization constant (the mutual information loss can be seen as a regularizing term

However, int the calculation of *I(c;G(z,c))*, we need to sample from the posterior distribution of the latent codes, which is usually intractable, and hence we replace it with a lower bound, calculated by approximating the posterior using an auxiliary distribution *Q(c|x)* and the reparameterization trick.

![Eqn2](https://miro.medium.com/max/552/1*NTYmbgNBT9RzhdLl71-koA.png)  

Where  
![Eqn3](https://miro.medium.com/max/552/1*92L-ml_k7iQcPIWcvT7TIw.png)  

Hence the final form of the loss function becomes:  
![Eqn4](https://miro.medium.com/max/552/1*W2G0DFBQUa52Piy1snYVjQ.png)

Thus, the problem basically reduces to the following process:
1. Sample a value for the latent code c from a prior of your choice
2. Sample a value for the noise z from a prior of your choice
3. Generate x = *G(c,z)*
4. Calculate *Q(c|x=G(c,z))*

## Implementation

In the implementation, we input a user defined number of noise variables, 10 categorical latent codes (hoping that in the output, each corresponds to a class of the datasets), and 2 uniform continuous latent codes (with values from -1 to 1), hoping that the correspond to some other features in the dataset

![Model](https://miro.medium.com/max/1104/1*dXLgTV8lNiTInvxomgZSAg.png)

We use the following default configuration: 
- Binary CE to calculate the loss in real and fake samples detection
- Categorical CE to calculate the loss in categorical classification
- Ordinary Least Squares to calculate the loss in continuous variable detection (The continuous variables are uniform in the input but in the architecture predicts them in the form of a Gaussian Distribution. So i tried outputting the mean and log variance of the predictions and hence calculating the losses using the reparameterization trick, but upon applying some basic mathematics, I realized that it all boils down to calculating the OLS of the predicted values)
- Lambda = 1, however, the weight given to the loss of the continuous codes can be varied (we used 0.5 for MNIST and 1 for CIFAR10)

# Results

## On MNIST Dataset

Results after training for 50 epochs:
![mnistFinal](https://github.com/AthaSSiN/model-zoo/blob/master/generative_models/InfoGAN_TensorFlow/assets/ReadmeImages/mnistfinal.png)

> **_NOTE:_** In this graph orange plot corresponds to dicriminator loss, blue to generator loss, Green to loss of continuous variables and Gray to loss in categorical variables.


Loss:  
![mnistloss](https://github.com/AthaSSiN/model-zoo/blob/master/generative_models/InfoGAN_TensorFlow/assets/ReadmeImages/mnistloss.png)

Plot of Real and Fake detection accuracies:  
![mnistreal](https://github.com/AthaSSiN/model-zoo/blob/master/generative_models/InfoGAN_TensorFlow/assets/ReadmeImages/mnistrealaccuracy.png)
![mnistfake](https://github.com/AthaSSiN/model-zoo/blob/master/generative_models/InfoGAN_TensorFlow/assets/ReadmeImages/mnistfakeaccuracy.png)

Here is the final image generated by the generator for a randomly generated noise and label, with one continuous code being varied along the rows.

In this one, the tilt in the images seems to change as we move left to right:  
![mnisttilt](https://github.com/AthaSSiN/model-zoo/blob/master/generative_models/InfoGAN_TensorFlow/assets/ReadmeImages/mnisttilt.png)

While in this, the thickness of the digits seems to change:  
![mnistthick](https://github.com/AthaSSiN/model-zoo/blob/master/generative_models/InfoGAN_TensorFlow/assets/ReadmeImages/mnistthick.png)

Note: In some cases, the digits have also changed while varying the continuous codes. I think that this is because there are many possible characters that the uniform codes can comply to, and its actually quite possible that they do not apply only to thickness / tilt etc, but can apply to curviness, or number of lines in a digit etc, which can make digits which look similar to each other, be generated by the same categorical code.

## On CIFAR10 Dataset

> **_NOTE_**: The paper does not have an implementation for the CIFAR10 dataset and hence the results aren't very good.

Results after training for 137 epochs

![cifargif](https://github.com/AthaSSiN/model-zoo/blob/master/generative_models/InfoGAN_TensorFlow/assets/ReadmeImages/CIFARfinal.png)

> **_NOTE:_** In this graph blue plot corresponds to generator loss and orange to discriminator loss

Here is the loss graph  
![cifarloss](https://github.com/AthaSSiN/model-zoo/blob/master/generative_models/InfoGAN_TensorFlow/assets/ReadmeImages/CIFARloss.png)

Plot of Real and Fake detection accuracies:  
![CIFARreal](https://github.com/AthaSSiN/model-zoo/blob/master/generative_models/InfoGAN_TensorFlow/assets/ReadmeImages/CIFARrealaccuracy.png)
![CIFARfake](https://github.com/AthaSSiN/model-zoo/blob/master/generative_models/InfoGAN_TensorFlow/assets/ReadmeImages/CIFARfakeaccuracy.png)

Here is the final image generated by the generator for a randomly generated noise and label.

In this one, the background color varies as we move left to right:  
![cifarbg](https://github.com/AthaSSiN/model-zoo/blob/master/generative_models/InfoGAN_TensorFlow/assets/ReadmeImages/CIFARbackground.png)

While in this, the foreground color/size varies:  
![cifarfg](https://github.com/AthaSSiN/model-zoo/blob/master/generative_models/InfoGAN_TensorFlow/assets/ReadmeImages/CIFARforeground.png)

It seems the continuous latent codes are working fine, but the categorical codes weren't able to represent the different classes too well, hence there is room for a lot of experiments!

# Sources

- [InfoGAN — Generative Adversarial Networks Part III](https://towardsdatascience.com/infogan-generative-adversarial-networks-part-iii-380c0c6712cd)  
Template on which the code was built:  
- [DCGAN on TensorFlow tutorials](https://www.tensorflow.org/tutorials/generative/dcgan)

