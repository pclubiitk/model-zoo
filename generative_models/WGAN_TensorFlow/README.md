# TENSORFLOW IMPLEMENTATION OF WGAN

## Usage:
```bash
$ python3 train.py
```
>**_NOTE_** On Notebook use :
```python
!git clone link-to-repo
%run train.py
```

## Help Log:
```
usage: train.py [-h] [--dataset {mnist,cifar-10}]
                [--learning_rate LEARNING_RATE] [--n_clip N_CLIP]
                [--n_epoch N_EPOCH] [--n_update_dis N_UPDATE_DIS]
                [--noise_dim NOISE_DIM] [--batch_size BATCH_SIZE]
                [--samples_dir SAMPLES_DIR] [--save_dir SAVE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --dataset {mnist,cifar-10}
                        choice of dataset
  --learning_rate LEARNING_RATE
                        initial learning rate
  --n_clip N_CLIP       Cliping weight
  --n_epoch N_EPOCH     max # of epoch
  --n_update_dis N_UPDATE_DIS
                        # of updates of discriminator per update of generator
  --noise_dim NOISE_DIM
                        dimension of random noise
  --batch_size BATCH_SIZE
                        # of batch size
  --samples_dir SAMPLES_DIR
                        directory for sample output
  --save_dir SAVE_DIR   directory for checkpoint models
```

## Contributed by:
* [Akshay Gupta](https://github.com/akshay-gupta123)

## Refrence :
* **Title** : Wassertian GAN
* **Link** : https://arxiv.org/abs/1701.07875
* **Author** : Martin Arjovsky,Soumith Chintala and L´eon Bottou
* **Tags** : Neural Network,Genreative Adversirial Network
* **Published** : 6 Dec, 2017

# Summary:

## Introduction:

<strong>Generative adversarial network</strong> contains the two components: generator and discriminator. The training process is just like zero-sum game, and it can be simply shown in Figure below. 
<img src="./assets/gan.png"/>

For generator, it should generate the image which is just like the real one. On the contrary, the discriminator should distinguish the image is fake or not. During the training, the generator should make itself have more capability to generate image which is more and more like the actual one, and the discriminator should make itself realize the difference with more and more accuracy.

The problem this paper is concerned with is that of unsupervised learning.Authors direct their attention towards<em>various ways to measure how close the model distribution and the real distribution are</em>, or equvalently on the various ways to define a distance or divergence  ρ(P<sub>θ</sub>, P<sub>r</sub>) where the real data distribution P<sub>r</sub> admits a density and P<sub>θ</sub> is the distribution of the parametrized density. The most fundamental difference between such distances is their impact on the convergence of sequences of probability distributions. In order to optimize the parameter θ, it is of course desirable to define our model distribution P<sub>θ</sub> in a manner that makes the mapping θ→P<sub>θ</sub> is continuous.

## Different Distances:

Let X be a compact metric set (such as the space of images [0, 1]<sup>d</sup>) and let Σ denote the set of all the Borel subsets of X . Let Prob(X) denote the space of probability measures defined on X . We can now define elementary distances and divergences between two distributions P<sub>r</sub>, P<sub>g</sub> ∈ Prob(X ):

<img src="./assets/dis.png"/>

For the distance measure of probability distribution, there are a lot of metric can be the choice which are shown in Figure above. The most left one is total variation distance (TV-divergence); the second one is KL-divergence which has been well known in VAE; the third one is JS-divergence.
<br>The following figure illustrates how apparently simple sequences of probability distributions converge under the<strong>EM distance </strong>but do not converge under the other distances and divergences defined above.

<img src="./assets/example.png"/>

Example above gives us a case where we can learn a probability distribution over a low dimensional manifold by doing gradient descent on the EM distance. This cannot be done with the other distances and divergences because the resulting loss function is not even continuous.<br>
The Figure below illustrates this example. The green region is the data distribution of P<sub>0</sub>, and the orange region is the data distribution of P<sub>θ</sub>. In the general case, the two distribution are separated.

<img src="./assets/illustration.png"/>


## Wassertian GAN :

Neither KL-divergence nor JS-divergence can give the right direction to learn the capability, Martin et al. changed another metric — <strong>EM distance</strong> (or called Wasserstein-1 distance) . The physical idea of EM distance is:<em> how much work you should spend to transport the distribution to another one</em>. As the result, the value is positive and the shape is symmetric. There are two properties that the EM-distance has:<br>
<strong>*1. The function is continuous anywhere<br>
*2. The gradient of the function is almost everywhere</strong>

*( For proof refer to <a href="https://arxiv.org/abs/1701.07875">Paper</a>)

<img src="./assets/dual form.png"/>

However, During finding the infimum, it’s hard to exhaust the whole possible sample in the joint distribution. By Kantorovich-Rubinstein duality method, we can approximate the problem into the dual format, and just find the supremum. The relation between the two form is shown in Figure above. The only constraint is that the function should be the <em>Lipschitz-1 continuous function</em>.

<img src="./assets/object-wgan.png"/>

In the usual GAN, we want to maximize the score of classification. If the image is fake, the discriminator should give it as 0 score; if the image is real one, the 1 score should be gotten. In WGAN, it changes the task of discriminator as regression problem, and Martin renamed it as <strong>critics</strong>. The critics should measure the EM-distance that how many work should spend, and find the maximum case

<img src="./assets/algorithm-wgan.png"/>

The training process of WGAN is shown above which is very similar like usual GAN. There are only 4 difference:
<br>1. The critics will update for multiple times
<br>2. We don’t need to take logarithm (don’t use cross entropy) while computing the loss
<br>3. We should do weight clipping to satisfy the constraint of Lipschitz continuity
<br>4. Don’t use momentum-based optimizer (for example, Adam optimizer)

<img src="./assets/loss-dis.png"/>

After the experiment by Martin, the WGAN can avoid the problem of <em>gradient vanishment</em>. As you can see in the Figure , the gradient of usual GAN drops to zero and becomes saturate phenomenon. However, <em>EM-distance provides meaningful loss and the model can still learn gradually</em>.

## Result: 

I train model having architecture of DCGAN with default values as follows:
* Dataset: MNIST
* Learnini rate: 5e-4
* Clipping Wight: 1e-2
* Epochs: 60
* Udates of discriminator per update of generator(n_update_dis): 5
* Noise Dimension: 32
* Batch Size: 64

<strong><ins>Generated after 60 epochs on MNIST Dataset</ins></strong>

<img src="./assets/image_gen.png"/>

<strong><ins>Generator Loss</ins></strong>
<img src="./assets/gloss.png"/>

<strong><ins>Discriminator Loss</ins></strong>
<img src="./assets/dloss.png"/>

## Conclusion:

An algorithm deemed as WGAN is introduced, an alternative to traditional GAN training. In this new model,the stability of learning gets improved, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, the corresponding optimization problem seems to be sound, and provided extensive theoretical work highlighting the deep connections to other distances between distributions.
