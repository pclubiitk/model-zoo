# Pytorch Implementation of VanillaGAN (MNIST Dataset)

### Usage
```bash
$ python3 main.py --k_steps 5 --epoch 50 
```
NOTE: on Colab Notebook use following command:
```bash
!git clone link-to-repo
%run main.py --k_steps 5 --epoch 50 
```
### References :
* Goodfellow et al. Generative adversarial networks.2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

### Contributed by:
* [Pramodh Gopalan](https://github.com/pramodh-1612/)

# Summary

## Introduction

>GANs are the most most interesting ideas in the last 10 years of machine learning.
>-Yann Lecun

GANs or generative adversarial networks first proposed in this paper,and was very well received by the reasearch community.It has over thousands of citations,and
moroever,the authors of this paper have also written a book on deep learning that has a lot of good reviews on it too.

Most generative algorithms,such as Naive Bayes (*Yes,it has some role to play here*) or Variational auto encoders(*VAEs*) try to learn the distribution from which the data was produced,so that it can
sample and generate images on it's own.But,the main drawback with them was the fact that it involved
optimizing a lower bound(*Not getting into the specifics*),which didn't yield the best results.GANs,on the other hand involve no such assumption.In fact,they use game theory concepts in order to acheive results better then the VAE's themseves.

## A few keywords before we get started: 

### Minimax-2-player-game :(definitions taken from [here](https://brilliant.org/wiki/minimax/)) 

In game theory, minimax is a decision rule used to minimize the worst-case potential loss; in other words, a player considers all of the best opponent responses to his strategies, and selects the strategy such that the opponent's best strategy gives a payoff as large as possible.

The name "minimax" comes from minimizing the loss involved when the opponent selects the strategy that gives maximum loss, and is useful in analyzing the first player's decisions both when the players move sequentially and when the players move simultaneously. As an historic example,consider [this](https://cs.stanford.edu/people/eroberts/courses/soco/projects/1998-99/game-theory/Minimax.html) :

>In 1943, the Allied forces received reports that a Japanese convoy would be heading by sea to reinforce their troops. The convoy could take on of two routes -- the Northern or the Southern route. The Allies had to decide where to disperse their reconnaissance aircraft -- in the north or the south -- in order to spot the convoy as early as possible. The following payoff matrix shows the possible decisions made by the Japanese and the Allies, with the outcomes expressed in the number of days of bombing the Allies could achieve with each possibility:

|                    | Japanese-ship-north | Japanese-ship-south |
|--------------------|:-------------------:|:-------------------:|
| Allies-recon-north |          2          |          2          |
| Allies recon-south |          1          |          3          |

>By this matrix, if the Japanese chose to take the southern route while the Allies decided to focus their recon planes in the north, the convoy would be bombed for 2 days. The best outcome for the Allies would be if they placed their airplanes in the south and the Japanese took the southern route. The best outcome for the Japanese would be to take the northern route, provided the Allies were looking for them in the south.

>To minimize the worst possible outcome, the Allies would have to choose the north as the focus of their reconnaisance efforts. This ensures them 2 days of bombing, whereas they risk only 1 day of bombing if they focus on the south. Therefore, by minimax, the best strategy for them would be to focus on the north.

>The Japanese can use the same strategy. The worst possible outcome for them is the 3 days of bombing which might occur if they took the southern route. Therefore, the Japanese would take the northern route.

>It is, in fact, what had occurred: both the Allies and the Japanese chose the north, and the Japanese convoy was bombed for 2 days.

### Nash equilibrium : (Defintions taken from [here](https://brilliant.org/wiki/nash-equilibrium/))

 Nash Equilibrium is a set of strategies that players act out, with the property that no player benefits from changing their strategy. Intuitively, this means that if any given player were told the strategies of all their opponents, they still would choose to retain their original strategy.

 Sometimes a nash equilibrium might occur when the two opponents are making
 simultaneous momvements.For an example,take : 

 >The simplest example of Nash equilibrium is the coordination game, in which both players benefit from coordinating but may also hold individual preferences. For instance, suppose two friends wish to plan an evening around either partying or watching a movie. Both friends prefer to engage in the same activity, but one prefers partying to movies by a factor of 2, and the other prefers movies to partying by the same ratio. This can be modeled by the following payoff matrix:

 |       | party | movie |
|-------|:-----:|:-----:|
| party |  2,1  |  0,0  |
| movie |  0,0  |  1,2  |

>where the payoff vector is listed under the appropriate strategy profile (the first player's strategies are listed on the left). In this case, both {**Party, Party**} and {**Movie, Movie**} are Nash equilibria, as neither side would choose to deviate when informed of the other's choice.

## Intuition

The intuition behind the GAN is simple.We have 2 neural networks,
a Generator and a Discriminator.A Generator takes in a random(*it's really random*) input and 
generates an image.A Discriminator takes in an input and tries to decide if
the data came from the generator,or if it is from the dataset itself. It is easy to see that this is a minimax game between 2 opponents (The generator G
and discriminator D),where both of them try and minimze the worst possible scenario.The training ends if both these NN's come to an impasse(*A Nash Equilibrium*).The paper provides a great analogy : 

>The generative model can be thought of as analogous to a team of counterfeiters,
trying to produce fake currency and use it without detection, while the discriminative model is
analogous to the police, trying to detect the counterfeit currency. Competition in this game drives
both teams to improve their methods until the counterfeits are indistiguishable from the genuine
articles.

## Math:

Like other generative models,this does have some probability behind it.
lets take a look at this figure : 

![GAN](https://www.researchgate.net/profile/Bajibabu_Bollepalli/publication/317388182/figure/fig1/AS:651858561486877@1532426600960/General-block-diagram-of-generative-adversarial-networks-GANs.png)

We define a prior on input noise P(z) (aka use this for sampling noise).The generator then maps this to a data distribution using a complicated(*I meant complicated for real,unlike Siraj*) differentiable function.The discriminator takes a data distribution and tries to find out the probability of the data distribution coming from the dataset as opposed to coming from the generator 

And as a whole,we would like to minimize this function : (Mathematically momdelling it)

![image](https://datascience.foundation/backend/web/uploads/blog/GAN%20Algorithm.png)

lets try and find some intuition behind it :

- when G is fixed,what happens to the equation?

 - The first half (*E[log D(x) | x ~ p(x)]*) suggests that you make all D(x) as close to 1 as       possible, correctly discriminating samples from plausible data.
 - The second half (*E[log(1-D(G(z))) | z ~ p(z)]*) suggests that you make D(G(z)) as close to 0 as  possible, so that it correctly discriminates samples that aren't plausible data.
- And when you have D fixed, and you need to solve min{G} V(G,D)? 

  - Now it looks like you want to make sure your G generates samples that are misclassified by D. You're looking for a network that generates, for a given D, samples that 'fool' it into misclassifying and marking D(G(z)) closer to 1 (&instead of 0, like in the first part*) .

We try and maximize the function wrt the parameters of the generator and the discriminator.


## Implementation/Architecture:

The paper suggests that we use simple MLP architecture,with activations of sigmoids at some layers
and Relu(*for the generator*) and Maxout activations(*for the generators*).We also need to train 
the discriminator for 'k' steps,and then train the generator for 1 step.

### In our implementation:

- We use the BCE loss (Binary cross entropy.For more on these confusing names,check [this](https://gombru.github.io/2018/05/23/cross_entropy_loss/) out!)
- Train both the models for 200 epochs.(*this happened in 40 minutes*)(*generative models are fast!*)
- we train the generator and the discriminator for one step each,test the generator on some test_noise,and compile that input into a image,and stack these images to form a
gif!

## Result :

Here is the loss graph :

![loss](./assets/losses.png)

Here is the gif !

![results](./assets/progress.gif)

## Scope for improvement:

I think this is pretty much the limit of what a MLP can do.With that said,maybe tweaking 'k' could lead to better quality.