# GloVe

## Usage
### Train
```bash
$ python3 main.py  --embedding_size 100 --context_size 10
```
### Check whether words are_Similar
```bash 
$ python3 main.py --mode "are_Similar" --word1 "big" --word2 "long"
```
### To get_ClosestWords to a given word
```bash
$ python3 main.py --mode "get_ClosestWords" --word "big"
```
### To use analogy and get word in :- word1 : word2 :: word : ?
```bash
$ python3 main.py --mode "analogy" --word1 "big" --word2 "long" --word "small"
```
### To plot the embeddings
```bash
$ python3 main.py --mode "plotEmbeddings"
```
## References
* [Stanford CS224n](http://web.stanford.edu/class/cs224n/)
* [Stanford GloVe Notes](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes02-wordvecs2.pdf)

* [Original GloVe paper](https://nlp.stanford.edu/pubs/glove.pdf)

## Contributed by:
* [Vansh Bansal](https://github.com/vanshbansal1505/)

# Summary

##Introduction:
GloVe is a speciﬁc weighted least squares model that trains on global word-word co-occurrence counts and thus makes efﬁcient use of statistics. The model produces a word vector space with meaningful substructure, as evidenced by its state-of-the-art performance of 75% accuracy on the word analogy dataset. Global Vectors (GloVe) the global corpus statistics are captured efficiently by the model.

## Model:
The relationship of these words can be examined by studying the ratio of their co-occurrence probabilities with various probe words, k. Compared to the raw probabilities, the ratio is better able to distinguish relevant words from irrelevant words and it is also better able to discriminate between the two relevant words.

<img src="https://img.sciencewal.com/img/machine-learning/emnlp-what-is-glove-partiii.png" alt="drawing" width="300"/>

We denote word vectors as w and separate context word vectors as ˜ w .
We require that F be a homomorphism, such that it establishes a relation betwwen word vectors and the co occurance counts.
It is given by

<img src="https://img.sciencewal.com/img/machine-learning/emnlp-what-is-glove-partiii-4.png" alt="drawing" width="400"/>

where word-word co-occurrence counts are denoted by X, whose entries Xij tabulate the number of times word j occurs in the context of word i.
The solution to these equations is exponential function. So, the equation, after adding the biases, becomes
 
 ![alt text](https://miro.medium.com/max/1400/1*AvJeMckcvOhJX0IGVVcZbg.jpeg)

We use a new weighted least squares regression model that addresses these problems. Casting above equation as a least squares problem and introducing a weighting function f(Xij) into the cost function gives us the model,

<img src="https://miro.medium.com/max/1224/1*oDcCpHSK7-Lt06zoW4NPMA.png" alt="drawing" width="500"/>
 
Where V is the size of the vocabulary.
The proposed weighting function is

 ![alt text](https://miro.medium.com/max/656/0*AZsJlwIghrdhD7c4)

we ﬁx to xmax = 100 for all our experiments. We found that α = 3/4 gives a modest improvement over a linear version with α = 1.

## A Sample of the Embedding Plot:

<img src="https://raw.githubusercontent.com/vanshbansal1505/GloVe_tf/master/embedding_plot_sample.png" alt="drawing">

## Comparison with word2vec:
GloVe controls for the main sources of variation by setting the vector length, context window size, corpus, and vocabulary size to the conﬁguration and most importantly training time.
![alt text](https://adriancolyer.files.wordpress.com/2016/04/glove-vs-word2vec.png?w=656&zoom=2)

## Conclusion:
GloVe is a new global log-bilinear regression model for the unsupervised learning of word representations that outperforms other models on word analogy, word similarity, and named entity recognition tasks.

### Note:
The dataset used in the implementation is same as used in word2vec tensorflow implementation to compare the results.

