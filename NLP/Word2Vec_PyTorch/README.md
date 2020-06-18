# Tensorflow Implementation of Word2Vec (Dataset from kaggle file)

## Usage
### To train
```bash
$ python3 main.py --epochs 100 --optimizer "adam" --batch_size 2000 --dim_embedding 100
```
### To getSimilarity between two words - word1 and word2
```bash 
$ python3 main.py --mode "getSimilarity" --word1 "window" --word2 "house"
```
### To getTenClosestWords to a given word
```bash
$ python3 main.py --mode "getTenClosestWords" --word "window"
```
### To use analogy and get word in ;- word1_ : word2_ :: word3_ : word4
```bash
$ python3 main.py --mode "analogy" --word1_ "window" --word2_ "house" --word3_ "door"
```
### To plot the embeddings in 2D
```bash
$ python3 main.py --mode "plot"
```
## References
* [Stanford CS224n](http://web.stanford.edu/class/cs224n/)
* [Stanford Word2Vec Notes](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)

* [Original Word2Vec paper](http://arxiv.org/pdf/1301.3781.pdf)

* [Google Word2vec paper which suggested improvements in training using negative sampling and sub-sampling](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

## Contributed by:
* [Aashish Patel](https://github.com/aashishpiitk/)

# Summary

Word2Vec is a model in which a network is trained to represent each word in the text corpus in form of an embedding, which is a vector containing numbers.
These embeddings can be used to perform a variety of tasks such a :-
```
    • finding similarity between two words
    • searching for top ten most similar words to a given words
    • finding analogies
```

There are two main approaches while training this model and creating word embeddings:-
```
    • Skip-gram 
    • Continuous Bag of Words 
```

### Skip-gram
```
Input – a single word
Output – probability of each word in corpus being in context of the given input word
```
### Continuous Bag Of Words(CBOW)
```
Inputs – context of a word in sentence/phrase
Output – a single word(in one-hot encoded form, each value being a number from probability distribution)
Loss – categorical cross entropy loss
```
In my current implementation I have used CBOW approach to train

### Preparing data for CBOW
```
    1. converting each vector in vocabulary to one-hot encoded representation
    2. forming a list of (context word, target word) , choosing a suitable window size
```
## Architecture of CBOW
```
1. This is a three layer neural network with the last one being the output layer
2. The weights of the first layer are the actual embeddings which will be used in further tasks
3. The output is of size of vocab with the entries being the softmax output
4. Cross Entropy loss is used 
```
## Instructions for using a custom dataset to train the model
```
1. In the  `dataloader.py` file on the `30th` line change the name of the file and path(file must be csv)
2. On the `31st` line change the name of the column to the name of the `column` in your `.csv` file
3. On the next line choose the number of example sentences to choose from the `.csv` file.This option is useful when there is not enough RAM on your machine to load all the lines
```
## Instructions for using kaggle dataset
```
1. !kaggle datasets download -d harmanpreet93/hotelreviews
2. unzip the dataset and keep it in a folder named hotelreviews 
3. if you want to change the folder name then follow the guidelines above this block
```
## Examples
```
getSimilarity("window","door")
result 0.067170754
getSimilarity("window","house")
result 0.029237064
getSimilarity("vegas","girls")
result 0.10303633
getSimilarity("vegas","money")
result 0.22041301
getSimilarity("vegas","gold")
result 0.072522774
print(getSimilarity("good","bad"))
0.23856965

print(getTenClosestWords("water"))
result [['guess', 0.30290997], ['disappointed', 0.29180372], ['understand', 0.29148042], ['also', 0.27842966], ['earth', 0.2709463], ['water', 0.26725885], ['one', 0.2629741], ['power', 0.25627777], ['unbelievably', 0.25437462], ['spouse', 0.2514236]]

print(getTenClosestWords("money"))
result [['need', 0.3438718], ['chose', 0.34114993], ['money', 0.332256], ['nearby', 0.3176784], ['think', 0.31027701], ['heading', 0.3087694], ['although', 0.30681318], ['understands', 0.3010662], ['lodging', 0.29836887], ['must', 0.29601774]]


```