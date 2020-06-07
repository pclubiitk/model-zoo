# TensorFlow Implementation of BERT

## Dependencies
[Transformer implementation by HuggingFace](https://huggingface.co/transformers/)  
This has been used for loading the WordPiece tokens for the pretraining task, and for providing the pre-trained models in the finetuning task.  
```bash
$ pip install transformers
```
> **_NOTE:_** on Colab Notebook use following command:
```python
!pip install transformers
```
## Usage
### 1. Pretraining (on user defined corpus)
```bash
$ python3 pretrain.py --train_corpus path/to/file.txt
```
> **_NOTE:_** on Colab Notebook use following command:
```python
!git clone link-to-repo
%run pretrain.py --train_corpus path/to/file.txt
```
#### Help Log
```
usage: pretrain.py [-h] [--num_layers NUM_LAYERS] [--epochs EPOCHS]
                   [--hidden_size HIDDEN_SIZE] [--num_heads NUM_HEADS]
                   [--max_length MAX_LENGTH] [--batch_size BATCH_SIZE]
                   --train_corpus TRAIN_CORPUS

optional arguments:
  -h, --help            show this help message and exit
  --num_layers NUM_LAYERS
                        Number of Encoder layers, default 12
  --epochs EPOCHS       Number of epochs in pretrain, default 40
  --hidden_size HIDDEN_SIZE
                        Number of neurons in hidden feed forward layer,
                        default 512
  --num_heads NUM_HEADS
                        Number of heads used in multi headed attention layer,
                        default 12
  --max_length MAX_LENGTH
                        Maximum token count of input sentence, default 512 
                        (Note: if number of token exceeds max length, an error 
                        will be thrown)
  --batch_size BATCH_SIZE
                        Batch size, default 2 (WARN! using batch size > 2 on
                        just one GPU can cause OOM)
  --train_corpus TRAIN_CORPUS
                        Path to training corpus, required argument.
```

#### Datasets
To replicate the no longer publicly available Toronto BookCorpus Dataset follow the instructions in [this github repository](https://github.com/sgraaf/Replicate-Toronto-BookCorpus)

This relatively small [BookCorpus](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html) can also be downloaded directly as an alternative to the above dataset.

To prepare the corpus from Wikipedia articles (on which BERT was originally trained) follow [this link](https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html)

### 2. Finetuning (On IMDB dataset)
```bash
$ python3 finetune.py
```
> **_NOTE:_** on Colab Notebook use following command:
```python
!git clone link-to-repo
%run finetune.py
```
#### Help Log
```
usage: finetune.py [-h] [--epochs EPOCHS] [--lr LR] [--batch_size BATCH_SIZE]
                   [--max_length MAX_LENGTH] [--train_samples TRAIN_SAMPLES]
                   [--test_samples TEST_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs in finetuning, default 2
  --lr LR               Learning rate for finetune, default 2e-5
  --batch_size BATCH_SIZE
                        Batch Size, default 32 (WARN! using batch size > 32 on
                        just one GPU can cause OOM)
  --max_length MAX_LENGTH
                        Maximum length of input string to bert, default 128
  --train_samples TRAIN_SAMPLES
                        Number of training samples, default (max): 25000
  --test_samples TEST_SAMPLES
                        Number of test samples, default (max): 25000
```

## Contributed by:
* [Atharv Singh Patlan](https://github.com/AthaSSiN)

## References

* **Title**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
* **Authors**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
* **Link**: https://arxiv.org/pdf/1810.04805.pdf
* **Tags**: Neural Network, Natural Language Processing
* **Year**: 2018

# Summary 

## Introduction

Vaswani et. al. introduced the concept of transformers in their seminal paper "Attention is all you need", which shook the NLP community, and marked a sharp evolution in the field of NLP (It’s been referred to as NLP’s ImageNet moment, referencing how years ago similar developments accelerated the development of machine learning in Computer Vision tasks)

This evolution was further accelerated due to the development of BERT, which is actually not a new model, but a training strategy for transformer encoders. 

BERT is a clever combination of up and coming NLP ideas in 2018, which in the right blend, produced very impressive results!

![BERTTL](http://jalammar.github.io/images/bert-transfer-learning.png)

## BERT

Traditional context-free models (like word2vec or GloVe) generate a single word embedding representation for each word in the vocabulary which means the word “right” would have the same context-free representation in “I’m sure I’m right” and “Take a right turn.” However, BERT would represent based on both previous and next context making it bidirectional. While the concept of bidirectional was around for a long time, BERT was first on its kind to successfully pre-train bidirectional in a deep neural network.

![Embsum](https://miro.medium.com/max/552/1*8416XWqbuR2SDgCY61gFHw.png)

Models like Transformer and Open-AI GPT did not use the idea of bidirectionality simply because is they did use bidirectionality, is that the network were being trained on the problem of next word prediction. Hence, if bidirectionality was used, the model would eventually learn that the next word in the input is actually the output, and the task would become trivial, which is not desired.

However, BERT uses Mask Language Model (MLM) — by Masking out some of the words in the input and then condition each word bidirectionally to predict the masked words. Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a [MASK] token(actually, 80% of that 15%, while 10% are replaced with a random token and remaining are kept the same). The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence. This can not be made trivial by masking!

![MLM](https://miro.medium.com/max/552/1*icb8KIyD7MGKVKf39-TO1A.png)  

The second technique is the Next Sentence Prediction (NSP), where BERT learns to model relationships between sentences. In the training process, the model receives pairs of sentences as input and learns to predict if the second sentence in the pair is the subsequent sentence in the original document. Let’s consider two sentences A and B, is B the actual next sentence that comes after A in the corpus, or just a random sentence? For example:
![NSP](https://miro.medium.com/max/552/1*K1em9OWRbZsA8f3IisUCig.png)  

When training the BERT model, both the techniques are trained together, thus minimizing the combined loss function of the two strategies.

## Implementation

The BERT architecture builds on top of Transformer. There are two variants available:
- BERT Base: 12 layers (transformer blocks), 12 attention heads, and 110 million parameters
- BERT Large: 24 layers (transformer blocks), 16 attention heads and, 340 million parameters

![Model](https://miro.medium.com/max/552/1*IOskqRtq3UOjvchtFxe-AA.png )

By default, our code implements the Bert Base model on both pretrain and finetune problems.

We use the following default configuration: 
- Binary CE to calculate the correctness of next sentence prediction problem
- Categorical CE to calculate the loss in masked word prediction
- Learning rate scheduling, such that the learning rate increases linearly for the first 10000 minibatches to let the model warm up, and subsequently reduces inversely proportional to the current iteration value

# Results

> **_NOTE:_** BERT is a very large model, hence training for too many epochs on a small dataset like IMDB or COLa causes overfitting, hence it is best to finetune bert on them for 2-4 epochs only

The results after training for 2 epochs on the IMDB dataset were:  
1. Plot of losses:  
![loss](https://github.com/AthaSSiN/model-zoo/blob/master/NLP/BERT_TensorFlow/assets/loss.png)

2. Plot of accuracy:  
![acc](https://github.com/AthaSSiN/model-zoo/blob/master/NLP/BERT_TensorFlow/assets/acc.png)

# Sources

- [Understanding BERT: Is it a Game Changer in NLP?](https://towardsdatascience.com/understanding-bert-is-it-a-game-changer-in-nlp-7cca943cf3ad)  
- Template on which the code was built:  [Transformer on TensorFlow tutorials](https://www.tensorflow.org/tutorials/text/transformer)

