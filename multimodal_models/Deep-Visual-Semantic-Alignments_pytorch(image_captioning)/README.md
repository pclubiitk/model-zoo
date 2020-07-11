# PyTorch implementation of Image Captioning [*Paper*](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
## Requirements
* python
* pytorch
* pytorch-vision
* pillow
* nltk
* pickle
* cuda(*highly recommended*)
## Usage 
Clone the repo to your local system :
```bash
git clone link-to-repo
```
> ***NOTE**: If you are using Colab, use* **!git** instead.

### 1. Preprocessing the data
* Dataset used is Flickr8k ([*download here*](https://www.kaggle.com/shadabhussain/flickr8k)). 
* Extract and move images to a folder named : images and text files to folder named : text
> ***NOTE** : Place both folders alongside the python files.*
* Run the following command in Windows : 
```bash
python preprocess.py
```
> ***NOTE** : Hey buddy :) , for all the python commands , keep in mind that these are for Windows. If using Linux/Mac use **python3**. On Colab use **!python**.*

### 2. Training the data
* Load the tensorboard (optional):
```bash
load_ext tensorboard
tensorboard --logdir=runs
```
> ***NOTE** : On Colab, use the command below.*
```bash
%load_ext tensorboard
%tensorboard --logdir=runs
```
* Run the following command : 
```bash
python train.py
```
```bash
usage: train.py [-h] [-model MODEL] [-dir DIR] [-save_epoch SAVE_EPOCH]
                [-learning_rate LEARNING_RATE] [-num_epoch NUM_EPOCH]
                [-hidden_dim HIDDEN_DIM] [-embedding_dim EMBEDDING_DIM]

optional arguments:
  -h, --help            show this help message and exit
  -model MODEL          Encoder CNN architecture.Default: 'resnet18', other
                        option is 'inception' (Inception_v3). Model dir is
                        automatically saved with name of model +
                        current_datetime.
  -dir DIR              Training Directory path, default: 'train'
  -save_epoch SAVE_EPOCH
                        Epochs after which model saves checkpoint, default : 2
  -learning_rate LEARNING_RATE
                        Adam optimizer learning rate, default : 1e-3 (0.001)
  -num_epoch NUM_EPOCH  Number of epochs, default : 10
  -hidden_dim HIDDEN_DIM
                        Dimensions in hidden state of LSTM decoder, default :
                        512
  -embedding_dim EMBEDDING_DIM
                        Dimensions of encoder output, default : 512
```

### 3. Testing the model
* Run the following command : 
```bash
python test.py model_dir <MODEL_DIR> filename <FILENAME> epoch <EPOCH>
```
```bash
usage: test.py [-h] [-model MODEL] [-test_dir TEST_DIR]
               filename epoch model_dir

positional arguments:
  filename            Image filename.
  epoch               Number of epochs model has been trained for.
  model_dir           Saved model directory, which has name of format: model +
                      current_datetime.

optional arguments:
  -h, --help          show this help message and exit
  -model MODEL        Encoder CNN architecture.Default: 'resnet18', other
                      option is 'inception' (Inception_v3).
  -test_dir TEST_DIR  Test dataset directory name, default: 'test'.
```
  
### 4. Validation 
* Run the following command : 
```bash
python validation.py model_dir <MODEL_DIR>
```
```bash
usage: validation.py [-h] [--model MODEL] [--dir DIR]
                     [--save_epoch SAVE_EPOCH] [--num_epoch NUM_EPOCH]
                     model_dir

positional arguments:
  model_dir             Saved model directory, which has name of format: model
                        + current_datetime.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Default: 'resnet18', other option is 'inception'
                        (Inception_v3).
  --dir DIR             Development Directory path, default: 'dev'
  --save_epoch SAVE_EPOCH
                        Epochs after which trained model has saved checkpoint,
                        default : 2
  --num_epoch NUM_EPOCH
                        Number of epochs model was trained for, default : 10
```
> ***NOTE** : The save_epoch and num_epoch should match with your corresponding training model .*

## Factual Details
- **Title** : Deep Visual-Semantic Alignments for Generating Image Descriptions
- **Dated** : 2015
- **Authors** : Andrej Karpathy, Li Fei-Fei
- **University** : Department of Computer Science, Stanford University
- **Field** : Deep Learning, NLP, CNN, Image Captioning

## Contributed by
- [*Gurbaaz Singh Nandra*](https://github.com/gurbaaz27)

# Summary

## Introduction
The main motivation behind the paper was to design a model capable enough to reason various parts of image and use them to generate a rich description. This is very easy for humans, even for a small child, but machine dosnt relate tokens with image itself. The latter would be great help in CCTV cameras, blind aid or even search engines. Earlier models tackling the same exist but rely on some hard-coded visual concepts and sentence templates, limiting the scope of model. The main two contributions of this model are:

* A deep neural network model that understands the latent alignment or relation between segments of sentences and the region of the image that they describe. Our model associates the two modalities through a common multimodal embedding.
* A multimodal Recurrent Neural Network architecture that takes an input image and generates its description in text, which surpasses retrieval based baselines, and produce sensible qualitative predictions.
## Model architecture
The model consists of encoder and decoder models that communicate through a latent space vector. Basically we map an image to some intractable latent space via encoding and map the latent space representation of the image to the sentence space via decoding.
<p align="center">
  <img width="973" height="331" src="assets/paper1.JPG">
</p>

#### Encoder
We need to provide image as fixed size of vector to generate text, hence a convolutional neural network is used to encode our images. Here transfer learning is preferred, pretained on ImageNet dataset, to cater the constratints of resources and computation. Torchvision has many pretrained models, and any model can be used like ResNet, AlexNet, Inception_v3, DenseNet. We have to remove the last softmax layer which is for classification purpose, as we only need an encoded vector, of size 512 or 1024 usually.
<p align="center">
  <img width="242" height="39" src="assets/paper4.JPG">
</p>

#### Decoder
This part is a Recurrent Neural Network with LSTM(Long Short Term Memory) cells. A Bidirectional Recurrent Neural Network (BRNN) to compute
the word representations. The BRNN takes a sequence of N words (encoded in a 1-of-k representation) and transforms each one into an h-dimensional vector. However, the representation of each word is enriched by a variably-sized context around that word. Using the index t = 1 . . . N to denote the position of a word in a sentence, the precise form of the BRNN is as follows(*here f is ReLU activation function, hence f(x) = max(0,x)*) : 
<p align="center">
  <img width="277" height="181" src="assets/paper5.JPG">
</p>
<p align="center">
  <img width="277" height="250" src="assets/paper2.JPG">
  <img width="277" height="250" src="assets/paper3.JPG">
</p>

#### Loss
Since the supervision is at the level of entire images and sentences, the strategy is to formulate an image-sentence score as a function of the individual regionword scores. Obviously, a sentence-image pair should have a high matching score if its words have a confident support in the image. The dot product between the i-th region and t-th word as a
measure of similarity and use it to define the score between image k and sentence l as:
<p align="center">
  <img  src="assets/paper7.JPG">
</p>
Here, gk is the set of image fragments in image k and gl
is the set of sentence fragments in sentence l. The indices
k, l range over the images and sentences in the training set.
<p align="center">
  <img width="270" height="240" src="assets/paper6.JPG">
</p>

## Results :
* Training time (model = resnet18):
<pre><code>Epoch : 0 , Avg_loss = 3.141907, Time = 9.89 mins
Epoch : 1 , Avg_loss = 2.978030, Time = 9.89 mins
Epoch : 2 , Avg_loss = 2.879061, Time = 9.88 mins
Epoch : 3 , Avg_loss = 2.800483, Time = 9.90 mins
Epoch : 4 , Avg_loss = 2.734463, Time = 9.88 mins
Epoch : 5 , Avg_loss = 2.676081, Time = 9.90 mins
Epoch : 6 , Avg_loss = 2.625130, Time = 9.89 mins
Epoch : 7 , Avg_loss = 2.579518, Time = 9.90 mins
Epoch : 8 , Avg_loss = 2.538572, Time = 9.90 mins
Epoch : 9 , Avg_loss = 2.501715, Time = 9.90 mins
      </code></pre>
<p align="center">
  <img width="697" height="398" src="assets/training_loss.JPG">
</p>

> ***NOTE** : Here loss plotted is total loss of dataset, which is for 6000 X 5 = 30000 captions. Mean loss is loss plotted divided by 30000 .*

* Validation time (model = resnet18):
<pre><code>
0 3.007883
2 3.035437
4 3.088141
6 3.161836
8 3.253206
</code></pre>
<p align="center">
  <img width="697" height="398" src="assets/validation_loss.JPG">
</p>

> ***NOTE** : Train for more than 50 epochs for optimum results . Training is highly GPU intensive!*

## Acknowledgement
- [Medium : Captioning Images with CNN and RNN, using PyTorch](https://medium.com/@stepanulyanin/captioning-images-with-pytorch-bc592e5fd1a3)
- [Word Embeddings](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)

