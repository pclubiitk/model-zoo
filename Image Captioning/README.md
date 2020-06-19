# Image Captioning
Image Captioning is the process of generating textual description of an image. It uses both Natural Language Processing and Computer Vision to generate the captions. The whole m-RNN architecture contains a language model part, an image part and a multimodal part. The language model part learns the dense feature embedding for each word in the dictionary and stores the semantic temporal context in recurrent layers. The image part contains a deep Convulutional Neural Network (CNN) which extracts image features. The multimodal part connects the language model and the deep CNN together by a one-layer representation.


## Encoder
The Convolutional Neural Network(CNN) can be thought of as an encoder. The input image is given to CNN to extract the features. The last hidden state of the CNN is connected to the Decoder.


## Decoder
The Decoder is a Recurrent Neural Network(RNN) which does language modelling up to the word level. The first time step receives the encoded output from the encoder and also the <START> vector.


## Dataset used
* [Flickr8k_Dataset.zip](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
* [Flickr8k_text.zip](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)
Download the dataset and unzip it.


## 1. Help log
```
usage: train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--optimizer OPTIMIZER] [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       No of EPOCHS: default 20
  --batch_size BATCH_SIZE
                        Batch size, default 256
  --optimizer OPTIMIZER
                        Optimizer, default RMSprop
  --model MODEL         Image features extraction model to be used, default
                        InceptionV3
```


## 2. Training Results
For the image, Results are

(./assets/image.png)
Model is trained on less number of images because of limited RAM. Results will be much better when trained on entire training data.

| Model & Config | Argmax |
| :--- | :--- |
| **Vgg16 + LSTM in the decoder** <ul><li>Epochs = 10</li><li>Batch Size = 128</li><li>Optimizer = adam</li><li>Training Size = 1500</li></ul> |<ul><br>Two people are riding bicycles on the street</br></ul> |
| **Vgg16 + Bi-directional LSTM in the decoder** <ul><li>Epochs = 10</li><li>Batch Size = 128</li><li>Optimizer = adam</li><li>Training Size = 1500</li></ul> |<ul><br>Two girls are riding bicycle on the streets</br></ul> |
| **Vgg16 + No LSTM layer in the decoder** <ul><li>Epochs = 10</li><li>Batch Size = 128</li><li>Optimizer = adam</li><li>Training Size = 1500</li></ul> |<ul><br>man in grey shirt is riding down the street</br></ul> |
| **InceptionV3 + LSTM in the decoder** <ul><li>Epochs = 10</li><li>Batch Size = 128</li><li>Optimizer = RMSprop</li><li>Training Size = 2000</li></ul> |<ul><br>man is riding down road</br></ul> |
| **InceptionV3 + Bi-directional LSTM in the decoder** <ul><li>Epochs = 10</li><li>Batch Size = 128</li><li>Optimizer = RMSprop</li><li>Training Size = 2000</li></ul> |<ul><br>person in red helmet riding bike down road</br></ul> |
| **InceptionV3 + No LSTM layer in the decoder** <ul><li>Epochs = 10</li><li>Batch Size = 128</li><li>Optimizer = RMSprop</li><li>Training Size = 2000</li></ul> |<ul><br>person in red rides gear rides his bicycle in before woodland</br></ul> |
| **InceptionV3 + Bi-directional layer in the decoder** <ul><li>Epochs = 10</li><li>Batch Size = 256</li><li>Optimizer = RMSprop</li><li>Training Size = 2000</li></ul> |<ul><br>man riding bicycle down dirt hill</br></ul> |
| **InceptionV3 + Bi-directional layer in the decoder** <ul><li>Epochs = 20</li><li>Batch Size = 256</li><li>Optimizer = RMSprop</li><li>Training Size = 2000</li></ul> |<ul><br>man riding bicycle down dirt hill</br></ul> |


## Contributers
* [Shreya Sharma](https://github.com/shreya0205/)


## References
* [Explain Images with Multimodal Recurrent Neural Networks](https://arxiv.org/pdf/1410.1090.pdf)
* [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555v2.pdf)
* [CS231n Winter 2016 Lecture 10 Recurrent Neural Networks, Image Captioning, LSTM](https://www.youtube.com/watch?v=cO0a0QYmFm8&feature=youtu.be&t=32m25s)

