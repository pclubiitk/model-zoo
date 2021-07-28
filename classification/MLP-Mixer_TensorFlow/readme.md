# MLP-Mixer: An all-MLP Architecture for Vision
---
> #### Contents:
> - ##### Installation & Usage
> - ##### Add more stuff here

### Installation & Usage
---
First you have to make sure you have installed the packages needed, you need to run :
`$ pip install -r requirements.txt`

We are making use of the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset for the training purpose. running `load_dataset.py` will download the dataset :
`$ python3 load_dataset.py`

Now conifgure the parameters in the `model.py` file, Please note that I have provided weights according that I myself trained on the already given parameters, So if you're planning to change any parameters that would affect the model, please consider deleting the files in the `models` directory. 

If you're wanting to train the network by yourself, set the `train_bool = True` in `model.py` .

Finally, You will have to run the `main.py` file to get the output according to your preferences entered.

`$ python main.py`

### Contributed by:
---
+[Prem Bharwani](https://github.com/prembharwani)
### References
---
* Title: MLP-Mixer: An all-MLP Architecture for Vision
* Authors: Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer,Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner,Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy
* Link: https://arxiv.org/abs/2105.01601
* Tags: Image Classification
* Year: 2021

### Summary
---
##### Motivation 
The main motive behind this paper is to invoke a push in research towards a simple architecture. As mentioned in the paper, Convolutions and Attention based models are capable of producing good results, but they are not necessary. With the help of this paper, the authors want to show that a very simple architecture involving basic operations can produce competetive results.

##### Architecture

The architecture for this model simply consists only of Multi Layer Pereceptrons (MLP's). The architecture involves two types of MLP's. 
* Type 1: Performs mixing of per-location features, i.e involves mixing of features between channels at the same location. This is Called the Channel Mixing MLP.
* Type 2: Involves mixing of features from different spatial locations, i.e across different patches(Explained ahead). This is called the Token Mixing MLP.

So we divide the image into patches. These are also referred to as tokens. The architecture is fairly simple and straight forward.

We have MLP as the low level component which comprises of two fully connected layers, seperated by a GELU activation.
![MLP representation](/classification/MLP-Mixer_tensorflow/assets/MLP.png "MLP Block Representation")

We have MLP Mixer block, which consists of two MLP's, each serving a different purpose. One of them would perform Token Mixing(mixing features among spatial locations), and the other one Channel Mixing(mixing features at the same location). Also we should notice here that this model also makes use of skip connections. And it makes use of Layer Normalization rather than Batch Normalization. Basic operations like transposing the matrix is also involved here
![Mixer Block Representation](/classification/MLP-Mixer_tensorflow/assets/mixer_block.png "Mixer Layer Representation")

Finally the combination of generating patches, and then reshaping them to a 2D tensor, and then passing them through a number of Mixer Layers, and then finally through a Classifier head.
![Final Architecture Representation](/classification/MLP-Mixer_tensorflow/assets/final_layer.png "Final Architecture Representation")


```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 32, 32, 3)]  0                                            
__________________________________________________________________________________________________
data_augmentation (Sequential)  (None, 64, 64, 3)    7           input_1[0][0]                    
__________________________________________________________________________________________________
tf.compat.v1.shape (TFOpLambda) (4,)                 0           data_augmentation[0][0]          
__________________________________________________________________________________________________
tf.image.extract_patches (TFOpL (None, 8, 8, 192)    0           data_augmentation[0][0]          
__________________________________________________________________________________________________
tf.__operators__.getitem (Slici ()                   0           tf.compat.v1.shape[0][0]         
__________________________________________________________________________________________________
tf.reshape (TFOpLambda)         (None, 64, 192)      0           tf.image.extract_patches[0][0]   
                                                                 tf.__operators__.getitem[0][0]   
__________________________________________________________________________________________________
dense_16 (Dense)                (None, 64, 256)      49408       tf.reshape[0][0]                 
__________________________________________________________________________________________________
sequential_8 (Sequential)       (None, 64, 256)      167680      dense_16[0][0]                   
__________________________________________________________________________________________________
global_average_pooling1d (Globa (None, 256)          0           sequential_8[0][0]               
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 256)          0           global_average_pooling1d[0][0]   
__________________________________________________________________________________________________
dense_17 (Dense)                (None, 100)          25700       dropout_8[0][0]                  
==================================================================================================
Total params: 242,795
Trainable params: 242,788
Non-trainable params: 7
__________________________________________________________________________________________________
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 32, 32, 3)]  0                                            
__________________________________________________________________________________________________
data_augmentation (Sequential)  (None, 64, 64, 3)    7           input_1[0][0]                    
__________________________________________________________________________________________________
tf.compat.v1.shape (TFOpLambda) (4,)                 0           data_augmentation[0][0]          
__________________________________________________________________________________________________
tf.image.extract_patches (TFOpL (None, 8, 8, 192)    0           data_augmentation[0][0]          
__________________________________________________________________________________________________
tf.__operators__.getitem (Slici ()                   0           tf.compat.v1.shape[0][0]         
__________________________________________________________________________________________________
tf.reshape (TFOpLambda)         (None, 64, 192)      0           tf.image.extract_patches[0][0]   
                                                                 tf.__operators__.getitem[0][0]   
__________________________________________________________________________________________________
dense_16 (Dense)                (None, 64, 256)      49408       tf.reshape[0][0]                 
__________________________________________________________________________________________________
sequential_8 (Sequential)       (None, 64, 256)      167680      dense_16[0][0]                   
__________________________________________________________________________________________________
global_average_pooling1d (Globa (None, 256)          0           sequential_8[0][0]               
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 256)          0           global_average_pooling1d[0][0]   
__________________________________________________________________________________________________
dense_17 (Dense)                (None, 100)          25700       dropout_8[0][0]                  
==================================================================================================
Total params: 242,795
Trainable params: 242,788
Non-trainable params: 7
__________________________________________________________________________________________________
```

### Results
---
I trained the model on CIFAR-100 myself, And I have uploaded my weights in the models directory.
Here's one result of the trained model,
![Test predicitons](/classification/MLP-Mixer_tensorflow/test.png "Test Prediction")

We could have improved using much more efficient parameters.
