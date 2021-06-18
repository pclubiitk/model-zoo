# RepVGG

## Usage
### Train & Test
```
python3 main.py --model "A0"
```
Other models: A1, A2, B0, B1, B2, B3

### References
* [RepVGG Paper](https://arxiv.org/pdf/2101.03697.pdf)
* [Tensorflow Documentation on Custom Layers](https://www.tensorflow.org/tutorials/customization/custom_layers) 
### Contributed by:
* [Aditya Tanwar](https://github.com/cliche-niche/)

## Summary

### Introduction
A simple but powerful architecture of convolutional neural network, which has a VGG-like inference time body composed of nothing but a stack of 3x3 convolution and ReLU, while the 
training-time model has a multi-branch topology.

Such decoupling of the training time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG.

### Architecture
The model consists of 5 stages followed by a Global Average Pooling Layer (2D) and a Fully Connected Layer (for Image Classification). 
In each stage, the first layer is used for downsampling by using `3x3 Conv2D` and `1x1 Conv2D` layers with stride=2. The rest of the layers each have `3x3 Conv2D` (with stride=1), `1x1 Conv2D` (with stride=1), and a shortcut connection. `Branch Normalization` is used just before every addition, and each addition is followed by a `ReLU` activation layer.
<img src ="https://github.com/cliche-niche/model-zoo-submissions/blob/main/RepVGG/assets/architecture.PNG?raw=true">

### Reparameterization
At inference time, only `3x3 Conv2D` layers and `ReLU` activation layers are used. The kernel and bias for `3x3 Conv2D` layers are obtained by Re-parameterization of the weights obtained during training. The multi-branch topology is useful for training, but is slower for testing and more memory consuming, while on the other hand, plain models are faster but have a poor accuracy. 
This is overcome by this model, by using multi-branch topology for training and then reparameterizing it to a plain model for inference time.

The output of a single layer can be represented as the following

<img src ="https://github.com/cliche-niche/model-zoo-submissions/blob/main/RepVGG/assets/add.PNG?raw=true">

- **M<sup>(1)</sup>** is the input
- **M<sup>(2)</sup>** is the resulting output
- **W** are the kernels of convolutional layers
- Identity layer (present when `input channel = output channels`) does not have a kernel.
- Superscript, <sup>(1)</sup> denotes weights of the `1x1 Conv`, <sup>(3)</sup> denotes weights of the `3x3 Conv`, and <sup>(0)</sup> denotes weights of the `Identity Shortcut` layer.


The bn() mentioned above, expands to:

<img src ="https://github.com/cliche-niche/model-zoo-submissions/blob/main/RepVGG/assets/bn.PNG?raw=true">

Where, _**&mu;**_= Accumulated mean, _**&sigma;**_= Standard Deviation, _**&gamma;**_= Learned Scaling Factor, _**&beta;**_= Bias of the BN layer used after the convolution layers. So, the kernel of the inference time `3x3 Conv2D` Layers (= _**W'**_), and its bias (= _**b'**_), are given by:

<img src ="https://github.com/cliche-niche/model-zoo-submissions/blob/main/RepVGG/assets/wb.PNG?raw=true">

Finally, we have just one layer, that can replace them.

<img src ="https://github.com/cliche-niche/model-zoo-submissions/blob/main/RepVGG/assets/fin.PNG?raw=true">

Where two layers (and one identity shortcut) were there in training, now there is only one layer during inference, which does the work of all three, thus, saving time as well as memory.
