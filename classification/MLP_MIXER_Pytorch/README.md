# MLP-Mixer: An all-MLP Architecture for Vision

# Contributed by:
 * [Imad Khan](https://github.com/imad08)

### Usage
```bash
$ python3 main.py 
```
NOTE: on Colab Notebook use following command:
```python
!git clone link-to-repo
%run main.py 
```

# References

* **Title**:MLP-Mixer: An all-MLP Architecture for Vision
* **Authors**: Ilya Tolstikhin∗
, Neil Houlsby∗
, Alexander Kolesnikov∗
, Lucas Beyer∗
,
Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner,
Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy
* **Link**: https://arxiv.org/pdf/2105.01601v4.pdf
* **Year**: 2021



# Summary 

MLP Mixer is based on multi layer perceptron it does not use modern days CNN , It has two kinds of multi layer preceptrons one is directly applied to image patches , which are created original image then we transpose the layer and apply MLP layer across patches In the extreme case, Multi layer perceptron architecture can be seen as a very special CNN, which uses 1×1 convolutions
for channel mixing, and single-channel depth-wise convolutions of a full receptive field and parameter
sharing for token mixing. However, the converse is not true as typical CNNs are not special cases of
Mixer. Furthermore, a convolution is more complex than the plain matrix multiplication in MLPs as
it requires an additional costly reduction to matrix multiplication and/or specialized implementation.
If you want to see training of cifar10 dataset using above architecture you can refer [here](https://github.com/imad08/MLP-Mixer/blob/main/MLP.ipynb)

# Architecture of MLP

Figure depicts the macro-structure of Mixer. It accepts a sequence of linearly projected image
patches (also referred to as tokens) shaped as a “patches × channels” table as an input, and maintains
this dimensionality. Mixer makes use of two types of MLP layers as told above one is applied to image patches , which are created original image then we transpose the layer and apply MLP layer across patches  . The channel-mixing MLPs allow communication between different channels; It is similar to attention models. 


![main architecture](https://github.com/imad08/model-zoo/blob/master/classification/MLP_MIXER_Pytorch/assets/Screenshot%20(2961).png)
# How perceptrons layers of multiple patches are mixed
 Modern deep vision architectures consist of layers that mix features (i) at a given spatial location,
(ii) between different spatial locations, or both at once. In CNNs, (ii) is implemented with N × N
convolutions (for N > 1) and pooling. Neurons in deeper layers have a larger receptive field [1, 28].
At the same time, 1×1 convolutions also perform (i), and larger kernels perform both (i) and (ii).
In Vision Transformers and other attention-based architectures, self-attention layers allow both (i)
and (ii) and the MLP-blocks perform (i). The idea behind the Mixer architecture is to clearly separate
the per-location (channel-mixing) operations (i) and cross-location (token-mixing) operations (ii).
Both operations are implemented with MLPs.
![single block of mlp](https://github.com/imad08/model-zoo/blob/master/classification/MLP_MIXER_Pytorch/assets/Screenshot%20(2963).png)


# Results
data - cifar 10 

2hr 28min 43sec 80 epoch 
accuracy approx 90.1%

Total params: 6,782,858
Trainable params: 6,782,858
Non-trainable params: 0

----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 87.15
Params size (MB): 25.87
Estimated Total Size (MB): 113.04

