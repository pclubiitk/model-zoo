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

![results][]

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
        
================================================================ 

         Rearrange-1               [-1, 32, 96]               0
         
            Linear-2              [-1, 32, 512]          49,664
            
         LayerNorm-3              [-1, 32, 512]           1,024
            Linear-4             [-1, 512, 512]          16,896
              GELU-5             [-1, 512, 512]               0
           Dropout-6             [-1, 512, 512]               0
            Linear-7              [-1, 512, 32]          16,416
           Dropout-8              [-1, 512, 32]               0
         LayerNorm-9              [-1, 32, 512]           1,024
           Linear-10              [-1, 32, 512]         262,656
             GELU-11              [-1, 32, 512]               0
          Dropout-12              [-1, 32, 512]               0
           Linear-13              [-1, 32, 512]         262,656
          Dropout-14              [-1, 32, 512]               0
       MixerBlock-15              [-1, 32, 512]               0
        LayerNorm-16              [-1, 32, 512]           1,024
           Linear-17             [-1, 512, 512]          16,896
             GELU-18             [-1, 512, 512]               0
          Dropout-19             [-1, 512, 512]               0
           Linear-20              [-1, 512, 32]          16,416
          Dropout-21              [-1, 512, 32]               0
        LayerNorm-22              [-1, 32, 512]           1,024
           Linear-23              [-1, 32, 512]         262,656
             GELU-24              [-1, 32, 512]               0
          Dropout-25              [-1, 32, 512]               0
           Linear-26              [-1, 32, 512]         262,656
          Dropout-27              [-1, 32, 512]               0
       MixerBlock-28              [-1, 32, 512]               0
        LayerNorm-29              [-1, 32, 512]           1,024
           Linear-30             [-1, 512, 512]          16,896
             GELU-31             [-1, 512, 512]               0
          Dropout-32             [-1, 512, 512]               0
           Linear-33              [-1, 512, 32]          16,416
          Dropout-34              [-1, 512, 32]               0
        LayerNorm-35              [-1, 32, 512]           1,024
           Linear-36              [-1, 32, 512]         262,656
             GELU-37              [-1, 32, 512]               0
          Dropout-38              [-1, 32, 512]               0
           Linear-39              [-1, 32, 512]         262,656
          Dropout-40              [-1, 32, 512]               0
       MixerBlock-41              [-1, 32, 512]               0
        LayerNorm-42              [-1, 32, 512]           1,024
           Linear-43             [-1, 512, 512]          16,896
             GELU-44             [-1, 512, 512]               0
          Dropout-45             [-1, 512, 512]               0
           Linear-46              [-1, 512, 32]          16,416
          Dropout-47              [-1, 512, 32]               0
        LayerNorm-48              [-1, 32, 512]           1,024
           Linear-49              [-1, 32, 512]         262,656
             GELU-50              [-1, 32, 512]               0
          Dropout-51              [-1, 32, 512]               0
           Linear-52              [-1, 32, 512]         262,656
          Dropout-53              [-1, 32, 512]               0
       MixerBlock-54              [-1, 32, 512]               0
        LayerNorm-55              [-1, 32, 512]           1,024
           Linear-56             [-1, 512, 512]          16,896
             GELU-57             [-1, 512, 512]               0
          Dropout-58             [-1, 512, 512]               0
           Linear-59              [-1, 512, 32]          16,416
          Dropout-60              [-1, 512, 32]               0
        LayerNorm-61              [-1, 32, 512]           1,024
           Linear-62              [-1, 32, 512]         262,656
             GELU-63              [-1, 32, 512]               0
          Dropout-64              [-1, 32, 512]               0
           Linear-65              [-1, 32, 512]         262,656
          Dropout-66              [-1, 32, 512]               0
       MixerBlock-67              [-1, 32, 512]               0
        LayerNorm-68              [-1, 32, 512]           1,024
           Linear-69             [-1, 512, 512]          16,896
             GELU-70             [-1, 512, 512]               0
          Dropout-71             [-1, 512, 512]               0
           Linear-72              [-1, 512, 32]          16,416
          Dropout-73              [-1, 512, 32]               0
        LayerNorm-74              [-1, 32, 512]           1,024
           Linear-75              [-1, 32, 512]         262,656
             GELU-76              [-1, 32, 512]               0
          Dropout-77              [-1, 32, 512]               0
           Linear-78              [-1, 32, 512]         262,656
          Dropout-79              [-1, 32, 512]               0
       MixerBlock-80              [-1, 32, 512]               0
        LayerNorm-81              [-1, 32, 512]           1,024
           Linear-82             [-1, 512, 512]          16,896
             GELU-83             [-1, 512, 512]               0
          Dropout-84             [-1, 512, 512]               0
           Linear-85              [-1, 512, 32]          16,416
          Dropout-86              [-1, 512, 32]               0
        LayerNorm-87              [-1, 32, 512]           1,024
           Linear-88              [-1, 32, 512]         262,656
             GELU-89              [-1, 32, 512]               0
          Dropout-90              [-1, 32, 512]               0
           Linear-91              [-1, 32, 512]         262,656
          Dropout-92              [-1, 32, 512]               0
       MixerBlock-93              [-1, 32, 512]               0
        LayerNorm-94              [-1, 32, 512]           1,024
           Linear-95             [-1, 512, 512]          16,896
             GELU-96             [-1, 512, 512]               0
          Dropout-97             [-1, 512, 512]               0
           Linear-98              [-1, 512, 32]          16,416
          Dropout-99              [-1, 512, 32]               0
       LayerNorm-100              [-1, 32, 512]           1,024
          Linear-101              [-1, 32, 512]         262,656
            GELU-102              [-1, 32, 512]               0
         Dropout-103              [-1, 32, 512]               0
          Linear-104              [-1, 32, 512]         262,656
         Dropout-105              [-1, 32, 512]               0
      MixerBlock-106              [-1, 32, 512]               0
       LayerNorm-107              [-1, 32, 512]           1,024
          Linear-108             [-1, 512, 512]          16,896
            GELU-109             [-1, 512, 512]               0
         Dropout-110             [-1, 512, 512]               0
          Linear-111              [-1, 512, 32]          16,416
         Dropout-112              [-1, 512, 32]               0
       LayerNorm-113              [-1, 32, 512]           1,024
          Linear-114              [-1, 32, 512]         262,656
            GELU-115              [-1, 32, 512]               0
         Dropout-116              [-1, 32, 512]               0
          Linear-117              [-1, 32, 512]         262,656
         Dropout-118              [-1, 32, 512]               0
      MixerBlock-119              [-1, 32, 512]               0
       LayerNorm-120              [-1, 32, 512]           1,024
          Linear-121             [-1, 512, 512]          16,896
            GELU-122             [-1, 512, 512]               0
         Dropout-123             [-1, 512, 512]               0
          Linear-124              [-1, 512, 32]          16,416
         Dropout-125              [-1, 512, 32]               0
       LayerNorm-126              [-1, 32, 512]           1,024
          Linear-127              [-1, 32, 512]         262,656
            GELU-128              [-1, 32, 512]               0
         Dropout-129              [-1, 32, 512]               0
          Linear-130              [-1, 32, 512]         262,656
         Dropout-131              [-1, 32, 512]               0
      MixerBlock-132              [-1, 32, 512]               0
       LayerNorm-133              [-1, 32, 512]           1,024
          Linear-134             [-1, 512, 512]          16,896
            GELU-135             [-1, 512, 512]               0
         Dropout-136             [-1, 512, 512]               0
          Linear-137              [-1, 512, 32]          16,416
         Dropout-138              [-1, 512, 32]               0
       LayerNorm-139              [-1, 32, 512]           1,024
          Linear-140              [-1, 32, 512]         262,656
            GELU-141              [-1, 32, 512]               0
         Dropout-142              [-1, 32, 512]               0
          Linear-143              [-1, 32, 512]         262,656
         Dropout-144              [-1, 32, 512]               0
      MixerBlock-145              [-1, 32, 512]               0
       LayerNorm-146              [-1, 32, 512]           1,024
          Linear-147             [-1, 512, 512]          16,896
            GELU-148             [-1, 512, 512]               0
         Dropout-149             [-1, 512, 512]               0
          Linear-150              [-1, 512, 32]          16,416
         Dropout-151              [-1, 512, 32]               0
       LayerNorm-152              [-1, 32, 512]           1,024
          Linear-153              [-1, 32, 512]         262,656
            GELU-154              [-1, 32, 512]               0
         Dropout-155              [-1, 32, 512]               0
          Linear-156              [-1, 32, 512]         262,656
         Dropout-157              [-1, 32, 512]               0
      MixerBlock-158              [-1, 32, 512]               0
      AdaptiveAvgPool1d-159        [-1, 512, 1]               0
          Linear-160                   [-1, 10]           5,130
          
==============================================================  
Total params: 6,782,858

Trainable params: 6,782,858

Non-trainable params: 0

