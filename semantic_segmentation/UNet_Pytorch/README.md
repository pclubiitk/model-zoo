# Pytorch Implementation of UNet Model
### Usage
```bash
$ python3 main.py --arguments
```
NOTE: on Colab Notebook use following command:
```python
!git clone link-to-repo
%run main.py --arguments
```
The arguments are as follows:
```python
optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  --input_channels INPUT_CHANNELS
  --num_classes NUM_CLASSES
  --init_features INIT_FEATURES
  --padding PADDING
  --optimizer {Adam,SGD}
  --lr LR, --learning_rate LR
  --weight_decay WEIGHT_DECAY
  --min_lr MIN_LR
  --early_stopping EARLY_STOPPING
  --device DEVICE
  --num_workers NUM_WORKERS
```
To download the dataset, run the following command and upload your Kaggle API token:
```python
!pip install -q kaggle
from google.colab import files
files.upload()

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
!unzip /content/lgg-mri-segmentation.zip
```

### References
* Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation"
[arXiv:1505.04597]( https://arxiv.org/abs/1505.04597 )

### Contributed by:
* [Rishav Bikarwar](https://github.com/ris27hav)
<p>&nbsp;</p>

# Summary

## Introduction

Convolutional Networks have existed for a long time. However, their success was limited due to the lack of a large number of available training sets and the high computational power needed to process deeper networks.

Convolutional Networks are typically used for classification tasks.
However, in many visual tasks, especially in biomedical image processing, there is a need for image localization, i.e., a class label is supposed to be assigned to each pixel. Moreover, there is limited availability of training images in biomedical tasks. 

In this paper, the authors build upon a “**fully convolutional network**”. They modified this architecture in such a way that it yields more precise segmentation by training it with very few images, as shown in figure 3. 
* Their main idea was to supplement a usual contracting network by successive layers, where **upsampling operators replace pooling operators**. Hence, these layers increase the resolution of the output. In order to localize, **high-resolution features from the contracting path are combined with the upsampled output**. A successive convolution layer can then learn to assemble a more precise output based on this information. 
* In the upsampling part, the authors used many feature channels, which allow the network to propagate context information to higher resolution layers.
Consequently, the expansive path is more or less symmetric to the contracting path and yields a **u-shaped architecture**. 
* The network does not have any fully connected layers. It only uses the valid part of each convolution, i.e., the segmentation map only contains the pixels, for which the whole context is available in the input image. This strategy allows the seamless segmentation of arbitrarily large images by **an overlap-tile strategy** (Figure 1). 
* To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring the input image. This tiling strategy is vital to apply the network to large images since otherwise, the resolution would be limited by the GPU memory. 

|![Architecture](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet_Pytorch/assets/fig2.png?raw=true)|
|:--:|
|(Fig. 1) Overlap-tile strategy for seamless segmentation of arbitrary large images. Prediction of the segmentation in the yellow area, requires image data within the blue area as input. Missing input data is extrapolated by mirroring.|

If there is very little training data available, excessive data augmentation can be used by applying elastic deformations.  This is particularly important in biomedical segmentation since deformation used to be the most common variation in tissue, and realistic deformations can be simulated efficiently. 

Another challenge in many cell segmentation tasks is separating touching objects of the same class, as shown in figure 3. The use of a weighted loss can fix this issue, where the separating background labels between touching cells obtain a considerable weight in the loss function.

|![Architecture](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet_Pytorch/assets/fig3.png?raw=true)|
|:--:|
|(Fig. 2) HeLa cells on glass recorded with DIC (differential interference contrast) microscopy. (a) raw image. (b) overlay with ground truth segmentation. Different colors indicate different instances of the HeLa cells. (c) generated segmentation mask (white: foreground, black: background). (d) map with a pixel-wise loss weight to force the network to learn the border pixels.|

---

## Network-Architecture

|![Architecture](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet_Pytorch/assets/arch.png?raw=true)|
|:--:|
|(Fig. 3) U-net architecture (example for 32x32 pixels in the lowest resolution).|

The network architecture consists of a contracting path (left side) and an expansive path (right side). 

The **contracting path** follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step, the authors doubled the number of feature channels. 

Every step in the **expansive path** consists of an upsampling of the feature map followed by a 2x2 convolution (“**up-convolution**”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. Upsampling can be achived by either directly upsampling the feature map using interpolation i.e. without any parameters or by using transpose convolution that has trainable kernel parameters. The cropping is necessary due to the loss of border pixels in every convolution. At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. 

In total the network has **23 convolutional layers**. 
To allow a seamless tiling of the output segmentation map, it is important to select the input tile size such that all 2x2 max-pooling operations are applied to a layer with an even x- and y-size.

---
## Experimentation and Results

This network was applied to the segmentation of neuronal structures in EM stacks, where it out-performed the other networks at that time. Furthermore, it was trained for cell segmentation in light microscopy images from the ISBI cell tracking challenge 2015. Here the authors won with a large margin on the two most challenging 2D transmitted light datasets.


|![Results Table 1](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet_Pytorch/assets/results1.png?raw=true)|
|:--:|
|(Table.1)  Ranking on the EM segmentation challenge (as of march 6th, 2015), sorted by warping error.|

|![Results Table 2](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet_Pytorch/assets/results2.png?raw=true)|
|:--:|
|(Table.2)  Segmentation results (IOU) on the ISBI cell tracking challenge 2015.|


>I have trained the UNet model on Brain MRI Segmentation dataset. Originally, batch norm was not used in UNet as it was not discovered till then, but I have used it as otherwise learning was very slow (really slow :(). I have also used padding to get the output size same as input size. To train the model faster, I have lowered the features by half for each layer.

Model Architecture (used for training):
``` python
UNet(
  (cb1): Conv_Block(
    (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (cb2): Conv_Block(
    (conv1): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (cb3): Conv_Block(
    (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (cb4): Conv_Block(
    (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (cb5): Conv_Block(
    (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (ub1): Upsampling_Block(
    (conv_transpose): ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
    (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
  )
  (ub2): Upsampling_Block(
    (conv_transpose): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
    (conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
  )
  (ub3): Upsampling_Block(
    (conv_transpose): ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2))
    (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
  )
  (ub4): Upsampling_Block(
    (conv_transpose): ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2))
    (conv1): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
  )
  (conv1x1): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1))
)
```

I trained the model for 90 epochs. Training for more epochs can definetely give better results. Loss and IoU graphs are given below:

![Results](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet_Pytorch/assets/results3.png?raw=true)

>Best IoU / Dice Coefficient obtained on train set :  **0.64 / 0.78**
>
>Best IoU / Dice Coefficient obtained on validation set : **0.62 / 0.76**

Best dice-coefficient obtained by the submissions in Kaggle is **0.88**


---

## Conclusion

U-Net is able to do image localisation by predicting the image pixel by pixel. This network is strong enough to do good prediction based on even few data sets by using excessive data augmentation techniques (especially with elastic deformations). There are many applications of image segmentation using UNet and it also occurs in lots of competitions.

>Results that I obtained can certainly be improved by training for more epochs. Furthermore, data-augmentation can also be used. Given below are some exapmles of predicted masks I obtained with this model. (not great, but good for starters)


![Outputs](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet_Pytorch/assets/predictions.png?raw=true)