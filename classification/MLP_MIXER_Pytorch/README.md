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

MLP Mixer is based on multi layer perceptron it does not use modern days CNN , It has two kinds of multi layer preceptrons one is directly applied to image patches , which are created original image then we transpose the layer and apply MLP layer across patches [here](https://github.com/imad08/model-zoo-submissions/blob/main/REPVGG/REPVGG_with_complete_reparamaterization_.ipynb) 

![fusing batch normalization and convolutions for reparametrization](https://media.arxiv-vanity.com/render-output/4507333/x1.png)

# Architecture of REPVGG

REPVGG heavily use 3x3 kernels and it has plain topology ,and it does not uses maxpool 2d the reason is author wants that the architecture has same kind of operators . In REPVGG we arrange 5 block architecture we can say that one stage , which uses 3x3 kernels and BatchNorm layers . In first layer of ech stage down the sample using the stride of (2,2). the first stage operates with large resolution hence in first stage block we just use one layer for lower latency . last stages has most channels.And most number of layers is in second last stage same as previous resnet architectures .

![main_architecture specifications](https://github.com/imad08/model-zoo/blob/master/classification/REPVGG_Pytorch/Assets/Screenshot%20(2863).png)
![REPVGG Architecture](https://media.arxiv-vanity.com/render-output/4507333/x3.png)


 
# Reparamateriztion is key in repvgg

The major difference that repvgg architecture has as compared to for RESNET etc , is the state of the art reparametrization . There are various kind of reparametrization removes batchnorm from Identity , Post addition of Batch Norm addition of ReLU in branches addition of 1X1 kernel . Most important reparametrization is fusing of kernel and BN in block. 


![fusing batch normalization and convolutions for reparametrization](https://pic3.zhimg.com/80/v2-686b26f8a41b54c10d76d7a90a6d8bbe_1440w.jpg)

# Results
data - cifar 10 

1hr 58min 43sec 80 epoch 
accuracy approx 86%

Total params: 15,681,066

Trainable params: 15,681,066

Non-trainable params: 0

Params size (MB): 59.82

Estimated Total Size (MB): 196.78

