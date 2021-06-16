# RepVGG: Making VGG-style ConvNets Great Again Pytorch Implementation

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

* **Title**:RepVGG: Making VGG-style ConvNets Great Again
* **Authors**: Xiaohan Ding,  Xiangyu Zhang,  Ningning Ma, 
Jungong Han,  Guiguang Ding, Jian Sun 
* **Link**: https://arxiv.org/pdf/2101.03697.pdf
* **Year**: 2021



# Summary 

REPVGG is a simple but powerful architecture of CNN which has a VGG like inference time .It runs 101% faster then RESNET 101 ,There are many complicated architecture which has better accuracy then simple architectures, but the drawback of this kind of architecture is that they are difficult to customize . And, has very high inference time .REPVGG has various advantages like , Ithas plain topology , just like its earlier models such as vgg 19 etc . Its architecture highly depends upon 3x3 kernels and ReLU. It has novel structural reparamaterization which decouple a training time of multi branch topology with a inference time plain architecture .You can also se training of REPVGG in google colab on CIFAR10 [here](https://github.com/imad08/model-zoo-submissions/blob/main/REPVGG/REPVGG_with_complete_reparamaterization_.ipynb) 

![fusing batch normalization and convolutions for reparametrization](https://media.arxiv-vanity.com/render-output/4507333/x1.png)

# Architecture of REPVGG

REPVGG heavily use 3x3 kernels and it has plain topology ,and it does not uses maxpool 2d the reason is author wants that the architecture has same kind of operators . In REPVGG we arrange 5 block architecture we can say that one stage , which uses 3x3 kernels and BatchNorm layers . In first layer of ech stage down the sample using the stride of (2,2). the first stage operates with large resolution hence in first stage block we just use one layer for lower latency . last stages has most channels.And most number of layers is in second last stage same as previous resnet architectures .

![main_architecture specifications](https://github.com/imad08/model-zoo/blob/master/classification/REPVGG_Pytorch/Assets/Screenshot%20(2863).png)
![REPVGG Architecture](https://media.arxiv-vanity.com/render-output/4507333/x3.png)


 
# Reparamateriztion is key in repvgg

The major difference that repvgg architecture has as compared to for RESNET etc , is the state of the art reparametrization . There are various kind of reparametrization removes batchnorm from Identity , Post addition of Batch Norm addition of ReLU in branches addition of 1X1 kernel . Most important reparametrization is fusing of kernel and BN in block. 


![fusing batch normalization and convolutions for reparametrization](https://pic3.zhimg.com/80/v2-686b26f8a41b54c10d76d7a90a6d8bbe_1440w.jpg)

# Results

The REPVGG model results before and after reparametrizations .


## before reparamaterization 

1hr 58min 43sec 80 epoch 
accuracy approx 86%

Total params: 15,681,066

Trainable params: 15,681,066

Non-trainable params: 0

Params size (MB): 59.82

Estimated Total Size (MB): 196.78

## After reparametrization 

1hr 0 min 33 se 

accuracy 84%

Total params: 7,041,194

Trainable params: 7,041,194 

Non Trainable params :0 

Params size (MB): 26.86

Estimated Total Size (MB): 54.82
