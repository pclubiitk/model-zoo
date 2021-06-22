# Pytorch Implementation of UNet++ Model
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
  --deep_supervision DEEP_SUPERVISION
  --input_channels INPUT_CHANNELS
  --num_classes NUM_CLASSES
  --init_features INIT_FEATURES
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
* Zhou, Liang, et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
[arXiv:1807.10165]( https://arxiv.org/abs/1807.10165 )

### Contributed by:
* [Rishav Bikarwar](https://github.com/ris27hav)
<p>&nbsp;</p>

# Summary

## Introduction

The state-of-the-art models for image segmentation are variants of the encoderdecoder architecture like U-Net and fully convolutional network (FCN). These encoder-decoder networks used for segmentation share a key similarity:
**skip connections**, which combine deep, semantic, coarse-grained feature maps from the decoder sub-network with shallow, low-level, fine-grained feature maps from the encoder sub-network. The skip connections have proved effective in recovering fine-grained details of the target objects; generating segmentation masks with fine details even on complex background.

While a precise segmentation mask may not be critical in natural images, even marginal segmentation errors in medical images can lead to poor user experience in clinical settings. Furthermore, inaccurate segmentation may also lead to a major change in the subsequent computergenerated diagnosis. For example, an erroneous measurement of nodule growth in longitudinal studies can result in the assignment of an incorrect Lung-RADS category to a screening patient. It is therefore desired to devise more effective image segmentation architectures that can effectively recover the fine details of the target objects in medical images.

This paper has introduced UNet++, a new, more powerful architecture to address the need for more accurate segmentation in medical images. This architecture is essentially a deeply-supervised encoder-decoder network where the encoder and decoder sub-networks are connected through a series of nested, dense skip pathways. The re-designed skip pathways aim at reducing the semantic gap between the feature maps of the encoder and decoder sub-networks. The authors have argued that the optimizer would deal with an easier learning task when the feature maps from the decoder and encoder networks are semantically similar. Experiments demonstrate that UNet++ with deep supervision achieved significant performance gain over UNet and wide UNet.


---

## Network-Architecture

|![Architecture](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet++_Pytorch/assets/arch.jpeg?raw=true)|
|:--:|
|(Fig.1) UNet++ Architecture (*left*) and UNet Architecture (*right*).|

UNet++ starts with an encoder sub-network or backbone followed by a decoder sub-network. UNet++ have 3 addtions to original U-Net:
1. **Redesigned skip pathways** (shown in green): 
    * Skip connections used in U-Net directly connects the feature maps between encoder and decoder, which results in fusing semantically dissimilar feature maps. However, with UNet++, the output from the previous convolution layer of the same dense block is fused with the corresponding up-sampled output of the lower dense block. This brings the semantic level of the encoded feature closer to that of the feature maps waiting in the decoder; thus optimisation is easier when semantically similar feature maps are received. 
    * All convolutional layers on the skip pathway use kernels of size 3×3.

3. **Dense skip connections** (shown in blue): 
    * These Dense blocks are inspired by **DenseNet** with the purpose to improve segmentation accuracy and improves gradient flow.
    * Dense skip connections ensure that all prior feature maps are accumulated and arrive at the current node because of the dense convolution block along each skip pathway. This generates full resolution feature maps at multiple semantic levels.

3. **Deep supervision** (shown in red): 
    - This allows the model to be pruned to adjust the model complexity, to balance between speed (inference time) and performance.
    - In accurate mode, the output from all segmentation branch is averaged.
    - In fast mode, the final segmentation map is selected from one of the segmentation branches.
    - The authors conducted experiments to determine the best segmentation performance with different levels of pruning. The metrics used are Intersection over Union and inference time.

The authors have used a combination of binary cross-entropy and dice coefficient as the loss function which is described as:

![Loss Function](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet++_Pytorch/assets/loss_function.png?raw=true)

where *Y^b* and *Yb* denote the flatten predicted probabilities and the flatten ground truths of bth image respectively, and *N* indicates the batch size.

---
## Experimentation and Results

The authors used four medical imaging datasets for model evaluation, covering lesions/organs from different medical imaging modalities.

Table 1 compares U-Net, wide U-Net, and UNet++ in terms of the number parameters and segmentation accuracy for the tasks of lung nodule segmentation, colon polyp segmentation, liver segmentation, and cell nuclei segmentation. 

|![Results Table](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet++_Pytorch/assets/results1.png?raw=true)|
|:--:|
|(Table.1)  Segmentation results (IoU: %) for U-Net, wide U-Net and our suggested architecture UNet++ with and without deep supervision (DS).|

As seen, wide U-Net consistently outperforms U-Net except for liver segmentation where the two architectures perform comparably. This improvement is attributed to the larger number of parameters in wide U-Net. UNet++ without deep supervision achieves a significant performance gain over both UNet and wide U-Net, yielding average improvement of 2.8 and 3.3 points in
IoU. UNet++ with deep supervision exhibits average improvement of 0.6 points over UNet++ without deep supervision. Specifically, the use of deep supervision leads to marked improvement for liver and lung nodule segmentation, but
such improvement vanishes for cell nuclei and colon polyp segmentation.

---

|![Dataset](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet++_Pytorch/assets/data.png?raw=true)|
|:--:|
|(Fig.2) Few examples of *Brain MRI Segmentation Dataset*.|

I have used the Brain MRI Segmentation Dataset for model evaluation. Few example images and their masks are given in figure 2. There were a total of 3929 examples. I used 3600 examples for training and 329 for validation. Due to large number of parameters, 12 GB memory of GPU could process only a batch of 8 examples at a time. Due to small batch size, training for one epoch took around 15 minutes.  I trained the model for 15 epochs and the results are not good at all. It would need more epochs to produce something presentable. Though training for around 50 epochs can give really great results as can be seen from other submissions in Kaggle.

|![Results Table 2](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet++_Pytorch/assets/results2.png?raw=true)|
|:--:|
|(Fig.3)  Training Result on Brain MRI Segmentation dataset for 15 epochs.|

---

## Conclusion

The proposed UNet++ network does gives a significant performance improvement over UNet and wide UNet. The suggested architecture takes advantage of re-designed skip pathways and deep supervision. The evaluations of UNet++ on four medical imaging datasets covering lung nodule segmentation, colon polyp segmentation, cell nuclei segmentation, and liver segmentation demonstrated that UNet++ with deep supervision achieved an average IoU gain of 3.9 and 3.4 points over U-Net and wide U-Net, respectively.

Deeper models usually do perform better, but it is clear that the dense architecture of UNet++ provides a significant advantage over the otherwise mundane UNet architecture. There is clear merit to the idea that dense connections will bridge the semantic gap between the encoder and decoder portions of the network.

I could not train the model on Brain MRI Segmentation dataset for more epochs due to time and memory constraints. But still got a promising start. Given below are the results that UNet++ can provide for enough epochs (taken from the submissions by others in Kaggle) :

|![Visualising Predictions](https://github.com/ris27hav/model-zoo/blob/master/semantic_segmentation/UNet++_Pytorch/assets/visualisation.png?raw=true)|
|:--:|
|(Fig.4)  Visualising the predictions by UNet++|


While peeking at other submissions, I observed that every model struggled to model over masks that depicted tiny tumours, often producing noise throughout the rest of the mask. This problem seems to become more evident as the tumour size grows smaller. Perhaps this is because the model's feature maps are not optimized to detect smaller tumour sizes, and the dataset does not contain a significant proportion of these samples. 

Further work on this could consist of hyperparameter optimization, experimentation with dataset augmentation, comparison with transfer learning models, and using some pre-trained backbone network as the encoder portion of the segmentation model.
