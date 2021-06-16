# SSD: Single Shot MultiBox Object Detector in PyTorch (VOC 2007 Dataset)
![1_tDKlbaeHGSm_TwN02jLB8w](https://user-images.githubusercontent.com/63955480/122269885-55692780-cefb-11eb-8431-57c5ef1fa375.jpeg)

# Summary
## Intro
- SSD is designed for object detection in real-time and it also eliminate the need for the region proposal network. To recover the drop in accuracy, SSD applies a few improvements including multi-scale features and default boxes which allow SSD to match the Faster R-CNN’s accuracy using lower resolution images, which further pushes the speed higher. 
###  *Comparison 
<img src= "https://github.com/Shrayansh19/SSD-Model-Zoo/blob/main/doc/1_rqGEyJKbKv3ecmjaMSiEtA.png">

To improve accuracy, SSD introduces:
- small convolutional filters to predict object classes and offsets to default boundary boxes.
- separate filters for default boxes to handle the difference in aspect ratios.
- multi-scale feature maps for object detection.

### The principle of the ssd model is :

- Multi-scale feature maps for detection
- After the basic network structure, additional convolutional layers are added. The size of these convolutional layers is decreasing layer by layer and can be predicted at multiple scales.
- Convolutional predictors for detection and added feature layers can use a series of convolution kernels to generate a series of fixed-size prediction results, as shown in the following figure. For a m × n feature layer with p channels.
- At each m × n feature map location, using the 3 × 3 kernel will produce an output value. The location offset value of the predicted bounding box is the relative distance between the output default bounding box position and the feature map location at this time
![7da6066b446d826f3ad05e4be56f5883](https://user-images.githubusercontent.com/63955480/122278693-cfea7500-cf04-11eb-9bc4-0bb5408b6c47.png)

### SSD model:
![ssd_architecture](https://user-images.githubusercontent.com/63955480/122277363-6fa70380-cf03-11eb-999c-1016ef1c9fa2.png)

### Key Observations:
- Accuracy increases with the number of default boundary boxes at the cost of speed.
- Multi-scale feature maps improve the detection of objects at a different scale.
- Better default boundary boxes will help accuracy.
- SSD has lower localization error comparing with R-CNN but more classification error dealing with similar categories. The higher classification errors are likely because we use the same boundary box to make multiple class predictions.


## Evaluation
- I have trained the model for 100 epochs 

Below are Avereage Precision values for all classes:-

 | Class | Average Precision |
 | :-----: | :------: |
 | _aeroplane_ | 0.7508596181869507 |
 | _bicycle_ | 0.8064944744110107 |  
 | _bird_ | 0.7118245959281921 |     
 | _boat_ | 0.6522005200386047 |     
 | _bottle_ | 0.4388348162174225 |   
 | _bus_ | 0.8313817381858826 |      
 | _car_ | 0.8420098423957825 |      
 | _cat_ | 0.8460454344749451 |
 | _chair_ | 0.4887792468070984 |
 | _cow_ | 0.7463425993919373 |
 | _diningtable_ | 0.7300090789794922 |
 | _dog_ | 0.7909976243972778 |
 | _horse_ | 0.8242510557174683 |
 | _motorbike_ | 0.804941713809967 |
 | _person_ | 0.7624345421791077 |
 | _pottedplant_ | 0.4471535384654999 |
 | _sheep_ | 0.7205353379249573 |
 | _sofa_ | 0.7502992749214172 |
 | _train_ | 0.8156716823577881 |
 | _tvmonitor_ | 0.6981692910194397 |

## Performance

#### VOC2007 Test

##### mAP

| Original | Converted weiliu89 weights | From scratch w/o data aug | From scratch w/ data aug |
|:-:|:-:|:-:|:-:|
| 77.2 % | 77.26 % | 58.12% | 77.43 % |


## Contributor

* [**Shrayansh Jakar**](https://github.com/Shrayansh19)

## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [Cornell Paper](https://arxiv.org/abs/1512.02325).
- COCO: Common Objects in Context. http://mscoco.org/dataset/
#detections-leaderboard (2016) [Online; accessed 25-July-2016].
- He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR.
(2016)
