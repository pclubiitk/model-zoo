# T3D ConvNets forVideo Classification

## Usage
### Train & Test
```bash
$ python3 main.py  --model "T3D_121" 
```

## References
* [Original T3D paper](https://arxiv.org/pdf/1711.08200.pdf)

## Contributed by:
* [Vansh Bansal](https://github.com/vanshbansal1505/)

# Summary

##Introduction:
The model introduces a new temporal layer that models variable temporal convolution kernel depths. The model embeds this new temporal layer in the proposed 3D CNN. The model extends
the DenseNet architecture - which normally is 2D - with 3D
filters and pooling kernels. They name this proposed video
convolutional network ‘Temporal 3D ConvNet’ (T3D) and
its new temporal layer ‘Temporal Transition Layer’ (TTL).

## Model:

<img src="https://github.com/vanshbansal1505/model-zoo/blob/master/Video_Classification/T3D_tensorflow/assets/model.png?raw=true" alt="drawing">



Temporal 3D ConvNet (T3D). The Temporal Transition Layer (TTL) is applied to the DenseNet3D. T3D uses
video clips as input. The 3D feature-maps from the clips are densely propagated throughout the network. The TTL operates
on the different temporal depths, thus allowing the model to capture the appearance and temporal information from the short,
mid, and long-range terms. The output of the network is a video-level prediction

## Model Architecture:

We train our T3D from scratch
on UCF101. Our T3D operates on a stack of 32 RGB
frames. We resize the video to  224 × 224. For the network training, we use SGD, Nesterov momentum of 0.9, weight decay
of 10e−4
and batch size of 64. The initial learning rate is
set to 0.1

<img src ="https://github.com/vanshbansal1505/model-zoo/blob/master/Video_Classification/T3D_tensorflow/assets/architecture.png?raw=true">


All the proposed architectures incorporate 3D filters and pooling
kernels. Each “conv” layer shown in the table corresponds the composite sequence BN-ReLU-Conv operations. The bold
numbers shown in the TTL layer, denotes to the variable temporal convolution kernel depths applied to the 3D feature-maps.


## Comparisons:

Comparison with state of art methods:

<img src="https://github.com/vanshbansal1505/model-zoo/blob/master/Video_Classification/T3D_tensorflow/assets/comparison.png?raw=true" alt="drawing">

## Conclusion:
The model clearly shows the benefit of exploiting temporal depths over shorter and longer time ranges
over fixed 3D homogeneous kernel depth architectures. In
this work, we also extend the DenseNet architecture with
3D convolutions, we name our architecture as ‘Temporal
3D ConvNets’ (T3D).
