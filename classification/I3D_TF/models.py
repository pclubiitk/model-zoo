import tensorflow as tf
import silence_tensorflow.auto
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, BatchNormalization,Lambda, AveragePooling3D, MaxPooling3D, Dense, Input, GlobalAveragePooling3D,Reshape,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import numpy as np
from tensorflow.keras import backend as K
import tensorflow.python.keras.engine
from tensorflow.keras import layers
from tensorflow.keras.utils import get_source_inputs
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file

WEIGHTS_NAME = ['rgb_kinetics_only', 'flow_kinetics_only', 'rgb_imagenet_and_kinetics', 'flow_imagenet_and_kinetics']

#Path of the weights of the model when trained on the Kinetics-400 and the Imagenet Datasets.

WEIGHTS_PATH = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
}
WEIGHTS_PATH_NO_TOP = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
}

#Function caculates the input shape of the tensor to be fed to the model.
#The input params contains the FRAME_HEIIGHT, FRAME_WIDTH, NUMBER_OF_FRAMES and the NUM_OF_CHANNELS( for RGB ==3)
#and for Optical Flow the NUM_OF_FLOW_CHANNELS are 2.

def inputdim(input_shape, default_fsize, default_frames, weights = True):
  flatten = True
  if weights=="kinetics_only" or weights =="imagenet_and_kinetics" and len(input_shape)==4:
    dshape = (default_frames, default_fsize, default_fsize,3)
  else:
    if weights!="kinetics_only" and weights!="imagenet_and_kinetics" and input_shape and len(input_shape)==4:
      if input_shape[-1] not in {1,3}:
          dshape = (default_frames, default_fsize, default_fsize,input_shape[0])  
    
  if input_shape is not None:
      if len(input_shape)!=4:
        raise ValueError('The input tensor must consist of 4 entries representing the nuo of frames, and the threee dims')
      if input_shape[-1]!=3 and (weights == "kinetics_only" or weights == "imagnet_and_kinetics"):
        raise ValueError('The number of channels must be 3 for each entry')  
  else:
     if flatten==True:
      input_shape = dshape
  
  return input_shape    

#Funciton for applying a 3D Convolution Operation along with a BatchNormalization followed by a RELU Activation function.

def conv3d_bn(X,filters, num_frames, num_rows, num_cols, padding='same', strides=(1, 1, 1), use_bias = False):
    
    X = Conv3D(filters, (num_frames, num_rows, num_cols), strides=strides, padding=padding, use_bias=use_bias)(X)
    X = BatchNormalization(axis=4, scale=False)(X)
    X = Activation('relu')(X)
    return X
#Function for applying a 3D Convolution Operation without BatchNormalization and RELU Activation function.

def conv3d(X,filters, num_frames, num_rows, num_cols, padding='same', strides=(1, 1, 1)):
  X = Conv3D(filters, (num_frames, num_rows, num_cols),strides=strides,padding=padding)(X)
  return X 

#This function contains the entire architecture of the I3D model. It contains the Inception-V1 blocks followd by Maxpooling3D 
#operation. The inception block creates multiple branches from a particular layer including Pointwise Convolutions.
#Depthwise Convolutions , Standard Convolutions and then creates a stack of the layers and merges them.This enables the Inception
#model to act as state-of-the-art Image Classifiers with a reduced nbumber of params than normal Conbvolutional layers(similar to the
#VGG architecture.The last layer includes am average pooling layer then the pre-trained-weights are downloaded since it is impossible 
#to train the model using a single GPU also due to the huge size of the datasets and the depth of the model.

def Inception_Inflated3d(include_top=True, weights=None, input_tensor=None, input_shape=None, logits=True, classes=400):
    
    input_shape = inputdim(
        input_shape,
        default_fsize= 224, 
        default_frames= 79,
        weights= weights)

    if input_tensor is None:
        inpt = Input(shape=input_shape)

    X = conv3d_bn(inpt, 64, 7, 7, 7, strides=(2, 2, 2), padding='same')
    X = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same' )(X)
    X = conv3d_bn(X, 64, 1, 1, 1, strides=(1, 1, 1), padding='same')
    X = conv3d_bn(X, 192, 3, 3, 3, strides=(1, 1, 1), padding='same')
    X = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same')(X)

#Each block comprising of the inb0,inb1,inb2,inb3 defines an Inception-V1 module.

    inb0 = conv3d_bn(X, 64, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(X, 96, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(inb1, 128, 3, 3, 3, padding='same')
    inb2 = conv3d_bn(X, 16, 1, 1, 1, padding='same')
    inb2 = conv3d_bn(inb2, 32, 3, 3, 3, padding='same')
    inb3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(X)
    inb3 = conv3d_bn(inb3, 32, 1, 1, 1, padding='same')
    
    X = layers.concatenate([inb0, inb1, inb2, inb3],axis=4)

    inb0 = conv3d_bn(X, 128, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(X, 128, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(inb1, 192, 3, 3, 3, padding='same')
    inb2 = conv3d_bn(X, 32, 1, 1, 1, padding='same')
    inb2 = conv3d_bn(inb2, 96, 3, 3, 3, padding='same')
    inb3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(X)
    inb3 = conv3d_bn(inb3, 64, 1, 1, 1, padding='same')

    X = layers.concatenate([inb0, inb1, inb2, inb3],axis=4)
    
    X = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(X)
    inb0 = conv3d_bn(X, 192, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(X, 96, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(inb1, 208, 3, 3, 3, padding='same')
    inb2 = conv3d_bn(X, 16, 1, 1, 1, padding='same')
    inb2 = conv3d_bn(inb2, 48, 3, 3, 3, padding='same')
    inb3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(X)
    inb3 = conv3d_bn(inb3, 64, 1, 1, 1, padding='same')

    X = layers.concatenate([inb0, inb1, inb2, inb3],axis=4)

    inb0 = conv3d_bn(X, 160, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(X, 112, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(inb1, 224, 3, 3, 3, padding='same')
    inb2 = conv3d_bn(X, 24, 1, 1, 1, padding='same')
    inb2 = conv3d_bn(inb2, 64, 3, 3, 3, padding='same')
    inb3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(X)
    inb3 = conv3d_bn(inb3, 64, 1, 1, 1, padding='same')

    X = layers.concatenate([inb0, inb1, inb2, inb3],axis=4)

    inb0 = conv3d_bn(X, 128, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(X, 128, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(inb1, 256, 3, 3, 3, padding='same')
    inb2 = conv3d_bn(X, 24, 1, 1, 1, padding='same')
    inb2 = conv3d_bn(inb2, 64, 3, 3, 3, padding='same')
    inb3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(X)
    inb3 = conv3d_bn(inb3, 64, 1, 1, 1, padding='same')

    X = layers.concatenate([inb0, inb1, inb2, inb3],axis=4)

    inb0 = conv3d_bn(X, 112, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(X, 144, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(inb1, 288, 3, 3, 3, padding='same')
    inb2 = conv3d_bn(X, 32, 1, 1, 1, padding='same')
    inb2 = conv3d_bn(inb2, 64, 3, 3, 3, padding='same')
    inb3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(X)
    inb3 = conv3d_bn(inb3, 64, 1, 1, 1, padding='same')

    X = layers.concatenate([inb0, inb1, inb2, inb3],axis=4)

    inb0 = conv3d_bn(X, 256, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(X, 160, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(inb1, 320, 3, 3, 3, padding='same')
    inb2 = conv3d_bn(X, 32, 1, 1, 1, padding='same')
    inb2 = conv3d_bn(inb2, 128, 3, 3, 3, padding='same')
    inb3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(X)
    inb3 = conv3d_bn(inb3, 128, 1, 1, 1, padding='same')

    X = layers.concatenate([inb0, inb1, inb2, inb3],axis=4)
    
    X = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(X)
    inb0 = conv3d_bn(X, 256, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(X, 160, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(inb1, 320, 3, 3, 3, padding='same')
    inb2 = conv3d_bn(X, 32, 1, 1, 1, padding='same')
    inb2 = conv3d_bn(inb2, 128, 3, 3, 3, padding='same')
    inb3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(X)
    inb3 = conv3d_bn(inb3, 128, 1, 1, 1, padding='same')

    X = layers.concatenate([inb0, inb1, inb2, inb3],axis=4)

    inb0 = conv3d_bn(X, 384, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(X, 192, 1, 1, 1, padding='same')
    inb1 = conv3d_bn(inb1, 384, 3, 3, 3, padding='same')
    inb2 = conv3d_bn(X, 48, 1, 1, 1, padding='same')
    inb2 = conv3d_bn(inb2, 128, 3, 3, 3, padding='same')
    inb3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(X)
    inb3 = conv3d_bn(inb3, 128, 1, 1, 1, padding='same')

    X = layers.concatenate([inb0, inb1, inb2, inb3],axis=4)
    X = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid')(X)
    X = conv3d(X, classes, 1, 1, 1, padding='same')
    nfms = int(X.shape[1])
    X = Reshape((nfms, 400))(X)
    X = Lambda(lambda X: K.mean(X, axis=1, keepdims=False), output_shape=lambda out: (out[0], out[2]))(X)
    if not logits:
        X = Activation('softmax', name='prediction')(X)

#creating the model architecture.
    model = Model(inpt, X)

    if weights in WEIGHTS_NAME:
        if weights == WEIGHTS_NAME[0]:   
            if include_top:
                weights_url = WEIGHTS_PATH['rgb_kinetics_only']
                model_name = 'i3d_inception_rgb_kinetics_only.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['rgb_kinetics_only']
                model_name = 'i3d_inception_rgb_kinetics_only_no_top.h5'
        elif weights == WEIGHTS_NAME[1]: 
            if include_top:
                weights_url = WEIGHTS_PATH['flow_kinetics_only']
                model_name = 'i3d_inception_flow_kinetics_only.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['flow_kinetics_only']
                model_name = 'i3d_inception_flow_kinetics_only_no_top.h5'
        elif weights == WEIGHTS_NAME[2]: 
            if include_top:
                weights_url = WEIGHTS_PATH['rgb_imagenet_and_kinetics']
                model_name = 'i3d_inception_rgb_imagenet_and_kinetics.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['rgb_imagenet_and_kinetics']
                model_name = 'i3d_inception_rgb_imagenet_and_kinetics_no_top.h5'
        elif weights == WEIGHTS_NAME[3]: 
            if include_top:
                weights_url = WEIGHTS_PATH['flow_imagenet_and_kinetics']
                model_name = 'i3d_inception_flow_imagenet_and_kinetics.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['flow_imagenet_and_kinetics']
                model_name = 'i3d_inception_flow_imagenet_and_kinetics_no_top.h5'

#Downloading the weights file and loading the model pretrained-weights
        downloaded_weights_path = get_file(model_name, weights_url, cache_subdir='models')
        model.load_weights(downloaded_weights_path)

    return model
