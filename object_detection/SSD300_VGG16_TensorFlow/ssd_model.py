from os import name
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Concatenate, Reshape, Activation
import numpy as np
from tensorflow.python.keras.backend import shape
from utils.ssd_utils import get_pred_4, get_pred_6

# The SSD model Architecture
def ssd_model():
    
    #input shape is assumed to be 300x300x3
    input = Input(shape=[300, 300, 3])

    scale = np.linspace(0.2, 0.9, 6)

    #Conv1
    conv1_1 = Conv2D(64, (3,3), padding='same', name='Conv1_1')(input)
    conv1_2 = Conv2D(64, (3,3), padding='same', name='Conv1_2')(conv1_1)

    #Pool1
    pool_1 = MaxPool2D(strides=(2, 2), padding='same', name='Pool1')(conv1_2)

    #Conv2
    conv2_1 = Conv2D(128, (3,3), padding='same', name='Conv2_1')(pool_1)
    conv2_2 = Conv2D(128, (3,3), padding='same', name='Conv2_2')(conv2_1)

    #Pool2
    pool_2 = MaxPool2D(strides=(2, 2), padding='same', name='Pool2')(conv2_2)

    #Conv3
    conv3_1 = Conv2D(256, (3,3), padding='same', name='Conv3_1')(pool_2)
    conv3_2 = Conv2D(256, (3,3), padding='same', name='Conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3,3), padding='same', name='Conv3_3')(conv3_2)

    #Pool3
    pool_3 = MaxPool2D(strides=(2, 2), padding='same', name='Pool3')(conv3_3)

    #Conv4
    conv4_1 = Conv2D(512, (3,3), padding='same', name='Conv4_1')(pool_3)
    conv4_2 = Conv2D(512, (3,3), padding='same', name='Conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3,3), padding='same', name='Conv4_3')(conv4_2)

    # In SSD paper, conv4_3 is required to be normalised

    '''Have a look here'''
    # conv4_3_norm = L2Normalization(gamma_init=20, name="Conv4_3_norm")(conv4_3)
    conv4_3_norm = conv4_3

    #Pool4
    pool_4 = MaxPool2D(strides=(2, 2), padding='same', name='Pool4')(conv4_3)

    #Conv5
    conv5_1 = Conv2D(512, (3,3), padding='same', name='Conv5_1')(pool_4)
    conv5_2 = Conv2D(512, (3,3), padding='same', name='Conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3,3), padding='same', name='Conv5_3')(conv5_2)

    ''' This was the base network (VGG_16), further now we will add layers to extract various feature maps'''
    pool_5 = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same",name="Pool5")(conv5_3)

    fc_6 = Conv2D(1024, (3,3), padding='same', name='fc6', dilation_rate=(6,6))(pool_5)
    fc_7 = Conv2D(1024, (1,1), padding='same', name='fc7')(fc_6)

    conv8_1 = Conv2D(256, (1,1), padding='same', name='Conv8_1')(fc_7)
    conv8_2 = Conv2D(512, (3,3),strides=(2, 2), padding='same', name='Conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1,1), padding='same', name='Conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3,3),strides=(2, 2), padding='same', name='Conv9_2')(conv9_1)

    conv10_1 = Conv2D(128, (1,1), padding='valid', name='Conv10_1')(conv9_2)
    conv10_2 = Conv2D(256, (3,3), padding='valid', name='Conv10_2')(conv10_1)

    conv11_1 = Conv2D(128, (1,1), padding='valid', name='Conv11_1')(conv10_2)
    conv11_2 = Conv2D(256, (3,3), padding='valid', name='Conv11_2')(conv11_1)


    # To get the scales for different feature maps(here 6)
    # Min scale is 0.2, and max scale is 0.9

    conf_layers = []
    loc_layers = []
    def_layers = []

    conv4_3_norm_conf, conv4_3_norm_loc, conv4_3_norm_def = get_pred_4(conv4_3_norm, scale[0], scale[1])
    conf_layers.append(conv4_3_norm_conf)
    loc_layers.append(conv4_3_norm_loc)
    def_layers.append(conv4_3_norm_def)

    fc_7_conf, fc_7_loc, fc_7_def = get_pred_6(fc_7, scale[1], scale[2])
    conf_layers.append(fc_7_conf)
    loc_layers.append(fc_7_loc)
    def_layers.append(fc_7_def)

    conv8_2_conf, conv8_2_loc, conv8_2_def = get_pred_6(conv8_2, scale[2], scale[3])
    conf_layers.append(conv8_2_conf)
    loc_layers.append(conv8_2_loc)
    def_layers.append(conv8_2_def)

    conv9_2_conf, conv9_2_loc, conv9_2_def = get_pred_6(conv9_2, scale[3], scale[4])
    conf_layers.append(conv9_2_conf)
    loc_layers.append(conv9_2_loc)
    def_layers.append(conv9_2_def)

    conv10_2_conf, conv10_2_loc, conv10_2_def = get_pred_4(conv10_2, scale[4], scale[5])
    conf_layers.append(conv10_2_conf)
    loc_layers.append(conv10_2_loc)
    def_layers.append(conv10_2_def)

    conv11_2_conf, conv11_2_loc, conv11_2_def = get_pred_4(conv11_2, scale[5], 1.0)
    conf_layers.append(conv11_2_conf)
    loc_layers.append(conv11_2_loc)
    def_layers.append(conv11_2_def)

    conf = Concatenate(axis=-2)(conf_layers)
    conf_act = Activation('softmax')(conf)
    loc = Concatenate(axis=-2)(loc_layers)
    defau = Concatenate(axis=-2)(def_layers)

    predictions = Concatenate(axis=-1, name='Predictions')([conf_act, loc, defau])

    return Model(inputs=input, outputs=predictions)