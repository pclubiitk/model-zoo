import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.regularizers import L2

import hyperparams as hprms
import blocks


def build_model(input_shape):

    input_layer = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=16, kernel_size=3, strides=(1,1), padding="same",
    kernel_initializer = GlorotNormal(), kernel_regularizer=L2(hprms.weight_decay)
    )(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(rate = 0.4)(x)

    for i in range(18):
      x = blocks.ResBlockIdentityShortcut(16)(x)
    x = layers.Dropout(rate = 0.4)(x)

    x = blocks.ResBlockProjectionShortcut(32)(x)
    for i in range(17):
      x = blocks.ResBlockIdentityShortcut(32)(x)
    x = layers.Dropout(rate = 0.4)(x)


    x = blocks.ResBlockProjectionShortcut(64)(x)
    for i in range(17):
      x = blocks.ResBlockIdentityShortcut(64)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    output = layers.Dense(10, activation='softmax', kernel_initializer=GlorotNormal(), 
    kernel_regularizer=L2(hprms.weight_decay)
    )(x)


    model = keras.Model(inputs=input_layer, outputs=[output])

    return model


# class ResNet53(keras.Model):

#   def __init__(self):
#       super(ResNet53, self).__init__()

#       self.conv = layers.Conv2D(filters=16, kernel_size=3, strides=(1,1), padding="same",
#       kernel_initializer = GlorotNormal(), kernel_regularizer=L2(hprms.weight_decay)
#       )
#       self.bn = layers.BatchNormalization()
#       self.relu = layers.Activation("relu")
#       self.dropout = layers.Dropout(rate = 0.4)

#       self.resblock_identity16 = blocks.ResBlockIdentityShortcut(16)
#       self.resblock_identity32 = blocks.ResBlockIdentityShortcut(32)
#       self.resblock_identity64 = blocks.ResBlockIdentityShortcut(64)
#       self.resbock_projection32 = blocks.ResBlockProjectionShortcut(32)
#       self.resbock_projection64 = blocks.ResBlockProjectionShortcut(64)

#       self.pool = layers.GlobalAveragePooling2D()
#       self.flatten = layers.Flatten()
#       self.dense = layers.Dense(10, activation='softmax', kernel_initializer=GlorotNormal(), 
#       kernel_regularizer=L2(hprms.weight_decay)
#       )

#   def call(self, input_tensor, training=False):

#     # x = layers.Conv2D(filters=16, kernel_size=3, strides=(1,1), padding="same",
#     # kernel_initializer = GlorotNormal(), kernel_regularizer=L2(hprms.weight_decay)
#     # )(input_tensor, training=training)
#     # x = layers.BatchNormalization()(x)
#     # x = layers.Activation("relu")(x)
#     # x = layers.Dropout(rate = 0.4)(x)
#     x = self.conv(input_tensor, training=training)
#     x = self.bn(x, training=training)
#     x = self.relu(x, training=training)
#     x = self.dropout(x, training=training)

#     for i in range(18):
#       # x = blocks.ResBlockIdentityShortcut(16)(x, training=training)
#       x = self.resblock_identity16(x, training=training)
#     # x = layers.Dropout(rate = 0.4)(x)
#     x = self.dropout(x, training=training)

#     # x = blocks.ResBlockProjectionShortcut(32)(x, training=training)
#     x = self.resbock_projection32(x, training=training)
#     for i in range(17):
#       # x = blocks.ResBlockIdentityShortcut(32)(x, training=training)
#       x = self.resblock_identity32(x, training=training)
#     # x = layers.Dropout(rate = 0.4)(x)
#     x = self.dropout(x, training=training)

#     # x = blocks.ResBlockProjectionShortcut(64)(x, training=training)
#     x = self.resbock_projection64(x, training=training)
#     for i in range(17):
#       # x = blocks.ResBlockIdentityShortcut(64)(x, training=training)
#       x = self.resblock_identity64(x, training=training)

#     # x = layers.GlobalAveragePooling2D()(x, training=training)
#     x = self.pool(x, training=training)
#     # x = layers.Flatten()(x, training=training)
#     x = self.flatten(x, training=training)
#     # x = layers.Dense(10, activation='softmax', kernel_initializer=GlorotNormal(), 
#     # kernel_regularizer=L2(hprms.weight_decay)
#     # )(x, training=training)
#     x = self.dense(x, training=training)

#     return x


#   def model(self):

#     x = keras.Input(shape=(32, 32, 3))
#     return keras.Model(inputs=[x], outputs=[self.call(10, x)])