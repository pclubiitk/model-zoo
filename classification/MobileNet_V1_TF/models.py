import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    DepthwiseConv2D,
    Conv2D,
    Activation,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam


# ? This function is defined for the standard convolutions operations applied on the images
# ? A standard convolution operation basically applies a kernel of dimensions Dk x Dk x M x N to the corresponding feature map  of dimensions
# ? Df x Df x M to reduce it to a output feature map of dimensions Df x Df x N, here M denotes the number of input channels of the input feature map and
# ? N denotes the number of output channels of the feature map obtained after each standard convolution operation. The overall computations
# ? in a standard convolution is Dk x Dk x Df x Df x M x N since in each convolution operation Dk x Dk elemenst of the kernel are multiplied to each
# ? element of the feature map and there are total of M x N filters.

# ? The given function return the feature map tensor after applying the operation ollowed by batchnormalisation and RELU activation to
# ? introduce non-linearity.


def standard_conv(X, filter, k_size, stride, width_mul):
    filter = filter * width_mul
    X = Conv2D(filters=filter, kernel_size=k_size, padding="same", strides=stride)(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    return X


# ? This function is defined for the depthwise convolutions applied in the mobilenet network which drastically reduces the computational costs.
# ? in depthwise convoltuions,


def depthwise_conv(X, filter, k_size, stride, width_mul):
    filter = filter * width_mul
    X = DepthwiseConv2D(kernel_size=k_size, strides=stride, padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Conv2D(filters=filter, kernel_size=(1, 1), padding="same", strides=(1, 1))(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    return X


# This function creates the Mobilenet_v1 model architecture which comprises of depthwise seperable convolutions and standard convolution at the beginning.


def Mobilenet_v1(X):

    # X_input = Input(shape = shape)
    X = standard_conv(X, 32, (3, 3), (2, 2), 1)
    X = depthwise_conv(X, 64, (3, 3), (1, 1), 1)
    X = depthwise_conv(X, 128, (3, 3), (2, 2), 1)
    X = depthwise_conv(X, 128, (3, 3), (1, 1), 1)
    X = depthwise_conv(X, 256, (3, 3), (2, 2), 1)
    X = depthwise_conv(X, 256, (3, 3), (1, 1), 1)
    X = depthwise_conv(X, 512, (3, 3), (2, 2), 1)

    for i in range(5):
        X = depthwise_conv(X, 512, (3, 3), (1, 1), 1)

    X = depthwise_conv(X, 1024, (3, 3), (2, 2), 1)
    X = depthwise_conv(X, 1024, (3, 3), (1, 1), 1)
    X = GlobalAveragePooling2D()(X)
    X = Dense(units=10, activation="softmax")(X)

    # input_tensor = Input(shape = (224,224,3))
    # model = Model(inputs = X_input, outputs = X)
    return X


# uncomment this if you want to see the model params and the architecture.
# model.summary()
