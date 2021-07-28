from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.regularizers import L2

import hyperparams as hprms


class ResBlockIdentityShortcut(layers.Layer):

  def __init__(self, filters):

    super(ResBlockIdentityShortcut, self).__init__()

    self.conv1 = layers.Conv2D (filters, kernel_size=3, strides=(1,1), padding="same",
    kernel_initializer = GlorotNormal(), kernel_regularizer=L2(hprms.weight_decay)
    )
    self.conv2 = layers.Conv2D (filters, kernel_size=3, strides=(1,1), padding="same",
    kernel_initializer = GlorotNormal(), kernel_regularizer=L2(hprms.weight_decay)
    )
    self.bn1 = layers.BatchNormalization()
    self.bn2 = layers.BatchNormalization()
    self.relu = layers.Activation("relu")


  def call(self, input_tensor, training=False):

    x_shortcut = input_tensor
    x = self.conv1(input_tensor)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)

    x = x + x+x_shortcut
    
    x = self.relu(x)

    return x



class ResBlockProjectionShortcut(layers.Layer):

  def __init__(self, filters):

    super(ResBlockProjectionShortcut, self).__init__()

    self.zeropad = layers.ZeroPadding2D(padding=(1, 1))
    self.conv_with_downsample =  layers.Conv2D (filters, kernel_size=3, strides=(2,2), padding="valid",
    kernel_initializer = GlorotNormal(), kernel_regularizer=L2(hprms.weight_decay)
    )
    self.conv_with_downsample_shortcut =  layers.Conv2D (filters, kernel_size=3, strides=(2,2), padding="valid",
    kernel_initializer = GlorotNormal(), kernel_regularizer=L2(hprms.weight_decay)
    )
    self.bn1 = layers.BatchNormalization()
    self.bn2 = layers.BatchNormalization()
    self.bn3 = layers.BatchNormalization()
    self.conv = layers.Conv2D (filters, kernel_size=3, strides=(1,1), padding="same",
    kernel_initializer = GlorotNormal(), kernel_regularizer=L2(hprms.weight_decay)
    )
    self.relu = layers.Activation("relu")


  def call(self, input_tensor, training=False):

    x_shortcut = self.zeropad(input_tensor)
    x_shortcut= self.conv_with_downsample_shortcut(x_shortcut)
    x_shortcut = self.bn1(x_shortcut)

    x = self.zeropad(input_tensor)
    x = self.conv_with_downsample(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.conv(x)
    x = self.bn3(x)

    x = x+x_shortcut

    x = self.relu(x)

    return x