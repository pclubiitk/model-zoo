import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import regularizers


class lay(layers.Layer):
    def __init__(self, filters, is_first=True):
        super(lay, self).__init__()
        """
        filters:= The number of channels the convolution layers must have

        If it is the first layer, then the convolution layers are used for
        downsampling and there is no identity shortcut.
        """

        self.is_first = is_first
        if self.is_first:
            self.con3 = layers.Conv2D(
                filters,
                kernel_size=3,
                strides=2,
                padding="same",
                kernel_regularizer=regularizers.l2(1e-4),
            )
            self.con1 = layers.Conv2D(
                filters,
                kernel_size=1,
                strides=2,
                padding="same",
                kernel_regularizer=regularizers.l2(1e-4),
            )
        else:
            self.con3 = layers.Conv2D(
                filters,
                kernel_size=3,
                padding="same",
                kernel_regularizer=regularizers.l2(1e-4),
            )
            self.con1 = layers.Conv2D(
                filters,
                kernel_size=1,
                padding="same",
                kernel_regularizer=regularizers.l2(1e-4),
            )
        self.bn3 = layers.BatchNormalization()
        self.bn1 = layers.BatchNormalization()
        self.bn0 = layers.BatchNormalization()
        self.re = layers.Activation(activations.relu)

    def call(self, inp):
        x = self.con3(inp)
        x = self.bn3(x)
        y = self.con1(inp)
        y = self.bn1(y)
        x = x + y
        if self.is_first:
            x = x + y
        else:
            z = self.bn0(inp)
            x = x + y + z
        x = self.re(x)
        return x

    def parameters(self):
        """
        This function is for re-parameterization, it extracts
        the weights from the convolution layers and returns them.
        """

        g3, b3, m3, v3 = self.bn3.get_weights()
        s3 = (v3) ** 0.5
        k3 = self.con3.get_weights()[0]
        w3 = k3 * (g3 / s3)
        b3 = b3 - m3 * (g3 / s3)

        g1, b1, m1, v1 = self.bn1.get_weights()
        s1 = (v1) ** 0.5
        k1 = self.con1.get_weights()[0]
        paddings = tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]])
        w1 = k1 * (g1 / s1)
        w1 = tf.pad(w1, paddings)
        b1 = b1 - m1 * (g1 / s1)

        if not self.is_first:
            g0, b0, m0, v0 = self.bn0.get_weights()
            s0 = (v0) ** 0.5
            k0 = k1 * 0
            k0 = k0 + 1
            w0 = k0 * (g0 / s0)
            w0 = tf.pad(w0, paddings)
            b0 = b0 - m0 * (g0 / s0)

            w = w3 + w1 + w0
            b = b3 + b1 + b0
            return w, b

        else:
            w = w3 + w1
            b = b3 + b1
            return w, b
