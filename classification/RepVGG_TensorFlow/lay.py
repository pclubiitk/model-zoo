import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import regularizers


class lay(layers.Layer):
    def __init__(self, filters, is_first=True, inference=False):
        """
        filters:= The number of channels the convolution layers must have

        If it is the first layer, then the convolution layers are used for
        downsampling and there is no identity shortcut.
        """

        super(lay, self).__init__()
        self.is_first = is_first
        self.filters = filters
        if self.is_first:
            self.strides = 2
        else:
            self.strides = 1
            self.bn0 = layers.BatchNormalization()

        if inference:
            self.inf = layers.Conv2D(
                self.filters, kernel_size=3, strides=self.strides, padding="same"
            )

        self.con3 = layers.Conv2D(
            filters, kernel_size=3, strides=self.strides, padding="same", use_bias=False
        )
        self.con1 = layers.Conv2D(
            filters, kernel_size=1, strides=self.strides, padding="same", use_bias=False
        )
        self.bn3 = layers.BatchNormalization()
        self.bn1 = layers.BatchNormalization()
        self.re = layers.Activation(activations.relu)

    def call(self, inp):
        if hasattr(self, "inf"):
            return self.re(self.inf(inp))
        x = self.con3(inp)
        x = self.bn3(x)
        y = self.con1(inp)
        y = self.bn1(y)
        if hasattr(self, "bn0"):
            z = self.bn0(inp)
            x = x + y + z
        else:
            x = x + y
        x = self.re(x)
        return x

    def parameters(self):
        """
        This function is for re-parameterization, it extracts
        the weights from the convolution layers and returns them.
        """

        g3, b3, m3, v3 = (
            self.bn3.gamma,
            self.bn3.beta,
            self.bn3.moving_mean,
            self.bn3.moving_variance,
        )
        e3 = self.bn3.epsilon
        s3 = (v3 + e3) ** 0.5
        k3 = self.con3.weights[0]
        w3 = k3 * (g3 / s3)
        b3 = b3 - m3 * (g3 / s3)

        g1, b1, m1, v1 = (
            self.bn1.gamma,
            self.bn1.beta,
            self.bn1.moving_mean,
            self.bn1.moving_variance,
        )
        e1 = self.bn1.epsilon
        s1 = (v1 + e1) ** 0.5
        k1 = self.con1.weights[0]
        paddings = tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]])
        w1 = k1 * (g1 / s1)
        w1 = tf.pad(w1, paddings)
        b1 = b1 - m1 * (g1 / s1)

        if not self.is_first:
            g0, b0, m0, v0 = (
                self.bn0.gamma,
                self.bn0.beta,
                self.bn0.moving_mean,
                self.bn0.moving_variance,
            )
            e0 = self.bn0.epsilon
            s0 = (v0 + e0) ** 0.5
            k0 = np.zeros(k3.shape, dtype="float32")
            k0[1, 1, :, :] = 1.0
            k0 = tf.convert_to_tensor(k0, dtype="float32")
            w0 = k0 * (g0 / s0)
            b0 = b0 - m0 * (g0 / s0)

            w = w3 + w1 + w0
            b = b3 + b1 + b0
        else:
            w = w3 + w1
            b = b3 + b1

        return w, b

    def repara(self):
        self.inf = layers.Conv2D(
            self.filters,
            kernel_size=3,
            strides=self.strides,
            padding="same",
            weights=self.parameters(),
        )
        delattr(self, "con3")
        delattr(self, "con1")
        delattr(self, "bn3")
        delattr(self, "bn1")
        if hasattr(self, "bn0"):
            delattr(self, "bn0")
        return
