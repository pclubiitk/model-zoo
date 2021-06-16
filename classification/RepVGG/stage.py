from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import regularizers


class stage(layers.Layer):
    def __init__(self, filters, layer, bn=False):
        super(stage, self).__init__()
        self.dow3 = layers.Conv2D(
            filters,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_regularizer=regularizers.l2(1e-4),
        )
        self.dow1 = layers.Conv2D(
            filters, kernel_size=1, strides=2, kernel_regularizer=regularizers.l2(1e-4)
        )
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
        self.bn = layers.BatchNormalization() if bn else None
        self.re = layers.Activation(activations.relu)
        self.lay = layer

    def call(self, inp):
        x = self.dow3(inp)
        y = self.dow1(inp)
        x = x + y
        x = self.re(x)
        for i in range(self.lay - 1):
            y = self.con3(x)
            z = self.con1(x)
            x = x + y + z
            x = self.re(x)
        if self.bn is not None:
            x = self.bn(x)
        return x
