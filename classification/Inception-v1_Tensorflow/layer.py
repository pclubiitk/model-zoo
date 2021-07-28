import tensorflow as tf
from tensorflow.keras import layers


class reduce(layers.Layer):
    def __init__(self, filter1x1, ker_size, filters):
        """
        Class to combine the 1x1 conv layers used for reduction
        with the corresponsing 3x3 or 5x5 conv layers.

        filter1x1:= Number of filters in the 1x1 convolutional layer
        ker_size:= The size of the corresponding convolutional layer
        filters:= Number of filters in the convolutional layer
        """
        super(reduce, self).__init__()
        self.con1 = layers.Conv2D(
            filter1x1, kernel_size=1, padding="same", activation="relu"
        )
        self.conv = layers.Conv2D(
            filters, kernel_size=ker_size, padding="same", activation="relu"
        )

    def call(self, inp):
        x = self.con1(inp)
        x = self.conv(x)
        return x


class poolproj(layers.Layer):
    def __init__(self, filter1x1):
        """
        Class to combine the Max Pooling layer with
        the 1x1 conv layer for pool projecting.
        """
        super(poolproj, self).__init__()
        self.max = layers.MaxPooling2D(pool_size=3, strides=1, padding="same")
        self.conv = layers.Conv2D(
            filter1x1, kernel_size=1, padding="same", activation="relu"
        )

    def call(self, inp):
        x = self.max(inp)
        x = self.conv(x)
        return x


class inceptionblock(layers.Layer):
    def __init__(self, filter1x1, red3, red5, pool):
        super(inceptionblock, self).__init__()
        """
        Class for an Incpetion Module.

        filter1x1:= Number of filters in the 1x1 conv layer
        red3:= A list corresponding to the parameters required for the 3x3 conv layer
        red5:= A list corresponding to the parameters required for the 5x5 conv layer
        pool:= Number of filters for 1x1 layer that follows the Pooling layer
        """
        self.conv1 = layers.Conv2D(
            filter1x1, kernel_size=1, padding="same", activation="relu"
        )
        self.conv3 = reduce(red3[0], red3[1], red3[2])
        self.conv5 = reduce(red5[0], red5[1], red5[2])
        self.poolp = poolproj(pool)

    def call(self, inp):
        x1 = self.conv1(inp)
        x2 = self.conv3(inp)
        x3 = self.conv5(inp)
        x4 = self.poolp(inp)
        return tf.concat([x1, x2, x3, x4], 3)


class auxiliary(layers.Layer):
    def __init__(self, channels=10):
        super(auxiliary, self).__init__()
        """
        Class for auxiliary classification.
        """
        self.avg = layers.AveragePooling2D(pool_size=5, strides=3)
        self.con1 = layers.Conv2D(128, kernel_size=1, padding="same", activation="relu")
        self.flat = layers.Flatten()
        self.full1 = layers.Dense(1024, activation="relu")
        self.drop = layers.Dropout(0.7)
        self.full2 = layers.Dense(channels, activation="softmax")

    def call(self, inp):
        x = self.avg(inp)
        x = self.con1(x)
        x = self.flat(x)
        x = self.full1(x)
        x = self.drop(x)
        x = self.full2(x)
        return x
