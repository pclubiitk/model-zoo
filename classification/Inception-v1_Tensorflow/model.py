import tensorflow as tf
from tensorflow.keras import layers
from layer import reduce, inceptionblock, auxiliary


class inceptionv1(tf.keras.Model):
    def __init__(self, channels=10):
        super(inceptionv1, self).__init__()

        self.con7 = layers.Conv2D(
            64, kernel_size=7, strides=2, padding="same", activation="relu"
        )
        self.max3_2 = layers.MaxPooling2D(3, strides=2)
        self.lrn1 = layers.Lambda(tf.nn.local_response_normalization)
        self.con3 = reduce(64, 3, 192)
        self.lrn2 = layers.Lambda(tf.nn.local_response_normalization)
        self.in3a = inceptionblock(64, [96, 3, 128], [16, 5, 32], 32)
        self.in3b = inceptionblock(128, [128, 3, 192], [32, 5, 96], 64)

        self.in4a = inceptionblock(192, [96, 3, 208], [16, 5, 48], 64)
        self.aux1 = auxiliary(channels)
        self.in4b = inceptionblock(160, [112, 3, 224], [24, 5, 64], 64)
        self.in4c = inceptionblock(128, [128, 3, 256], [24, 5, 64], 64)
        self.in4d = inceptionblock(112, [144, 3, 288], [32, 5, 64], 64)
        self.aux2 = auxiliary(channels)
        self.in4e = inceptionblock(256, [160, 3, 320], [32, 5, 128], 128)

        self.in5a = inceptionblock(256, [160, 3, 320], [32, 5, 128], 128)
        self.in5b = inceptionblock(384, [192, 3, 384], [48, 5, 128], 128)
        self.avgp = layers.AveragePooling2D(pool_size=7, strides=1)
        self.drop = layers.Dropout(0.4)
        self.flat = layers.Flatten()
        self.full = layers.Dense(channels, activation="softmax")

    def call(self, inp):
        x = self.con7(inp)
        x = self.max3_2(x)
        x = self.lrn1(x)
        x = self.con3(x)
        x = self.lrn2(x)
        # x = self.max3_2(x)
        x = self.in3a(x)
        x = self.in3b(x)
        # x = self.max3_2(x)
        x = self.in4a(x)
        a1 = self.aux1(x)
        x = self.in4b(x)
        x = self.in4c(x)
        x = self.in4d(x)
        a2 = self.aux2(x)
        x = self.in4e(x)
        # x = self.max3_2(x)
        x = self.in5a(x)
        x = self.in5b(x)
        x = self.avgp(x)
        x = self.drop(x)
        x = self.flat(x)
        x = self.full(x)
        return [x, a1, a2]

    def model(self, inp):
        x = layers.Input(shape=inp[0].shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))
