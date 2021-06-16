import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from stage import stage


class repvgg(tf.keras.Model):
    def __init__(self, a=0.75, b=2.5, l=[1, 2, 4, 14, 1], nc=10):
        # default A0 architecture training on CIFAR-10 dataset

        super(repvgg, self).__init__()
        self.aug = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomCrop(32, 32),
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            ]
        )
        self.st1 = stage(min(64, 64 * a), l[0])
        self.st2 = stage(64 * a, l[1])
        self.st3 = stage(128 * a, l[2])
        self.st4 = stage(256 * a, l[3])
        self.st5 = stage(512 * b, l[4])
        self.gap = layers.GlobalAveragePooling2D()
        self.end = layers.Dense(
            units=nc, activation="softmax", kernel_regularizer=regularizers.l2(1e-4)
        )

    def call(self, inp):

        x = self.aug(inp)
        x = self.st1(x)
        x = self.st2(x)
        x = self.st3(x)
        x = self.st4(x)
        x = self.st5(x)
        x = self.gap(x)
        x = self.end(x)

        return x

    def model(self, inp):
        x = layers.Input(shape=inp[0].shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))
