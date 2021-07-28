import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from stage import stage


class repvgg(tf.keras.Model):
    def __init__(self, a=0.75, b=2.5, l=[1, 2, 4, 14, 1], nc=10):
        # default A0 architecture training on CIFAR-10 dataset

        super(repvgg, self).__init__()
        """
        a:= Same purpose as that in paper, used for number of channels
        b:= Same purpose as that in paper, used for number of channels
        l:= The number of layers per stage
        nc:= The total number of classifications of dataset

        Model consists of a layer for augmentation, followed by 5 stages, and by a 
        Global Average Pooling layer and a fully connected layer
        """
        self.aug = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomCrop(32, 32),
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            ]
        )
        self.st = []
        self.l = l
        self.f = [min(64, 64 * a), 64 * a, 128 * a, 256 * a, 512 * b]
        for i in range(5):
            self.st.append(stage(self.f[i], l[i]))
        self.gap = layers.GlobalAveragePooling2D()
        self.end = layers.Dense(
            units=nc, activation="softmax", kernel_regularizer=regularizers.l2(1e-4)
        )

    def call(self, inp):

        x = self.aug(inp)
        for i in range(5):
            x = self.st[i](x)
        x = self.gap(x)
        x = self.end(x)

        return x

    def model(self, inp):
        # Can be used for printing model summary
        x = layers.Input(shape=inp[0].shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))

    def repara(self):
        for i in range(5):
            self.st[i].repara()
        return
