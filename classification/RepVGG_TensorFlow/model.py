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

    def rep(self):
        """
        Returns the weights of the whole model.
        a0, a1 = Weights of augmentation layers
        w = contains all the kernels of the model stage-wise
        b = contains all the biases of the model stage-wise
        """

        a0 = self.aug.get_layer(index=0).get_weights()
        a1 = self.aug.get_layer(index=1).get_weights()
        w = []
        b = []
        for i in range(5):
            wi, bi = self.st[i].new_para()
            w.append(wi)
            b.append(bi)
        d = self.end.get_weights()
        return a0, a1, w, b, d

    def inf(self):
        """
        Returns an inference-time model corresponding to the trained model,
        using weights from self.rep()
        """
        inf_model = tf.keras.Sequential()

        a0, a1, w, b, d = self.rep()

        inf_model.add(layers.experimental.preprocessing.RandomCrop(32, 32, weights=a0))
        inf_model.add(
            layers.experimental.preprocessing.RandomFlip(
                "horizontal_and_vertical", weights=a1
            )
        )

        for i in range(5):
            for j in range(self.l[i]):
                if j == 0:
                    inf_model.add(
                        layers.Conv2D(
                            self.f[i],
                            kernel_size=3,
                            strides=2,
                            padding="same",
                            weights=[w[i][j], b[i][j]],
                        )
                    )
                    inf_model.add(layers.Activation(activations.relu))
                else:
                    inf_model.add(
                        layers.Conv2D(
                            self.f[i],
                            kernel_size=3,
                            padding="same",
                            weights=[w[i][j], b[i][j]],
                        )
                    )
                    inf_model.add(layers.Activation(activations.relu))

        inf_model.add(layers.GlobalAveragePooling2D())
        inf_model.add(layers.Dense(units=10, activation="softmax", weights=d))
        return inf_model
