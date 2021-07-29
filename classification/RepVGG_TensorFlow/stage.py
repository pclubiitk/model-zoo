from tensorflow.keras import layers
from lay import lay


class stage(layers.Layer):
    def __init__(self, filters, layer):
        """
        filters:= The number of channels the layers must have
        layer:= The number of layers in the stage
        """

        super(stage, self).__init__()
        self.num = layer
        self.lay = []
        for i in range(self.num):
            self.lay.append(lay(filters, i == 0))

    def call(self, inp):
        x = inp
        for i in range(self.num):
            x = self.lay[i](x)
        return x

    def repara(self):
        for i in range(self.num):
            self.lay[i].repara()
        return
