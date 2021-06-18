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
            self.append(lay(filters, i == 0))

    def call(self, inp):
        x = inp
        for i in range(self.num):
            x = self.lay[i](x)
        return x

    def new_para(self):
        """
        Collects the weights of the stage layer-wise and returns them.
        """
        w = []
        b = []
        for i in range(self.num):
            wi, bi = self.lay[i].parameters()
            w.append(wi)
            b.append(bi)
        return w, b
