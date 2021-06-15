import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from generate_def_box import generate_def_box

# Following the rules for creating custom class, from tensorflow doocumentation

class DefBoxes(Layer):
    def __init__(self,
                 image_shape,
                 scale,
                 next_scale,
                 aspect_ratios,
                 variances,
                 offset=(0.5,0.5),
                 **kwargs
                ):
        self.image_shape = image_shape
        self.scale = scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.variances = variances
        self.offset = offset
        super(DefBoxes, self).__init__(**kwargs)
    
    def build(self, input_shape):
        _, feature_map_height, feature_map_width, _ = input_shape
        image_height, image_width, _ = self.image_shape
        
        # Necessary condition is that image and feature map must be "Squares" i.e. have equal height and widthq
        self.feature_map_size = feature_map_height
        self.image_size = image_height
        
        super(DefBoxes, self).build(input_shape)
        
    def call(self, inputs):
        default_boxes = generate_def_box(
            feature_map_size=self.feature_map_size,
            image_size=self.image_size,
            offset=self.offset,
            scale=self.scale,
            next_scale=self.next_scale,
            aspect_ratios=self.aspect_ratios,
            variances=self.variances,
        )
        
        default_boxes = np.expand_dims(default_boxes, axis=0)
        default_boxes = tf.constant(default_boxes, dtype='float32')
        default_boxes = tf.tile(default_boxes, (tf.shape(inputs)[0], 1, 1, 1, 1))
        return default_boxes
    
    
    def get_config(self):
        config = {
            "image_shape": self.image_shape,
            "scale": self.scale,
            "next_scale": self.next_scale,
            "aspect_ratios": self.aspect_ratios,
            "variances": self.variances,
            "offset": self.offset,
            "feature_map_size": self.feature_map_size,
            "image_size": self.image_size
        }
        base_config = super(DefBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)