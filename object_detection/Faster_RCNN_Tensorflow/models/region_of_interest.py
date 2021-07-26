import tensorflow as tf


def roi_pooling(feature, rois, img_size, pool_size):
    
    box_ind = tf.zeros(rois.shape[0], dtype=tf.int32)
    normalization = tf.cast(tf.stack([img_size[0], img_size[1], img_size[0], img_size[1]], axis=0), dtype=tf.float32)
    boxes = rois / normalization
    pool = tf.image.crop_and_resize(feature, boxes, box_ind, crop_size=pool_size)
    return pool

# Using ROI pooling 2D from keras for faster computation
class RoIPooling2D(tf.keras.Model):

    def __init__(self, pool_size):
        super(RoIPooling2D, self).__init__()
        self.pool_size = pool_size

    def __call__(self, feature, rois, img_size):
        return roi_pooling(feature, rois, img_size, self.pool_size)


class RoIHead(tf.keras.Model):

    def __init__(self, n_class, pool_size):
        # n_class includes the background
        super(RoIHead, self).__init__()

        self.fc = tf.keras.layers.Dense(4096)
        self.cls_loc = tf.keras.layers.Dense(n_class * 4)
        self.score = tf.keras.layers.Dense(n_class)

        self.n_class = n_class
        self.roi = RoIPooling2D(pool_size)

    def __call__(self, feature, rois, img_size, training=None):

        rois = tf.constant(rois, dtype=tf.float32)
        pool = self.roi(feature, rois, img_size)
        pool = tf.reshape(pool, [rois.shape[0], -1])
        fc = self.fc(pool)
        roi_cls_locs = self.cls_loc(fc)
        roi_scores = self.score(fc)

        return roi_cls_locs, roi_scores
