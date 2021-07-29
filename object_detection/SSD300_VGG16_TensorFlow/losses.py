import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import square

class SmoothL1Loss:

    def compute(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2

        # Whenever abs_loss is less than 1 return square loss, otherwise return abs loss
        final_loc_loss = tf.where(tf.less(abs_loss, 1.0), square_loss, abs_loss)
        return tf.reduce_sum(final_loc_loss, axis=-1)

class SoftmaxLoss:

    def compute(self, y_true, y_pred):

        # since there is logarithm in final loss, we need to ensure that there is no 0 in y_pred
        y_pred = tf.maximum(y_pred, 1e-15)

        final_soft_loss = tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return -1 * final_soft_loss

class SSDLoss:
    def __init__(self, alpha=1, min_neg_box=0, neg_box_ratio=3):
        self.smoothloss = SmoothL1Loss()
        self.softloss = SoftmaxLoss()
        self.min_neg_box = min_neg_box
        self.neg_box_ratio = neg_box_ratio
        self.alpha = alpha

    def compute(self, y_true, y_pred):

        smooth_loss = self.smoothloss.compute(y_true[:,:,:-12], y_pred[:,:,:-12]) #class
        soft_loss = self.softloss.compute(y_true[:,:,-12:-8], y_pred[:,:,-12:-8]) #bbox params

        batch_size = tf.shape(y_true)[0]
        num_box = tf.shape(y_true)[1]

        pos_class = tf.reduce_max(y_true[:, :, 1:-12], axis=-1)
        neg_class = y_true[:, :, 0]

        num_pos = tf.cast(tf.reduce_sum(pos_class), tf.int32)
        pos_smooth_loss = tf.reduce_sum(smooth_loss * pos_class, axis=-1)
        pos_soft_loss = tf.reduce_sum(soft_loss * pos_class, axis=-1)

        neg_soft_loss = soft_loss * neg_class
        num_neg_soft_loss = tf.math.count_nonzero(neg_soft_loss, dtype=tf.int32)

        num_neg_soft_loss_final = tf.minimum(tf.maximum(self.neg_box_ratio * num_pos, self.min_neg_box), num_neg_soft_loss)

        def f1():
            return tf.zeros([batch_size])

        def f2():
            neg_soft_loss_1d = tf.reshape(neg_soft_loss, [-1])
            _, indices = tf.nn.top_k(
                neg_soft_loss_1d,
                k=num_neg_soft_loss_final,
                sorted=False
            )
            negatives_keep = tf.scatter_nd(
                indices=tf.expand_dims(indices, axis=1),
                updates=tf.ones_like(indices, dtype=tf.int32),
                shape=tf.shape(neg_soft_loss_1d)
            )
            negatives_keep = tf.cast(tf.reshape(negatives_keep, [batch_size, num_box]), tf.float32)
            neg_class_loss = tf.reduce_sum(soft_loss * negatives_keep, axis=-1)
            return neg_class_loss

        neg_soft_loss = tf.cond(tf.equal(num_neg_soft_loss, tf.constant(0)), f1, f2)
        soft_loss = pos_soft_loss + neg_soft_loss

        total = (soft_loss + self.alpha * pos_smooth_loss) / tf.maximum(1.0, tf.cast(num_pos, tf.float32))
        total = total * tf.cast(batch_size, tf.float32)
        return total
