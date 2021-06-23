import tensorflow as tf

#  Calculating the smooth l1 loss
def smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma):
    sigma2 = sigma ** 2
    sigma2 = tf.constant(sigma2, dtype=tf.float32)
    diff = in_weight * (pred_loc - gt_loc)
    abs_diff = tf.math.abs(diff)
    abs_diff = tf.cast(abs_diff, dtype=tf.float32)
    flag = tf.cast(abs_diff.numpy() < (1./sigma2), dtype=tf.float32)
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return tf.reduce_sum(y)

# Calculating localization loss
def fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    idx = gt_label > 0
    idx = tf.stack([idx, idx, idx, idx], axis=1)
    idx = tf.reshape(idx, [-1, 4])
    in_weight = tf.cast(idx, dtype=tf.int32)
    loc_loss = smooth_l1_loss(pred_loc, gt_loc, in_weight.numpy(), sigma)
    # Normalize by total number of negative and positive rois.
    loc_loss /= (tf.reduce_sum(tf.cast(gt_label >= 0, dtype=tf.float32)))  
    return loc_loss