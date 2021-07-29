import tensorflow as tf
from utils.utils_anchor import AnchorTargetCreator, ProposalTargetCreator
from utils.utils_model import fast_rcnn_loc_loss

class FasterRCNNTrainer(tf.keras.Model):

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = 3.0
        self.roi_sigma = 1.0
        # Target creator create gt_bbox & gt_label as training targets.
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

    def __call__(self, imgs, bbox, label, scale, training=None):
        _, H, W, _ = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs, training=training)
        rpn_locs, rpn_scores, roi, anchor = self.faster_rcnn.rpn(features, img_size, scale, training=training)

        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]

        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox.numpy(), label.numpy())
        roi_cls_loc, roi_score = self.faster_rcnn.head(features, sample_roi, img_size, training=training)

        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox.numpy(), anchor, img_size)
        gt_rpn_label = tf.constant(gt_rpn_label, dtype=tf.int32)
        gt_rpn_loc = tf.constant(gt_rpn_loc, dtype=tf.float32)
        rpn_loc_loss = fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
        idx_ = gt_rpn_label != -1
        rpn_cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(gt_rpn_label[idx_], rpn_score[idx_])

        # ROI losses
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = tf.reshape(roi_cls_loc, [n_sample, -1, 4])
        idx_ = [[i, j] for i, j in zip(tf.range(n_sample), tf.constant(gt_roi_label))]
        roi_loc = tf.gather_nd(roi_cls_loc, idx_)
        gt_roi_label = tf.constant(gt_roi_label)
        gt_roi_loc = tf.constant(gt_roi_loc)
        roi_loc_loss = fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
        idx_ = gt_roi_label != 0
        roi_cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(gt_roi_label[idx_], roi_score[idx_])

        return rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss