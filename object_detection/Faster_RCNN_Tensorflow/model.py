import tensorflow as tf
import numpy as np
from region_proposal_network import RegionProposalNetwork, Extractor
from region_of_interest import RoIHead
from utils.utils_anchor import loc2bbox, AnchorTargetCreator, ProposalTargetCreator


class FasterRCNN(tf.keras.Model):

    def __init__(self, n_class, pool_size):
        super(FasterRCNN, self).__init__()
        self.n_class = n_class
        self.extractor = Extractor()
        self.rpn = RegionProposalNetwork()
        self.head = RoIHead(n_class, pool_size)
        self.score_thresh = 0.7
        self.nms_thresh = 0.3

    def __call__(self, x):
        img_size = x.shape[1:3]
        feature_map, rpn_locs, rpn_scores, rois, roi_score, anchor = self.rpn(x)
        roi_cls_locs, roi_scores = self.head(feature_map, rois, img_size)

        return roi_cls_locs, roi_scores, rois

    def predict(self, imgs):
        bboxes = []
        labels = []
        scores = []
        img_size = imgs.shape[1:3]
        
        roi_cls_loc, roi_score, rois = self(imgs)
        prob = tf.nn.softmax(roi_score, axis=-1)
        prob = prob.numpy()
        roi_cls_loc = roi_cls_loc.numpy()
        roi_cls_loc = roi_cls_loc.reshape(-1, self.n_class, 4)

        for label_index in range(1, self.n_class):

            cls_bbox = loc2bbox(rois, roi_cls_loc[:, label_index, :])
            # clip bounding box
            cls_bbox[:, 0::2] = tf.clip_by_value(cls_bbox[:, 0::2], clip_value_min=0, clip_value_max=img_size[0])
            cls_bbox[:, 1::2] = tf.clip_by_value(cls_bbox[:, 1::2], clip_value_min=0, clip_value_max=img_size[1])
            cls_prob = prob[:, label_index]

            mask = cls_prob > 0.05
            cls_bbox = cls_bbox[mask]
            cls_prob = cls_prob[mask]
            keep = tf.image.non_max_suppression(cls_bbox, cls_prob, max_output_size=-1, iou_threshold=self.nms_thresh)

            if len(keep) > 0:
                bboxes.append(cls_bbox[keep.numpy()])
                # The labels are in [0, self.n_class - 2].
                labels.append((label_index - 1) * np.ones((len(keep),)))
                scores.append(cls_prob[keep.numpy()])
        if len(bboxes) > 0:
            bboxes = np.concatenate(bboxes, axis=0).astype(np.float32)
            labels = np.concatenate(labels, axis=0).astype(np.float32)
            scores = np.concatenate(scores, axis=0).astype(np.float32)

        return bboxes, labels, scores