import tensorflow as tf
import numpy as np
from models.region_proposal_network import Region_Proposal_Network, Feature_extractor
from models.region_of_interest import RoIHead
from utils.utils_anchor import loc2bbox, AnchorTargetCreator, ProposalTargetCreator


# Main faster rcnn model
class FasterRCNN(tf.keras.Model):

    def __init__(self, n_class, pool_size):
        super(FasterRCNN, self).__init__()
        self.n_class = n_class
        self.extractor = Feature_extractor()
        self.rpn = Region_Proposal_Network()
        self.head = RoIHead(n_class, pool_size)
        self.score_thresh = 0.7
        self.nms_thresh = 0.3

    def __call__(self, x):
        img_size = x.shape[1:3]
        feature_map, rpn_locs, rpn_scores, rois, roi_score, anchor = self.rpn(x)
        roi_cls_locs, roi_scores = self.head(feature_map, rois, img_size)

        return roi_cls_locs, roi_scores, rois