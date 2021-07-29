# This file contains utility functions for bounding boxes
import numpy as np

# Converts the bounding box parameters from (cx, cy, width, height) to (xmin, ymin, xmax, ymax) format
def centre_to_corner(bbox):
    # bbox is having dimension (number_of_boxes, 4)
    
    # a[0] = cx - width/2
    # a[1] = cx + width/2

    bbox2 = np.copy(bbox)

    bbox2[..., 0] = bbox[..., 1] - (0.5 * bbox[..., 2])
    bbox2[..., 1] = bbox[..., 2] - (0.5 * bbox[..., 3])
    bbox2[..., 2] = bbox[..., 1] + (0.5 * bbox[..., 2])
    bbox2[..., 3] = bbox[..., 2] + (0.5 * bbox[..., 3])
    
    return bbox2

# Converts the bounding box parameters from (xmin, ymin, xmax, ymax) to (cx, cy, width, height)format
def corner_to_centre(bbox):

    bbox2 = np.copy(bbox)

    bbox2[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    bbox2[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    bbox2[..., 2] = np.abs(bbox[..., 0] - bbox[..., 2])
    bbox2[..., 3] = np.abs(bbox[..., 1] - bbox[..., 3])

    return bbox2

# This is a function to encode the true "y_value" so that it can be passes in model for training
def encode_bboxes(y, epsilon=10e-5):
   
    gt_boxes = y[:, -12:-8]
    df_boxes = y[:, -8:-4]
    variances = y[:, -4:]
    encoded_gt_boxes_cx = ((gt_boxes[:, 0] - df_boxes[:, 0]) / (df_boxes[:, 2])) / np.sqrt(variances[:, 0])
    encoded_gt_boxes_cy = ((gt_boxes[:, 1] - df_boxes[:, 1]) / (df_boxes[:, 3])) / np.sqrt(variances[:, 1])
    encoded_gt_boxes_w = np.log(epsilon + gt_boxes[:, 2] / df_boxes[:, 2]) / np.sqrt(variances[:, 2])
    encoded_gt_boxes_h = np.log(epsilon + gt_boxes[:, 3] / df_boxes[:, 3]) / np.sqrt(variances[:, 3])
    y[:, -12] = encoded_gt_boxes_cx
    y[:, -11] = encoded_gt_boxes_cy
    y[:, -10] = encoded_gt_boxes_w
    y[:, -9] = encoded_gt_boxes_h
    return y