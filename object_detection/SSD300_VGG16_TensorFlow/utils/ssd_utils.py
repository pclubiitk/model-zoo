import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from utils.bbox_util import centre_to_corner
from iou import iou
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Reshape
from Default_box_layers import DefBoxes

num_class = 21
variance = [0.1, 0.1, 0.2, 0.2]
aspect_ratio_4 = [1.0, 2.0, 0.5]
aspect_ratio_6 = [1.0, 2.0, 0.5, 3.0, 0.33]

# The below 2 are custom functions, used in model architecture
'''
It takes the output of a feature map and returns three layers.
First layer : Confidence scores for different classes.
Second layer : Bounding Box Parameters for predictions.
Third layer : Default Boxes layer corresponding to the feature map.
'''
def get_pred_4(input, scale1, next_scale1):
  
    conf_4 = Conv2D(4 * num_class, (3,3), padding='same')(input)
    loc_4 = Conv2D(4 * 4, (3,3), padding='same')(input)
    def_box_4 = DefBoxes((300, 300, 3), scale1, next_scale1, aspect_ratio_4, variance)(input)

    conf_4 = Reshape((-1, num_class))(conf_4)
    loc_4 = Reshape((-1, 4))(loc_4)
    def_box_4 = Reshape((-1, 8))(def_box_4)

    return conf_4, loc_4, def_box_4

def get_pred_6(input, scale1, next_scale1):
  
    conf_6 = Conv2D(6 * num_class, (3,3), padding='same')(input)
    loc_6 = Conv2D(6 * 4, (3,3), padding='same')(input)
    def_box_6 = DefBoxes((300, 300, 3), scale1, next_scale1, aspect_ratio_6, variance)(input)

    conf_6 = Reshape((-1, num_class))(conf_6)
    loc_6 = Reshape((-1, 4))(loc_6)
    def_box_6 = Reshape((-1, 8))(def_box_6)

    return conf_6, loc_6, def_box_6

# Generates a label vector with value 1 corresponding to given class
def one_hot_class_label(classname, label_maps):
    temp = np.zeros((len(label_maps)), dtype=np.int)
    temp[label_maps.index(classname)] = 1
    return temp

'''
As in SSD training, not all default boxes will correspond to an object.
Hence we need to match default boxes with ground truth boxes ( here done by Jaccard overlap ).
This function does the purpose and returns matched boxes.
'''
def match_gt_boxes_to_default_boxes(gt_boxes, default_boxes, match_threshold=0.5, neutral_threshold=0.3):
    gt_boxes = centre_to_corner(gt_boxes)
    default_boxes = centre_to_corner(default_boxes)

    num_gt = gt_boxes.shape[0]
    num_def = default_boxes.shape[0]

    matches = np.zeros((num_gt, 2), dtype=np.int)

    for i in range(num_gt):
        g_box = gt_boxes[i]
        g_box = np.tile(np.expand_dims(g_box, axis=0), (num_def, 1))
        
        ious = iou(g_box, default_boxes)
        matches[i] = [i, np.argmax(ious)]
    
    gt_boxes = np.tile(np.expand_dims(gt_boxes, axis=1), (1, num_def, 1))
    default_boxes = np.tile(np.expand_dims(default_boxes, axis=0), (num_gt, 1, 1))
    ious = iou(gt_boxes, default_boxes)
    ious[:, matches[:, 1]] = 0

    matched_gt_boxes_idxs = np.argmax(ious, axis=0)  # for each default boxes, select the ground truth box that has the highest iou
    matched_ious = ious[matched_gt_boxes_idxs, list(range(num_def))]  # get iou scores between gt and default box that were selected above
    matched_df_boxes_idxs = np.nonzero(matched_ious >= match_threshold)[0]  # select only matched default boxes that has iou larger than threshold
    matched_gt_boxes_idxs = matched_gt_boxes_idxs[matched_df_boxes_idxs]

    # concat the results of the two matching process together
    matches = np.concatenate([
        matches,
        np.concatenate([
            np.expand_dims(matched_gt_boxes_idxs, axis=-1),
            np.expand_dims(matched_df_boxes_idxs, axis=-1)
        ], axis=-1),
    ], axis=0)
    ious[:, matches[:, 1]] = 0

    # find neutral boxes (ious that are higher than neutral_threshold but below threshold)
    # these boxes are neither background nor has enough ious score to qualify as a match.
    background_gt_boxes_idxs = np.argmax(ious, axis=0)
    background_gt_boxes_ious = ious[background_gt_boxes_idxs, list(range(num_def))]
    neutral_df_boxes_idxs = np.nonzero(background_gt_boxes_ious >= neutral_threshold)[0]
    neutral_gt_boxes_idxs = background_gt_boxes_idxs[neutral_df_boxes_idxs]
    neutral_boxes = np.concatenate([
        np.expand_dims(neutral_gt_boxes_idxs, axis=-1),
        np.expand_dims(neutral_df_boxes_idxs, axis=-1)
    ], axis=-1)

    return matches, neutral_boxes
