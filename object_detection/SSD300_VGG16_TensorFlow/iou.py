import numpy as np

# function to evaluate IOU between two boxes
def iou(bbox1, bbox2):

    # shape of both tensor is (num_box, 4) 
    # value in format (xmin, ymin, xmax, ymax)
    
    xmin_inter = np.maximum(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = np.maximum(bbox1[..., 1], bbox2[..., 1])

    xmax_inter = np.minimum(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = np.minimum(bbox1[..., 3], bbox2[..., 3])

    inter = (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter)
    bb1_ar = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    bb2_ar = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])
    union_ar =  bb1_ar + bb2_ar - inter

    iou_res = inter/union_ar

    iou_res[xmax_inter < xmin_inter] = 0
    iou_res[ymax_inter < ymin_inter] = 0
    iou_res[iou_res < 0] = 0
    iou_res[iou_res > 1] = 0

    return iou_res