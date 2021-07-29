import matplotlib.pyplot as plt
from utils.utils_data import LABEL_NAMES
import tensorflow as tf
import numpy as np
from utils.utils_anchor import loc2bbox

def vis_train(img, bbox, label, roi, roi_score, epoch,sample_roi,roi_cls_loc):
    
    img_size = img.shape[1:3]
    img = (img[0] * 0.225) + 0.45
    img = img.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    
    roi_prob = tf.nn.softmax(roi_score, axis=-1)
    roi_prob = roi_prob.numpy()
    roi_cls_loc = roi_cls_loc.numpy()
    roi_cls_loc = roi_cls_loc.reshape(-1, 21, 4)
    
    modified_roi_bboxes, modified_roi_labels, modified_roi_scores = [],[],[]

    for label_index in range(1, 21):
        
        cls_bbox = loc2bbox(sample_roi, roi_cls_loc[:, label_index, :])
        
        cls_bbox[:, 0::2] = tf.clip_by_value(cls_bbox[:, 0::2], clip_value_min=0, clip_value_max=img_size[0])
        cls_bbox[:, 1::2] = tf.clip_by_value(cls_bbox[:, 1::2], clip_value_min=0, clip_value_max=img_size[1])
        cls_prob = roi_prob[:, label_index]
        
        mask = cls_prob > 0.05
        cls_bbox = cls_bbox[mask]
        cls_prob = cls_prob[mask]
        
        #Using non max suppression to reduce no of bboxes
        keep = tf.image.non_max_suppression(cls_bbox, cls_prob, max_output_size=len(bbox), iou_threshold=0.7)

        if len(keep) > 0:
            modified_roi_bboxes.append(cls_bbox[keep.numpy()])
            modified_roi_labels.append((label_index - 1) * np.ones((len(keep),)))
            modified_roi_scores.append(cls_prob[keep.numpy()])
            
        
    if len(bbox) > 0:
        modified_roi_bboxes = np.concatenate(modified_roi_bboxes, axis=0).astype(np.float32)
        modified_roi_labels = np.concatenate(modified_roi_labels, axis=0).astype(np.float32)
        modified_roi_scores = np.concatenate(modified_roi_scores, axis=0).astype(np.float32)
    print(modified_roi_bboxes.shape)

    for i in range(len(bbox)):
        y1 = bbox[i][0]
        x1 = bbox[i][1]
        y2 = bbox[i][2]
        x2 = bbox[i][3]
        height = y2 - y1
        width = x2 - x1
        ax.add_patch(plt.Rectangle((x1,y1), width, height, fill=False, edgecolor='green', linewidth=2))
        ax.text(x1,y1,LABEL_NAMES[label[i]],style='normal',bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 0})
    
    for i in range(len(modified_roi_bboxes)):
        y1 = modified_roi_bboxes[i][0]
        x1 = modified_roi_bboxes[i][1]
        y2 = modified_roi_bboxes[i][2]
        x2 = modified_roi_bboxes[i][3]
        height = y2 - y1
        width = x2 - x1
        ax.add_patch(plt.Rectangle((x1,y1), width, height, fill=False, edgecolor='red', linewidth=1))
        ax.text(x1,y1,LABEL_NAMES[int(modified_roi_labels[i])],style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 0})
    plt.show()