import numpy as np

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

def corner_to_centre(bbox):

    bbox2 = np.copy(bbox)

    bbox2[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    bbox2[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    bbox2[..., 2] = np.abs(bbox[..., 0] - bbox[..., 2])
    bbox2[..., 3] = np.abs(bbox[..., 1] - bbox[..., 3])

    return bbox2