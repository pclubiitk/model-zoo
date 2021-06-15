import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# say feature map size is 38x38 and image size is 300x300
# so there will be 38x38 grids in the image each grid of size 300 / 38
# offset is the position of centre of default boxes (usually centre of grid)
# scale and next_scale is the scale of current feature map and next feature map respectively
# aspect_ratios is a list containing different aspect ratios
# variances 

def generate_def_box(feature_map_size, 
                    image_size,
                    offset,
                    scale,
                    next_scale,
                    aspect_ratios,
                    variances):
    
    width_height = []
    
    # In the paper, every feature map had two default boxes with aspect ratio 1, with different scale
    width_height.append([
                image_size * np.sqrt(scale * next_scale) * np.sqrt(1.0),
                image_size * np.sqrt(scale * next_scale) * (1 / np.sqrt(1.0)),
            ])
    
    for ar in aspect_ratios:
        width_height.append([
            image_size * scale * np.sqrt(ar),
            image_size * scale * (1 / np.sqrt(ar))
        ])
    
    #we will pass aspect_ratio list such that lenght of the array + 1 is number of default boxes for the feature map
    num_box = len(width_height)
    
    
    wh_list= np.array(width_height, dtype=np.float)
    
    grid_size = image_size / feature_map_size
    offset_x, offset_y = offset
    
    # get all center points (they will be equispaced obviously, hence linspace is used) of each grid cells
    cx = np.linspace(offset_x * grid_size, image_size - (offset_x * grid_size), feature_map_size)
    cy = np.linspace(offset_y * grid_size, image_size - (offset_y * grid_size), feature_map_size)
    
    cx_grid, cy_grid = np.meshgrid(cx, cy)
    cx_grid, cy_grid = np.expand_dims(cx_grid, axis=-1), np.expand_dims(cy_grid, axis=-1)
    cx_grid, cy_grid = np.tile(cx_grid, (1, 1, num_box)), np.tile(cy_grid, (1, 1, num_box))
    #
    
    #dimension of default boxes (part of output) has to be (38 x 38 x num_box x 4) if feature_map has size 38
    default_boxes = np.zeros((feature_map_size, feature_map_size, num_box, 4))
    default_boxes[:, :, :, 0] = cx_grid
    default_boxes[:, :, :, 1] = cy_grid
    default_boxes[:, :, :, 2] = wh_list[:, 0]
    default_boxes[:, :, :, 3] = wh_list[:, 1]
    
    default_boxes[:, :, :, [0, 2]] /= image_size
    default_boxes[:, :, :, [1, 3]] /= image_size
    
    
    variances_tensor = np.zeros_like(default_boxes)
    variances_tensor += variances
    default_boxes = np.concatenate([default_boxes, variances_tensor], axis=-1)
    
    return default_boxes