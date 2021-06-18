import cv2
import numpy as np
import tensorflow as tf
from generate_def_box import generate_def_box
from utils.ssd_utils import one_hot_class_label, match_gt_boxes_to_default_boxes
from utils.bbox_util import encode_bboxes
from utils.data_utils import read_sample

scale = np.linspace(0.2, 0.9, 6)
aspect_ratios = [[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 0.33], [1.0, 2.0, 0.5, 3.0, 0.33], 
                 [1.0, 2.0, 0.5, 3.0, 0.33], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]
feature_maps = [38, 19, 10, 5, 3, 1]
variance = [0.1, 0.1, 0.2, 0.2]

class SSD_DATA_GENERATOR(tf.keras.utils.Sequence):

    def __init__(
        self,
        samples,
        label_maps,
        batch_size,
        process_input_fn,
    ):
        self.samples = samples
        self.batch_size = batch_size
        self.match_threshold = 0.5
        self.neutral_threshold = 0.3
        self.label_maps = ["__backgroud__"] + label_maps
        self.num_classes = len(self.label_maps)
        self.indices = range(0, len(self.samples))
        #
        assert self.batch_size <= len(self.indices), "batch size must be smaller than the number of samples"
        self.input_size = 300
        self.input_template = self.__get_input_template()
        self.process_input_fn = process_input_fn
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
    
    def __get_input_template(self):

        def_s = []
        conf_s = []
        loc_s = []
         
        for i in range(6):
            def_box = generate_def_box(feature_maps[i], self.input_size, [0.5, 0.5],
                                        scale[i], scale[i+1] if i < 5 else 1, aspect_ratios[i], variance)
            
            def_box = np.reshape(def_box, (-1, 8))
            conf = np.zeros((def_box.shape[0], self.num_classes))
            conf[:, 0] = 1

            def_s.append(def_box)
            loc_s.append(np.zeros((def_box.shape[0], 4)))
            conf_s.append(conf)
        
        def_s_c = np.concatenate(def_s, axis=0)
        conf_s_c = np.concatenate(conf_s, axis=0)
        loc_s_c = np.concatenate(loc_s, axis=0)

        template = np.concatenate([conf_s_c, loc_s_c, def_s_c], axis=-1)
        template = np.expand_dims(template, axis=0)
        return np.tile(template, (self.batch_size, 1, 1))

    def __get_data(self, batch):
        X = []
        y = self.input_template.copy()

        for batch_idx, sample_idx in enumerate(batch):
            image_path, label_path = self.samples[sample_idx].split(" ")
            image, bboxes, classes = read_sample(
                image_path=image_path,
                label_path=label_path
            )

            image_height, image_width, _ = image.shape
            height_scale, width_scale = self.input_size/image_height, self.input_size/image_width
            input_img = cv2.resize(np.uint8(image), (self.input_size, self.input_size))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = self.process_input_fn(input_img)

            gt_classes = np.zeros((bboxes.shape[0], self.num_classes))
            gt_boxes = np.zeros((bboxes.shape[0], 4))
            default_boxes = y[batch_idx, :, -8:]

            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                cx = (((bbox[0] + bbox[2]) / 2) * width_scale) / self.input_size
                cy = (((bbox[1] + bbox[3]) / 2) * height_scale) / self.input_size
                width = (abs(bbox[2] - bbox[0]) * width_scale) / self.input_size
                height = (abs(bbox[3] - bbox[1]) * height_scale) / self.input_size
                gt_boxes[i] = [cx, cy, width, height]
                gt_classes[i] = one_hot_class_label(classes[i], self.label_maps)

            matches, neutral_boxes = match_gt_boxes_to_default_boxes(
                gt_boxes=gt_boxes,
                default_boxes=default_boxes[:, :4],
                match_threshold=self.match_threshold,
                neutral_threshold=self.neutral_threshold
            )
            # set matched ground truth boxes to default boxes with appropriate class
            y[batch_idx, matches[:, 1], self.num_classes: self.num_classes + 4] = gt_boxes[matches[:, 0]]
            y[batch_idx, matches[:, 1], 0: self.num_classes] = gt_classes[matches[:, 0]]  # set class scores label
            # set neutral ground truth boxes to default boxes with appropriate class
            y[batch_idx, neutral_boxes[:, 1], self.num_classes: self.num_classes + 4] = gt_boxes[neutral_boxes[:, 0]]
            y[batch_idx, neutral_boxes[:, 1], 0: self.num_classes] = np.zeros((self.num_classes))  # neutral boxes have a class vector of all zeros
            # encode the bounding boxes
            y[batch_idx] = encode_bboxes(y[batch_idx])
            X.append(input_img)

        X = np.array(X, dtype=np.float)
        return X,y
        