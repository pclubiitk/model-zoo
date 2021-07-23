import datetime

from utils.utils_visualizer import vis_train
from models.model import FasterRCNN
from models.trainer import FasterRCNNTrainer
import tensorflow as tf
from utils.utils_data import Dataset, vis, LABEL_NAMES, DATA_DIR
import pkbar
import matplotlib.pyplot as plt

# HYPERPARAMETER
epochs = 5

print('Loading dataset')
dataset = Dataset(DATA_DIR)


frcnn = FasterRCNN(21, (7, 7))
print('Constructing Model')

model = FasterRCNNTrainer(frcnn)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)



loss = []
for epoch in range(epochs):
    kbar = pkbar.Kbar(target=len(dataset), epoch=epoch, num_epochs=epochs, width=20, always_stateful=False)
    for i in range(len(dataset)):
        img, bboxes, labels, scale = dataset[1]
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.float32)
        with tf.GradientTape() as tape:
            rpn_loc_l, rpn_cls_l, roi_loc_l, roi_cls_l = model(img, bboxes, labels, scale, training=True)
            total_loss = rpn_loc_l + rpn_cls_l + roi_loc_l + roi_cls_l
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        kbar.update(i, values=[("loss", total_loss)])

        # Displaying predictions on train, TODO: Use test data for predictions.
        if(i%100==0 and i!=0):
            img, bboxes, labels, scale = dataset[1]
            feature_map = model.faster_rcnn.extractor(img)
            f = tf.reduce_sum(feature_map, axis=-1)
            img_size = img.shape[1:3]
            rpn_locs, rpn_scores, rois, anchor = model.faster_rcnn.rpn(feature_map, img_size, scale)
            sample_roi, gt_roi_loc, gt_roi_label = model.proposal_target_creator(rois, bboxes, labels)
            roi_cls_loc, roi_score = model.faster_rcnn.head(feature_map, sample_roi, img_size)
            
            vis_train(img, bboxes, labels, rois, roi_score, epoch,sample_roi,roi_cls_loc)