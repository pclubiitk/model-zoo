import tensorflow as tf
import numpy as np
from utils.utils_anchor import generate_anchor_base, ProposalCreator, enumerate_shifted_anchor


class Feature_extractor(tf.keras.Model):
    def __init__(self):
        super(Feature_extractor, self).__init__()
        
        # VGG16 model has been used here but later to be replaced by pretrained VGG16 TF
        # TODO : Replace VGG16 with pretrained model for better classification
        
        # conv1 block
        self.conv1_1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.conv1_2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        # conv2
        self.conv2_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.conv2_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        # conv3
        self.conv3_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.conv3_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.conv3_3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        # conv4
        self.conv4_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv4_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv4_3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.pool4 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        # conv5
        self.conv5_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv5_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv5_3 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')

    def __call__(self, imgs, training=None):
        h = self.pool1(self.conv1_2(self.conv1_1(imgs)))
        h = self.pool2(self.conv2_2(self.conv2_1(h)))
        h = self.pool3(self.conv3_3(self.conv3_2(self.conv3_1(h))))
        h = self.pool4(self.conv4_3(self.conv4_2(self.conv4_1(h))))
        h = self.conv5_3(self.conv5_2(self.conv5_1(h)))
        return h

class Region_Proposal_Network(tf.keras.Model):

    def __init__(self, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        super(Region_Proposal_Network, self).__init__()

        # Region Proposal Conv layer
        self.region_proposal_conv = tf.keras.layers.Conv2D(512, kernel_size=3, activation=tf.nn.relu, padding='same')
        # Bounding Boxes Regression layer 
        self.loc = tf.keras.layers.Conv2D(36, kernel_size=1, padding='same')
        self.score = tf.keras.layers.Conv2D(18, kernel_size=1, padding='same')

        self.anchor = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.proposal_layer = ProposalCreator()

    def __call__(self, x, img_size, scale, training=None):

        n, hh, ww, _ = x.shape
        anchor = enumerate_shifted_anchor(np.array(self.anchor), 16, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)

        h = self.region_proposal_conv(x)
        rpn_loc = self.loc(h)
        rpn_loc = tf.reshape(rpn_loc, [n, -1, 4])

        rpn_score = self.score(h)
        
        rpn_softmax_score = tf.nn.softmax(tf.reshape(rpn_score, [n, hh, ww, n_anchor, 2]), axis=-1)
        rpn_fg_score = rpn_softmax_score[:, :, :, :, 1]
        rpn_fg_score = tf.reshape(rpn_fg_score, [n, -1])
        rpn_score = tf.reshape(rpn_score, [n, -1, 2])

        roi = self.proposal_layer(rpn_loc[0].numpy(), rpn_fg_score[0].numpy(), anchor, img_size, scale)

        return rpn_loc, rpn_score, roi, anchor