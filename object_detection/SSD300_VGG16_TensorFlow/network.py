from tensorflow.keras import optimizers
from utils.ssd_utils import *
from utils.data_utils import *
from utils.bbox_util import *
from Datagen import *
from losses import *
from ssd_model import *
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import preprocess_input


train_split_path = "PascalVocDataset\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\ImageSets\\Layout\\train.txt"
img_dir = "PascalVocDataset\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages"
xml_dir = "PascalVocDataset\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\Annotations"

label_maps = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def ssd300_vgg16():
    train_samples = generate_samples_from_split(train_split_path, img_dir, xml_dir)
    training_data_generator = SSD_DATA_GENERATOR(train_samples, label_maps, 16, preprocess_input)
    loss = SSDLoss()

    model = ssd_model()

    optimizer = SGD(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=loss.compute)

    model.fit(x=training_data_generator, batch_size=16, epochs=1)

ssd300_vgg16()



# train_samples = generate_samples_from_split(train_split_path, img_dir, xml_dir)
# training_data_generator = SSD_DATA_GENERATOR(train_samples, label_maps, 16, preprocess_input)


