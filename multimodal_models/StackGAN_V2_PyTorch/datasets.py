import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from helper_functions.ret_image import ret_image
# from miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 base_size=64, Train=True, transform=None, target_transform=None, StageNum = 3):
        self.StageNum = StageNum
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        self.imsize = []
        for i in range(3):
            self.imsize.append(base_size)
            base_size = base_size * 2
        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.captions = self.load_all_captions()
        if Train:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = 'CUB_200_2011/bounding_boxes.txt'
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        filepath = 'CUB_200_2011/images.txt'
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        return filename_bbox

    def load_all_captions(self):
        def load_captions(caption_name):
            cap_path = caption_name
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
            captions = [cap.replace("\ufffd\ufffd", " ")
                        for cap in captions if len(cap) > 0]
            return captions

        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text_c10/%s.txt' % (self.data_dir, key)
            captions = load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = 'char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = 'char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = 'skip-thought-embeddings.pickle'
        with open(data_dir + "/" + embedding_filename, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            embeddings = u.load()
            embeddings = np.array(embeddings)
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                class_id = u.load()
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = 'CUB_200_2011'
        else:
            bbox = None
            data_dir = self.data_dir
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = ret_image(img_name, self.imsize, 
                        self.StageNum, bbox, self.transform, normalize=self.norm)

        wrong_ix = random.randint(0, len(self.filenames) - 1)
        if(self.class_id[index] == self.class_id[wrong_ix]):
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        if self.bbox is not None:
            wrong_bbox = self.bbox[wrong_key]
        else:
            wrong_bbox = None
        wrong_img_name = '%s/images/%s.jpg' % \
            (data_dir, wrong_key)
        wrong_imgs = ret_image(wrong_img_name, self.imsize,
                              self.StageNum, wrong_bbox, self.transform, normalize=self.norm)

        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)

        return imgs, wrong_imgs, embedding, key

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = 'CUB_200_2011'
        else:
            bbox = None
            data_dir = self.data_dir
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = ret_image(img_name, self.imsize,
                        self.StageNum, bbox, self.transform, normalize=self.norm)
        if self.target_transform is not None:
            embeddings = self.target_transform(embeddings)
        return imgs, embeddings, key

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)