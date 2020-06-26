import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torchvision

class TextDataset(data.Dataset):
    def __init__(self,img_dir, data_dir,transform1,transform2, split,imsize1,imsize2):
        self.img_dir=img_dir
        self.transform1 = transform1
        self.transform2=transform2
        self.imsize1 = imsize1
        self.imsize2 = imsize2
        self.data = []
        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        # self.captions = self.load_all_captions()

    def get_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        load_size1 = int(self.imsize1 * 76 / 64)
        load_size2 = int(self.imsize2 * 76 / 64)
        img1 = img.resize((load_size1, load_size1), PIL.Image.BILINEAR)
        img2 = img.resize((load_size2, load_size2), PIL.Image.BILINEAR)
        if self.transform1 is not None:
            img1 = self.transform1(img1)
        if self.transform2 is not None:
            img2 = self.transform2(img2)    
        return img1,img2

  

    def load_embedding(self, data_dir):
       
        embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        with open(data_dir + embedding_filename, 'rb') as f:
            #embeddings = pickle.load(f)
            embeddings = pickle._Unpickler(f)
            embeddings.encoding = 'latin1'
            embeddings = embeddings.load()
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def __getitem__(self, index):
        key = self.filenames[index]
        # cls_id = self.class_id[index]
       
        data_dir = self.data_dir
        img_dir = self.img_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/%s.jpg' % ( img_dir,key)
        img1,img2 = self.get_img(img_name)
        embedding_ix = random.randint(0, embeddings.shape[0]-1)
        embedding = embeddings[embedding_ix, :]
        return img1,img2, embedding

    def __len__(self):
        return len(self.filenames)
