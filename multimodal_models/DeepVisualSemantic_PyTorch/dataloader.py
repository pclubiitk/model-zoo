import os
import json
import nltk
import time
import torch
from PIL import Image

class DataLoader():

    def __init__(self, dir_path, vocab, transform):
        self.images = None
        self.captions_dict = None
        self.vocab = vocab
        self.transform = transform
        self.load_captions(dir_path)
        self.load_images(dir_path)

    def load_images(self, dir_path):
        files = os.listdir(dir_path)
        images = {}
        for file in files :
            extension = file.split('.')[1]
            if extension == 'jpg':
                images[file] = self.transform(Image.open(os.path.join(dir_path, file)))

        self.images = images

    def load_captions(self, dir_path):
        file = os.path.join(dir_path, 'captions.txt')
        captions_dict = {}
        with open(file) as f:
            for line in f:
                curr_dict = json.loads(line)
                for i,txt in curr_dict.items():
                    captions_dict[i] = txt

        self.captions_dict = captions_dict

    def caption2ids(self, caption):
        vocab = self.vocab
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        vec = []
        vec.append(vocab.get_id('<start>'))
        vec.extend([vocab.get_id(word) for word in tokens])
        vec.append(vocab.get_id('<end>'))

        return vec

    def gen_data(self):
        images = []
        captions = []
        for image_id, curr_captions in self.captions_dict.items():
            num_captions = len(curr_captions)
            images.extend([image_id] * num_captions)
            for caption in curr_captions:
                captions.append(self.caption2ids(caption))

        data = images, captions
        return data

    def get_image(self, image_id):
        return self.images[image_id]

def shuffle_data(data, seed=0):
    images, captions = data
    shuffled_imgs = []
    shuffled_captions = []
    num_images = len(images)
    torch.manual_seed(seed)
    perm = list(torch.randperm(num_images))
    for i in range(num_images):
        shuffled_captions.append(captions[perm[i]])
        shuffled_imgs.append(images[perm[i]])

    return shuffled_imgs, shuffled_captions
