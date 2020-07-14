import tensorflow as tf
import keras
import numpy as np
from numpy import array
import os
from PIL import Image
import glob
import pickle
from pickle import dump, load
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from utils import *
from model import *
##############################################
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type = int, default = 10, help = "No of epochs: default 20 ")
parser.add_argument('--base_dir', default = '.', help = "Base Directory for storing the dataset")
parser.add_argument('--num_photos_per_batch', type = int, default = 3, help = "Number of photos per batch in training: default 3 ")
parser.add_argument('--em_dem', type = int, default = 200, help = "Denote embedding dimension : default 200 ")
args = parser.parse_args()

filename_token = args.base_dir + '/all_captions/Flickr8k.token.txt'
fname_trainImage = args.base_dir +'/all_captions/Flickr_8k.trainImages.txt'
images = args.base_dir +'/all_images/Flickr8k_Dataset/'
img = glob.glob(args.base_dir +"/all_images/Flicker8k_Dataset/*.jpg")
train_images_file = args.base_dir +'/all_captions/Flickr_8k.trainImages.txt'
test_images_file = args.base_dir +'/all_captions/Flickr_8k.testImages.txt'
filenameGlove = args.base_dir +'/glove/glove.6B.200d.txt'
##############################################

file = open(filename_token , 'r')
doc = file.read()
################################################
descriptions = dict()
concat_descriptions(doc , descriptions)
clean_descriptions(descriptions)

vocabulary = set()
vocabulary = make_vocabulary(vocabulary , descriptions)

train = make_train_list(fname_trainImage)
train_descriptions = load_clean_descriptions(train , descriptions)
print(train_descriptions['937559727_ae2613cee5'])

vocab = make_vocab(train_descriptions)
######################################################
train_img = make_train_image_array(train_images_file,img)
test_img = make_test_image_array(test_images_file,img)
########################################################
model_new = inception_model()
#########################################################



encoding_train = {}
encoding_test = {}

for img in train_img:
    encoding_train[img[len('./all_images/Flickr8k_Dataset/'):]] = encode(img,model_new)
print(len(encoding_train ))

with open("encoded_train_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)


for img in test_img:
    encoding_test[img[len('./all_images/Flickr8k_Dataset/'):]] = encode(img,model_new)
print(len(encoding_test))

with open("encoded_test_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)

train_features = load(open("encoded_train_images.pkl", "rb"))
print('Photos: train=%d' % len(train_features))
test_features = load(open("encoded_test_images.pkl", "rb"))
print('Photos: train=%d' % len(test_features))
#############################################################
ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab :
  wordtoix[w] = ix
  ixtoword[ix] = w
  ix += 1
vocab_size = len(ixtoword) + 1
###############################################################

embeddings_index = load_embedding_index(filenameGlove)
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = args.em_dem
embedding_matrix = np.zeros((vocab_size,embedding_dim))
for word , i in wordtoix.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None :
    embedding_matrix[i] = embedding_vector

################################################################

epochs = args.epochs
num_photos_per_batch = args.num_photos_per_batch
steps = len(train_descriptions)//num_photos_per_batch
max_length = max_length(train_descriptions)


model = make_model(max_length , embedding_dim , vocab_size )
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False 
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

history = LossHistory()
for i in range(epochs):
  generator = data_generator(train_descriptions , train_features ,wordtoix , max_length ,num_photos_per_batch , vocab_size)
  model.fit_generator(generator , epochs =epochs ,steps_per_epoch = steps ,verbose =1,callbacks=[history])
  model.save('model_weigths'+str(i)+'.h5')