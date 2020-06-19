
import string
import pickle
import nltk
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from dataloader import load_file, load_all_descriptions, save_descriptions, load_dataset, load_selected_captions
from utils import get_vocabulary, extract_features, create_tokenizer, create_sequences
import requests
from io import BytesIO
from zipfile import ZipFile 
from model import vgg_model, inception_model, rnn_cnn_model
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type = int, default = 20, help = "No of EPOCHS: default 20 ")
parser.add_argument('--batch_size', type = int, default = 256, help = "Batch size, default 256")
parser.add_argument('--optimizer', type = str, default = "RMSprop", help = "Optimizer, default RMSprop")
parser.add_argument('--model', type = str, default = "inception", help = "Image features extraction model to be used, default InceptionV3 ")
args = parser.parse_args()


"""
# Downloading Dataset

print("Downloading fickr8k dataset, it may take some minutes depending upon your internet connection")
url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
request1 = requests.get(url)
file1 = ZipFile(BytesIO(request1.content))
file1.extractall()

url2 = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
request2 = requests.get(url)
file2 = ZipFile(BytesIO(request2.content))
file2.extractall()

"""

# Loading all captions, getting vocabulary 

captions = load_file('Flickr8k.token.txt')
all_descriptions = load_all_descriptions(captions)
vocab, all_descriptions = get_vocabulary(all_descriptions)



# Saving captions 

save_descriptions(all_descriptions, "final_captions.txt")

entire_dataset = load_file("final_captions.txt")



# getting model 

if args.model == "inception":
	model = inception_model()
else:
	model = vgg_model()



# extracting features from model for all training images and saving it

train_features = extract_features(model, 'Flickr_8k.trainImages.txt', args.model)
pickle.dump(train_features, open('train_final_features.pkl', 'wb'))



# getting captions for training images 

train_dataset = load_dataset("Flickr_8k.trainImages.txt")
max_len, train_captions_dect = load_selected_captions(train_dataset, entire_dataset)


# creating tokenizer and saving it for later use

tokenizer = create_tokenizer(train_captions_dect)
vocab_size = len(tokenizer.word_index) + 1
dump(tokenizer, open('flickr_tokenizer.pkl', 'wb'))


# load trained captions

train_features = pickle.load(open("train_final_features.pkl", 'rb'))


# getting rnn_cnn_model

if args.model == "inception":
	rc_model = rnn_cnn_model(2048, max_len, vocab_size, args.optimizer)
else:
	rc_model = rnn_cnn_model(4096, max_len, vocab_size, args.optimizer)


# getting data to train

in_img, in_seq, out_word = create_sequences(tokenizer, max_len, train_captions_dect , train_features, vocab_size)



#fitting model

tf.config.experimental_run_functions_eagerly(True)
rc_model.fit([in_img, in_seq], out_word, batch_size=args.batchsize, epochs=args.epochs, verbose=1)


#saving weights

rc_model.save_weights('rc_model_weights.h5')


