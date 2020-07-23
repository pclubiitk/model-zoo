from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence, image
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import string
import pickle
import nltk
import os
nltk.download('punkt')


def get_vocabulary(all_descriptions):
    
    table = str.maketrans('', '', string.punctuation)
    vocab = set()
    for key, desc_list in all_descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] =  ' '.join(desc)
            vocab.update(desc_list[i].split())
            
    return vocab, all_descriptions


def preprocess_image_vgg(path):
    
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x[:, :, :, ::-1] 
    x[:, :, :, 0] -= 103.939 
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
    
    return x 


def preprocess_image_inception(path):
    
    img = image.load_img(path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.
    x -= 0.5
    x *= 2.
    
    return x 


def extract_features(model, path, architecture):
    
    feature_dict = {}
    with open(path, "r") as file:
        data = file.read()
    data = data.split('\n')
    file.close()
    
    for i in range(len(data)):
        if data[i] != "":
            image_path = 'Flicker8k_Dataset/' + data[i]

            if architecture == "inception":
               imag = preprocess_image_inception(image_path)
            else:
               imag = preprocess_image_vgg(image_path)

           
            image_id = data[i].split('.')[0]
            pred = model.predict(imag)
            pred = np.reshape(pred, pred.shape[1])
            feature_dict[image_id] = pred

    return feature_dict


def create_tokenizer(caption_dect):
    
    desc = list()
    for key in caption_dect.keys():
        [desc.append(d) for d in caption_dect[key]]
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc)
    
    return tokenizer


def create_sequences(tokenizer, max_length, t_descriptions, feature, vocab_size, number):
    images, captions, next_word = list(), list(), list()
    
    count = 0
    for key, desc_list in t_descriptions.items():
        
        for desc in desc_list:

            seq = tokenizer.texts_to_sequences([desc])[0]

            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                images.append(feature[key])
                captions.append(in_seq)
                next_word.append(out_seq)
            count=count+1
            if count==number:
                return np.array(images), np.array(captions), np.array(next_word)

