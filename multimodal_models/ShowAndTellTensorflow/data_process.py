from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


# Loading all the captions data into a dictionary with the image_name as key
# Enter the path of "Flickr8k.token.txt" present in Dataset in "caps_path" variable
caps_path = "../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt"
f = open(caps_path, 'r')
caps = f.read()

# For easy preprocessing of data, let us store it in a dictionary
captions_dict = dict()

for line in caps.split('\n'):
    txt = line.split(" ")
    img_name = txt[0].split('#')[0]

    if img_name not in captions_dict.keys():
        captions_dict[img_name] = list()

    # Appending the start and end tokens in the captions while loading them
    captions_dict[img_name].append("startseq " + " ".join(txt[1:]) + " endseq")


# Enter the path of file "Flickr_8k.trainImages.txt" present in Dataset, in "train_image_names_path" variable
train_image_names_path = "../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt"
g = open(train_image_names_path, 'r')
train_image_names = g.read()
train_image_names = train_image_names.split('\n')

train_image_names.remove('')

# Store the captions of training images in a different list
train_caps = []
for img in train_image_names:
    train_caps.append(captions_dict[img])

# Since each image has 5 captions, to get a 1 D array of captions, flattening is required
train_flat = [cap for caps in train_caps for cap in caps]

# Converting the text data into sequence data, which can be padded and fed to neural network
tokenizer = Tokenizer(num_words=5000,
                      oov_token="<unk>",
                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_flat)
word_index = tokenizer.word_index
index_word = dict((word_index[k], k) for k in word_index.keys())

train_tokens = [tokenizer.texts_to_sequences(caps) for caps in train_caps]


# This is a custom function that picks a caption at random out 5, for any given image index
def one_of_five_caps(temp):
    caps1 = []
    for x in temp:
        y = np.random.choice(5)
        caps1.append(train_tokens[x][y])
    return caps1
