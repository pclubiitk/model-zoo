from PIL import Image
from keras.preprocessing.sequence import pad_sequences
import string
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
from numpy import array
#%matplotlib inline
from keras.applications.inception_v3 import preprocess_input
def concat_descriptions(doc , descriptions):
 for line in doc.split('\n'):
  tokens = line.split()
  if (len(tokens)== 0): break 
  image_id , image_desc = tokens[0] , tokens[1:]
  image_id = image_id.split('.')[0]

  image_desc = ' '.join(image_desc)
  if image_id not in descriptions:
    descriptions[image_id] = list()
  descriptions[image_id].append(image_desc)


def clean_descriptions(descriptions):
 table = str.maketrans('','',string.punctuation)
 for key , desc_list in descriptions.items():
  for i in range(len(desc_list)):
    desc = desc_list[i]
    desc = desc.split()
    desc = [word.lower() for word in desc]
    desc = [word.translate(table) for word in desc ]
    desc = [word for word in desc if len(word)>1 ]
    desc = [word for word in desc if word.isalpha() ]
    desc_list[i] = ' '.join(desc)



def make_vocabulary(vocabulary , descriptions):
  for key in descriptions.keys():
    [vocabulary.update(d.split()) for d in descriptions[key]]
  print('original voabulary size: %d'% len(vocabulary) )
  return vocabulary

def make_vocab(train_descriptions):
 all_train_captions = []
 for key ,val in train_descriptions.items():
  for cap in val :
    all_train_captions.append(cap)
 print(len(all_train_captions))
 word_count_threshold = 10
 word_counts = {}
 nsents = 0
 for sent in all_train_captions:
  nsents += 1
  for w in sent.split(' '):
    word_counts[w] = word_counts.get(w,0)+1

 vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold ]
 print('preprocessed words %d '% len(vocab))
 return vocab

def make_train_list(fname):
 file = open(fname , 'r')
 doc_ = file.read()
 train = list()
 for line in doc_.split('\n'):
  if(len(line)<1):
    continue
  identifier = line.split('.')[0]
  train.append(identifier)
 train = set(train)
 print('dataset : %d' % len(train))
 return train



def make_train_image_array(train_images_file,img):
 train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
 print('train_image_example :', img[0].split('/')[-1])
 train_img = []
 for i in img:
    if i.split('/')[-1] in train_images: 
        train_img.append(i)
 print(len(train_img))
 return train_img 

def make_test_image_array(test_images_file,img):
 test_images = set(open(test_images_file, 'r').read().strip().split('\n'))
 test_img = []

 for i in img: 
    if i.split('/')[-1] in test_images: 
        test_img.append(i)
 print(len(test_img))
 return test_img

def load_clean_descriptions(dataset , descriptions1 ):
  doc = descriptions1
  descriptions_ = dict()
  for image_id, image_desc in doc.items():
    
    for val in image_desc :
      if image_id in dataset:
        if image_id not in descriptions_:
          descriptions_[image_id] = list()
			  
        desc = 'startseq ' + ' '.join(val.split()) + ' endseq'
        descriptions_[image_id].append(desc)
  return descriptions_

def load_embedding_index(filenameGlove):
    embeddings_index = {}
    f = open(filenameGlove , encoding = "utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image,model_new):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

def to_lines(descriptions):
   all_desc = list()
   for key in descriptions.keys():
     [all_desc.append(d) for d in descriptions[key]]
   return all_desc

def max_length(descriptions):
  lines = to_lines(descriptions)
  return max(len(d.split()) for d in lines)

def greedy(photo):
  in_text = 'startseq'
  for i in range(max_length):
    sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
    sequence = pad_sequences([sequence], maxlen=max_length)
    yhat = model.predict([photo , sequence], verbose = 0)
    yhat = np.argmax(yhat)
    word = ixtoword(yhat)
    in_text += ' '+word
    if word == 'endseq':break
  final = in_text.split()
  final = final[1:-1]
  final = ' '.join(final)
  return final

