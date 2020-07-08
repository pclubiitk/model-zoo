import keras
import numpy
from numpy import array
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras import Input, layers
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense,Input , Dropout , RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.layers.merge import add
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
def data_generator(descriptions , photos , wordtoix , max_length , num_photos_per_batch,vocab_size):
  x1,x2 ,y = list(), list() , list()
  n = 0
  while( True):
    for key , desc_list in descriptions.items():
      n+=1 
      photo = photos['/'+key+'.jpg']
      for desc in desc_list:
        seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]

        for i in range(1 , len(seq )):
          in_seq ,out_seq = seq[:i] , seq[i]
          in_seq  = pad_sequences([in_seq], maxlen = max_length)[0]
          out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
          
          x1.append(photo)
          x2.append(in_seq)
          y.append(out_seq)

        if n == num_photos_per_batch : 
          yield [[array(x1) , array(x2) ],array(y)]
          x1,x2 ,y = list(), list() , list()
          n= 0


def inception_model():
    
    model = InceptionV3(weights = 'imagenet')
    model.summary()

    model_new = Model(model.inputs , model.layers[-2].output)
    print(model_new.summary())
    return model_new


class LossHistory(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.loss = []
        self.acc=[]

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs['loss'])  
        self.acc.append(logs['accuracy'])
    



def make_model(max_length , embedding_dim , vocab_size ):
 input1 = Input(shape =(2048,))
 fe1 = Dropout(0.5)(input1)
 fe2 = Dense(256 ,activation = 'relu')(fe1)

 input2 = Input(shape = (max_length,))
 se1 = Embedding(vocab_size , embedding_dim , mask_zero = True )(input2)
 se2 = Dropout(0.5)(se1)
 se3 = LSTM(256)(se2)

 decoder1  = add([fe2 , se3])
 decoder2  = Dense(256 , activation = 'relu')(decoder1)
 outputs = Dense(vocab_size , activation = 'softmax')(decoder2)

 model = Model(inputs = [input1 ,input2] ,outputs = outputs)
 model.summary()
 return model




