import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding, Concatenate, Input
import h5py

#############################################

def Word2Vec(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    
    # Text model

    w2v_input = Input((seq_length,))
    w2v_embed = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=seq_length,
                          weights=[embedding_matrix],trainable=False)(w2v_input)
    w2v_lstm1 = LSTM(512, input_shape=(seq_length, embedding_dim),return_sequences=True)(w2v_embed)
    w2v_drop1 = Dropout(dropout_rate)(w2v_lstm1)
    w2v_lstm2 = LSTM(512, return_sequences=False)(w2v_drop1)
    w2v_drop2 = Dropout(dropout_rate)(w2v_lstm2)
    w2v_dense = Dense(1024, activation='tanh')(w2v_drop2)

    model = Model(w2v_input, w2v_dense)
    return model

#############################################

def FromVGG(dropout_rate):

    #Image model
    vgg_input = Input((4096,))
    vgg_dense = Dense(1024, activation='tanh')(vgg_input)

    model = Model(vgg_input, vgg_dense)
    return model

##############################################

def VQA(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    
    # VQA model
    vgg_model = FromVGG(dropout_rate)
    lstm_model = Word2Vec(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    
    concat = Concatenate()([vgg_model.output, lstm_model.output])
    drop1 = Dropout(dropout_rate)(concat)
    dense1 = Dense(1000, activation='tanh')(drop1)
    drop2 = Dropout(dropout_rate)(dense1)
    dense2 = Dense(num_classes, activation='softmax')(drop2)

    model = Model([vgg_model.input, lstm_model.input], dense2)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

