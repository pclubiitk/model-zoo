import tensorflow as tf
from keras import layers
import keras 


class EmbeddingLayer(layers.Layer):
    
  def __init__(self, units, input_dim):
    super(EmbeddingLayer, self).__init__()
    
    self.input_dim = input_dim

    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), 
                        dtype= 'float32'),
                        trainable=True, 
                        name="emb")

  def call(self, inputs):
    embedding = tf.matmul(inputs, self.w)
    return embedding

class ScoringLayer(layers.Layer):
    
  def __init__(self, units, input_dim):
    super(ScoringLayer, self).__init__()

    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                              dtype='float32'),
                          trainable=True) 
    
  def call(self, inputs):
    output = tf.matmul(inputs, self.w)
    softmax = tf.nn.softmax(output, axis=-1)
    return softmax

class Word2Vec(keras.Model):
      
  def __init__(self, units, input_dim):
    super(Word2Vec, self).__init__()

    self.embedding = EmbeddingLayer(units, input_dim)
    self.scoring = ScoringLayer(input_dim,units)

  def call(self, inputs):
    embedding = self.embedding(inputs)
    output = self.scoring(embedding)
    return outputs