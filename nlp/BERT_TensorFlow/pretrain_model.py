import tensorflow as tf
from utils import gelu, scaled_dot_product_attention
import numpy as np
##########################################

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, hidden_size, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    
    assert hidden_size % self.num_heads == 0
    
    self.depth = hidden_size // self.num_heads
    
    self.wq = tf.keras.layers.Dense(hidden_size)
    self.wk = tf.keras.layers.Dense(hidden_size)
    self.wv = tf.keras.layers.Dense(hidden_size)
    
    self.dense = tf.keras.layers.Dense(hidden_size)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, hidden_size)
    k = self.wk(k)  # (batch_size, seq_len, hidden_size)
    v = self.wv(v)  # (batch_size, seq_len, hidden_size)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.hidden_size))  # (batch_size, seq_len_q, hidden_size)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, hidden_size)
        
    return output, attention_weights

###################################################

class point_wise_feed_forward_network(tf.keras.layers.Layer):

  def __init__(self,hidden_size, dff):
    super(point_wise_feed_forward_network, self).__init__()
    
    self.dense1 = tf.keras.layers.Dense(dff)
    self.dense2 = tf.keras.layers.Dense(hidden_size)

  def call(self,x):
    
    x = self.dense1(x)
    x = self.dense2(gelu(x))

    return x

##################################################

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, hidden_size, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(hidden_size, num_heads)
    self.ffn = point_wise_feed_forward_network(hidden_size, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask = False):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, hidden_size)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, hidden_size)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, hidden_size)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, hidden_size)
    
    return out2
    
############################################

class Embeddings(tf.keras.layers.Layer):
  def __init__(self, hidden_size, input_vocab_size, max_length, rate = 0.1):
    super(Embeddings, self).__init__()

    self.hidden_size = hidden_size
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, hidden_size)
    self.pos_embedding = tf.keras.layers.Embedding(max_length, hidden_size)
    self.seg_embedding = tf.keras.layers.Embedding(2, hidden_size)

    self.dropout = tf.keras.layers.Dropout(rate)

    self.layer_norm = tf.keras.layers.LayerNormalization()
  
  def call(self, x, seg, training):

    seq_len = tf.shape(x)[1]
    seq = np.arange(seq_len)
    # adding embedding, position encoding, segment embedding.
    x = self.embedding(x)*tf.math.sqrt(tf.cast(self.hidden_size, tf.float32)) + self.pos_embedding(seq) + self.seg_embedding(seg)  # (batch_size, input_seq_len, hidden_size)
    x = self.dropout(x, training=training)
    x = self.layer_norm(x)
    return x
    
###########################################

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, hidden_size, num_heads, dff, input_vocab_size,
               max_length, rate=0.1):
    super(Encoder, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers
    
    self.embeddings = Embeddings(hidden_size, input_vocab_size, max_length, rate)
    
    self.layer_norm = tf.keras.layers.LayerNormalization()
    
    self.enc_layers = [EncoderLayer(hidden_size, num_heads, dff, rate) 
                       for _ in range(num_layers)]
        
  def call(self, x, seg, training, mask = False):

    x = self.embeddings(x, seg, training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, hidden_size)
    
##############################################

class BertModel(tf.keras.Model):
  def __init__(self, num_layers, hidden_size, num_heads, dff, input_vocab_size,
               max_length, rate=0.1):
    super(BertModel, self).__init__()

    self.hidden_size = hidden_size
    self.input_vocab_size = input_vocab_size
    self.encode = Encoder(num_layers, hidden_size, num_heads, dff, input_vocab_size,
               max_length, rate=0.1)
    self.NSPdense1 = tf.keras.layers.Dense(hidden_size, activation = 'tanh')
    self.NSPdense2 = tf.keras.layers.Dense(1)

    self.MLMdense1 = tf.keras.layers.Dense(hidden_size)
    self.layer_norm = tf.keras.layers.LayerNormalization()
    self.reverseEmbeddings = tf.keras.layers.Dense(input_vocab_size, use_bias = False)
    self.reverseEmbeddings(tf.ones((1,self.hidden_size), dtype = tf.float32))

  def call(self, x, seg, training, mask = False):

    res = self.encode(x, seg, training, mask)

    nsp = self.NSPdense1(res[:,0])
    nsp = self.NSPdense2(nsp)  # For next sentence prediction

    mlm = self.MLMdense1(res)
    mlm = self.layer_norm(gelu(mlm)) # for masked token prediction
    
    self.reverseEmbeddings.set_weights(tf.reshape(self.encode.embeddings.embedding.get_weights(), (1,self.hidden_size, self.input_vocab_size)))
    pred = self.reverseEmbeddings(mlm)
    
    return nsp, pred
    

