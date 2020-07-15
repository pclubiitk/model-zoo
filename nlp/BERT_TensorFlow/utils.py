import tensorflow as tf
import tensorflow_datasets as tfds
##############################################

def get_segments(tokens, max_seq_length):
  """Segments: 0 for the first sequence, 1 for the second"""
  if len(tokens)>max_seq_length:
    raise IndexError("Token length more than max seq length!")
  segments = []
  current_segment_id = 0
  for token in tokens:
      segments.append(current_segment_id)
      if token == "[SEP]":
          current_segment_id = 1
  return segments + [0] * (max_seq_length - len(tokens))

#################################################

def get_ids(tokens, tokenizer, max_seq_length):
  """Token ids from Tokenizer vocab"""
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
  return input_ids

#################################################

def create_padding_mask(seq):
  """Need to add axes to create padding masks"""
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  return seq[:, tf.newaxis, tf.newaxis, :] 

###################################################
def gelu(x):
  return x * 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
 
###################################################

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights."""
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e4)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

#################################################

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, hidden_size, warmup_steps=10000):
    super(CustomSchedule, self).__init__()
    
    self.hidden_size = hidden_size
    self.hidden_size = tf.cast(self.hidden_size, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.hidden_size) * tf.math.minimum(arg1, arg2)
    
##################################################

def encode_examples(ds, tokenizer, max_length, limit=-1):
  # prepare list, so that we can build up final TensorFlow dataset from slices.
  input_ids_list = []
  token_type_ids_list = []
  attention_mask_list = []
  label_list = []
  if (limit > 0):
      ds = ds.take(limit)
    
  for review, label in tfds.as_numpy(ds):
    bert_input = tokenizer.encode_plus(
                        review.decode(),                      
                        add_special_tokens = True, # add [CLS], [SEP]
                        max_length = max_length, # max length of the text that can go to BERT
                        pad_to_max_length = True, # add [PAD] tokens
                        return_attention_mask = True, # add attention mask to not focus on pad tokens
                        truncation = True
              )
  
    input_ids_list.append(bert_input['input_ids'])
    token_type_ids_list.append(bert_input['token_type_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    label_list.append([label])
  return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)
  
###################################################

def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label
  
###################################################
