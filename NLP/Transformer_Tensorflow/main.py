import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import os
import argparse

from dataloader import Load_Data
from model import Transformer, create_masks
from evaluate import translate

#### PARSE ARGUMENTS ####

parser = argparse.ArgumentParser()

parser.add_argument('--EPOCHS', type = int, default = 20, help = "No of training epochs ")
parser.add_argument('--num_layers', type = int, default = 6, help = "No of layers of encoder and decoder ")
parser.add_argument('--d_model', type = int, default = 512, help = "dimension")
parser.add_argument('--dff', type = int, default = 2048, help = "dimension")
parser.add_argument('--num_heads', type = int, default = 8, help = "No of attention heads ")
parser.add_argument('--BUFFER_SIZE', type = int, default = 20000, help = "Buffer size ")
parser.add_argument('--BATCH_SIZE', type = int, default = 64, help = "Batch size ")
parser.add_argument('--MAX_LENGTH', type = int, default = 40, help = "Maximum allowable length of input and output sentences")
parser.add_argument('--dropout_rate', type = float, default = 0.1, help = "Dropout rate ")
parser.add_argument('--beta_1', type = float, default = 0.9, help = "Exponential decay rate for 1st moment")
parser.add_argument('--beta_2', type = float, default = 0.98, help = "Exponential decay rate for 2nd moment")
parser.add_argument('--input', type = str, default = '.', help = "Input sentence in portuguese")
parser.add_argument('--real_translation', type = str, default = '.', help = "Real translation of input sentence in English")
parser.add_argument('--outdir', type = str, default = '.', help = "Directory in which to store data")
parser.add_argument('--plot', type = str, default = 'decoder_layer1_block2', help = "Decoder layer and block whose attention weights are to be plotted")

args = parser.parse_args()

#### SET HYPERPARAMETERS ####

EPOCHS = args.EPOCHS
num_layers = args.num_layers
d_model = args.d_model
dff = args.dff
num_heads = args.num_heads
BUFFER_SIZE = args.BUFFER_SIZE
BATCH_SIZE = args.BATCH_SIZE
MAX_LENGTH = args.MAX_LENGTH
dropout_rate = args.dropout_rate
beta_1 = args.beta_1
beta_2 = args.beta_2

input_sentence = args.input
real_translation = args.real_translation
plot = args.plot

#### SETUP INPUT PIPELINE ####

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, test_examples = examples['train'], examples['test']

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2

dataloader = Load_Data(MAX_LENGTH,tokenizer_en,tokenizer_pt)

train_dataset = train_examples.map(dataloader.tf_encode)
train_dataset = train_dataset.filter(dataloader.filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = test_examples.map(dataloader.tf_encode)
test_dataset = test_dataset.filter(dataloader.filter_max_length).padded_batch(BATCH_SIZE)

#### OPTIMIZER ####

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1, beta_2=beta_2, 
                                     epsilon=1e-9)

#### LOSS AND METRICS ####

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

#### CREATE TRANSFORMER ####

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

#### CHECKPOINTING ####

checkpoint_dir = os.path.join(args.outdir, "training_checkpoints")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

#### TRAINING ####                       

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, predictions)

def test_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  predictions, _ = transformer(inp, tar_inp, 
                                 False, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
  loss = loss_function(tar_real, predictions)  
  
  test_loss(loss)
  test_accuracy(tar_real, predictions)

for epoch in range(EPOCHS):
  start = time.time()
  
  for (batch, (inp, tar)) in enumerate(train_dataset):
    train_step(inp, tar)
    
    if batch % 50 == 0:
      print ('Epoch {} Batch {} Train_loss {:.4f} Train_accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
  for (batch, (inp, tar)) in enumerate(test_dataset):
    test_step(inp, tar)
    
    if batch % 50 == 0:
      print ('Epoch {} Batch {} Test_loss {:.4f} Test_accuracy {:.4f}'.format(
          epoch + 1, batch, test_loss.result(), test_accuracy.result()))
      
  if (epoch) % 5 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)
    print ('Saving checkpoint for epoch {}'.format(epoch+1))
    
  print ('Epoch {} Train_loss {:.4f} Train_accuracy {:.4f} Test_loss {:.4f} Test_accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result(),test_loss.result(),test_accuracy.result()))

  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

  train_loss.reset_states()
  test_loss.reset_states()
  train_accuracy.reset_states()
  test_accuracy.reset_states()

#### TRANSLATE PORTUGUESE TO ENGLISH ####

translate(input_sentence,tokenizer_en,tokenizer_pt,MAX_LENGTH,transformer,plot=plot)
print ("Real translation: {}".format(real_translation))
