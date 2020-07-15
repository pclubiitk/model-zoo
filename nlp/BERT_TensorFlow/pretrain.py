import tensorflow as tf
from pretrain_preprocess import preprocess
from pretrain_model import BertModel
from transformers import BertTokenizer
import time
import argparse
from utils import CustomSchedule

parser = argparse.ArgumentParser()

parser.add_argument('--num_layers', type = int, default = 12, help = "Number of Encoder layers, default 12")
parser.add_argument('--epochs', type = int, default = 40, help = "Number of epochs in pretrain, default 40")
parser.add_argument('--hidden_size', type = int, default = 512, help = "Number of neurons in hidden feed forward layer, default 512")
parser.add_argument('--num_heads', type = int, default = 12, help = "Number of heads used in multi headed attention layer, default 12")
parser.add_argument('--max_length', type = int, default = 512, help = "Maximum token count of input sentence, default 512 (Note: if number of token exceeds max length, an error will be thrown)")
parser.add_argument('--batch_size', type = int, default = 2, help = "Batch size, default 2 (WARN! using batch size > 2 on just one GPU can cause OOM)")
parser.add_argument('--train_corpus', type = str, required = True, help = "Path to training corpus, required argument.")

############ PARSING ARGUMENTS ###########
args = parser.parse_args()

num_layers = args.num_layers
hidden_size = args.hidden_size
dff = 4 * hidden_size
num_heads = args.num_heads
max_length = args.max_length

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

########### Define Tokenizer ############
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

input_vocab_size = len(tokenizer.vocab)

########### Define Learning rate and optimzer ########
learning_rate = CustomSchedule(hidden_size)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-9) # values as in the paper

########### Define Loss function #######
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) # for NSP
sce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True) # for MLM

def loss_function(nsp, mlm, is_next, seg_input, masked):
  nsp_result = bce_loss(is_next, nsp)

  mlm_result = 0
  for i in range(len(masked)):
    seg_val = 0
    for j in range(len(masked[i])):
      if(seg_input[i][j] < seg_val):
        break
      seg_val = seg_input[i][j]
      if masked[i][j] is not 0:
        mlm_result += sce_loss(masked[i][j], mlm[i,j])

  return nsp_result + mlm_result

########### Define Metrics ##############
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
nsp_accuracy = tf.keras.metrics.BinaryAccuracy( name = 'nsp_accuracy', threshold=0.0)

########## Define Model ################
BertPretrain = BertModel(num_layers, hidden_size, num_heads, 
                         dff, input_vocab_size,
                         max_length)

########## Define Checkpoints ##########
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(model=BertPretrain,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')
  
########## Define Training step #########
def train_step(model, index_input, seg_input, mask_input, is_next, is_masked):
  
  with tf.GradientTape() as tape:
    nsp, mlm = model(index_input, training=False, seg = seg_input, mask = mask_input)
    loss = loss_function(nsp, mlm, is_next, seg_input, masked = is_masked)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  for i in range(len(is_masked)):
    seg_val = 0
    for j in range(len(is_masked[i])):
      if(seg_input[i][j] < seg_val):
        break
      seg_val = seg_input[i][j]
      if is_masked[i][j] is not 0:
        train_accuracy(is_masked[i][j], mlm[i,j])
  
  nsp_accuracy(is_next,nsp)
  
######## Get preprocessed training dataset #
train_dataset = preprocess(args.train_corpus, BATCH_SIZE, max_length, tokenizer)

##########################################

def main():
  # Train Loop
  for epoch in range(EPOCHS):
    start = time.time()
    
    train_loss.reset_states()
    train_accuracy.reset_states()
    nsp_accuracy.reset_states()
    
    for ind, batch in enumerate(train_dataset):
      input_ids, segments, masks, is_next_list, is_masked = batch
      train_step(BertPretrain, input_ids, segments, masks, is_next_list, is_masked)
      
      print ('Epoch {} batch {} BatchLoss {:.4f} MLM Accuracy {:.4f} NSP Accuracy {:.4f}'.format(epoch + 1, ind, train_loss.result(), train_accuracy.result(), nsp_accuracy.result()))
        
    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
      
    print ('Epoch {} Loss {:.4f} MLM Accuracy {:.4f} NSP Accuracy {:.4f}'.format(epoch + 1,train_loss.result(), train_accuracy.result(), nsp_accuracy.result()))
  
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    
if __name__ == '__main__':
    main()