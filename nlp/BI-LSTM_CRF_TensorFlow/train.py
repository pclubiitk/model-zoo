from utils import modifying
import tensorflow as tf
from model import NerModel
import pandas as pd
import tensorflow_addons as tf_ad
import os
import numpy as np
import datetime
import argparse
from dataloader import load_data

######################################################################################################

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--output_dir", type=str, default="checkpoints/",help="output_dir")
parser.add_argument("--max_len",type=int,default=50,help="max_len")
parser.add_argument("--batch_size", type=int, default=64,help="batch_size")
parser.add_argument("--hidden_num", type=int, default=512,help="hidden_num")
parser.add_argument("--embedding_size", type=int, default=300,help="embedding_size")
parser.add_argument("--embedding_file", type=str, default=None,help="embedding_file")
parser.add_argument("--epoch", type=int, default=50,help="epoch")
parser.add_argument("--lr", type=float, default=1e-3,help="lr")
parser.add_argument("--require_improvement", type=int, default=100,help="require_improvement")
args = parser.parse_args()

########################################################################################################
# directory for tensorboard

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

#########################################################################################################
#Preprocessing Data
data_train = load_data()
getter = modifying(data_train)
getter.get_next()

tag2id,n_tags,word2id,n_words = getter.indexing()
text_sequences,label_sequences = getter.padding(args.max_len,word2id,tag2id) # making length of all sentences to be equal

train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences, label_sequences)) # converting to tensorflow dataset
train_dataset = train_dataset.shuffle(len(text_sequences)).batch(args.batch_size, drop_remainder=True)

print("hidden_num:{}, vocab_size:{}, label_size:{}".format(args.hidden_num, len(word2id), len(tag2id)))

#######################################################################################################

model = NerModel(hidden_num = args.hidden_num, vocab_size = len(word2id)+1, label_size= len(tag2id), embedding_size = args.embedding_size)
optimizer = tf.keras.optimizers.Adam(args.lr)


ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(args.output_dir))
ckpt_manager = tf.train.CheckpointManager(ckpt,args.output_dir,checkpoint_name='model.ckpt',max_to_keep=3)

#########################################################################################################

# @tf.function
def train_one_step(text_batch, labels_batch):
  with tf.GradientTape() as tape:
      logits, text_lens, log_likelihood = model(text_batch, labels_batch,training=True)
      loss = - tf.reduce_mean(log_likelihood)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss,logits, text_lens


def get_acc_one_step(logits, text_lens, labels_batch):
    paths = []
    accuracy = 0
    for logit, text_len, labels in zip(logits, text_lens, labels_batch):
        viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
        paths.append(viterbi_path)
        correct_prediction = tf.equal(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                 dtype=tf.int32),
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
                                 dtype=tf.int32)
        )
        accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
    accuracy = accuracy / len(paths)
    return accuracy

#####################################################################################################################
#training loop

best_acc = 0
step = 0
acc=0
for epoch in range(args.epoch):
    for _, (text_batch, labels_batch) in enumerate(train_dataset):
        step = step + 1
        loss, logits, text_lens = train_one_step(text_batch, labels_batch)
        
        if step % 20 == 0:
            accuracy = get_acc_one_step(logits, text_lens, labels_batch)
            acc = accuracy
            print('epoch %d, step %d, loss %.4f , accuracy %.4f' % (epoch+1, step, loss, accuracy))
            if accuracy > best_acc:
              best_acc = accuracy
              ckpt_manager.save()
        
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=step)
            tf.summary.scalar('accuracy', acc, step=step)

#######################################################################################################################
