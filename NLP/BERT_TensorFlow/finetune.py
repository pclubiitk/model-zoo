import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer
from utils import encode_examples
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type = int, default = 2, help = "Number of epochs in finetuning, default 2")
parser.add_argument('--lr', type = float, default = 2e-5, help = "Learning rate for finetune, default 2e-5")
parser.add_argument('--batch_size', type = int, default = 4, help = "Batch Size, default 32 (WARN! using batch size > 32 on just one GPU can cause OOM) ")
parser.add_argument('--max_length', type = int, default = 128, help = "Maximum length of input string to bert, default 128")
parser.add_argument('--train_samples', type = int, default = 25000, help = "Number of training samples, default (max): 25000")
parser.add_argument('--test_samples', type = int, default = 25000, help = "Number of test samples, default (max): 25000")

args = parser.parse_args()

######## Define Tokenizer ################
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

######## Get IMDB dataset from tfds ######
(ds_train, ds_test), ds_info = tfds.load('imdb_reviews', 
          split = (tfds.Split.TRAIN, tfds.Split.TEST),
          as_supervised=True,
          with_info=True)
print('info', ds_info)

######## Encode dataset in bert format ####
# train dataset
ds_train_encoded = encode_examples(ds_train, tokenizer, args.max_length, args.train_samples).shuffle(25000).batch(args.batch_size)
# test dataset
ds_test_encoded = encode_examples(ds_test, tokenizer, args.max_length, args.test_samples).shuffle(25000).batch(args.batch_size)

######### Define bert model ###############
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

######### Define optimizer ################
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, epsilon=1e-08)

######### Define Loss function and metrics #########
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

######### Compile model ###################
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

######### Get predictions #################
history = model.fit(ds_train_encoded, epochs=args.epochs, validation_data=ds_test_encoded)
