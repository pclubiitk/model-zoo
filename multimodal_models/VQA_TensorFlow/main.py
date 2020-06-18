import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import os
import argparse
from model import VQA
from dataloader import read_data, get_val_data, get_metadata, prepare_embeddings
from utils import get_data

############# Parsing Arguments ##################

parser = argparse.ArgumentParser()

parser.add_argument('--type', type=str, default='train', help = "Whether you want to train or validate, default train")
parser.add_argument('--base_path', default = ".", help = "Relative path to location where to download data, default '.'")
parser.add_argument('--epochs', type=int, default=10, help = "Number of epochs, default 10")
parser.add_argument('--batch_size', type=int, default=256, help = "Batch Size, default 256")
parser.add_argument('--data_limit', type=int, default=215359, help="Number of data points to feed for training, default 215359 (size of dataset)")
parser.add_argument('--weights_load', default=False, help="Boolean to say whether to load pretrained model or train new model, default False")
parser.add_argument('--weight_path', help = "Relative path to location of saved weights, default '.'")

args = parser.parse_args()

##### Setting global variables and file paths #######
seq_length = 26
embedding_dim = 300

glove_path = args.base_path + "/data/glove/glove.6B.300d.txt"
train_questions_path = args.base_path + "/data/train_ques/MultipleChoice_mscoco_train2014_questions.json"
val_annotations_path = args.base_path + "/data/val_annotations/mscoco_val2014_annotations.json"
ckpt_model_weights_filename = args.base_path + "/data/ckpts/model_weights.h5"
data_img = args.base_path + "/data/image_data/data_img.h5"
data_prepro = args.base_path + "/data/image_data/data_prepro.h5"
data_prepro_meta = args.base_path + "/data/image_data/data_prepro.json"
embedding_matrix_filename = args.base_path + "/data/ckpts/embeddings_%s.h5"%embedding_dim
save_dest = args.base_path + "/data/model/saved_model.h5"


#####################################################
def get_model(dropout_rate, model_weights_filename, weights_load):

    print("Creating Model...")
    metadata = get_metadata(data_prepro_meta)
    num_classes = len(metadata['ix_to_ans'].keys())
    num_words = len(metadata['ix_to_word'].keys())

    embedding_matrix = prepare_embeddings(num_words, embedding_dim, metadata, glove_path, train_questions_path, embedding_matrix_filename)
    model = VQA(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes)
    if (weights_load and os.path.exists(model_weights_filename)):
        print("Loading Weights...")
        model.load_weights(model_weights_filename)
    
    return model

##################################################

def train(args):
    dropout_rate = 0.5
    train_X, train_y = read_data(data_img, data_prepro, args.data_limit)    
    model = get_model(dropout_rate, args.weight_path, args.weights_load)
    checkpointer = ModelCheckpoint(filepath=ckpt_model_weights_filename,verbose=1)
    model.fit(train_X, train_y, epochs=args.epochs, batch_size=args.batch_size, callbacks=[checkpointer], shuffle="batch")
    if not os.path.exists(args.base_path + "/data/model"):
        os.makedirs(args.base_path + "/data/model")
        
    model.save_weights(save_dest, overwrite=True)

##################################################

def val():
    val_X, val_y, multi_val_y = get_val_data(val_annotations_path, data_img, data_prepro, data_prepro_meta) 
    model = get_model(0.0, args.weight_path, args.weights_load)
    print("Evaluating Accuracy on validation set:")
    metric_vals = model.evaluate(val_X, val_y)
    print("")
    for metric_name, metric_val in zip(model.metrics_names, metric_vals):
        print(metric_name, " is ", metric_val)

    # Comparing prediction against multiple choice answers
    true_positive = 0
    preds = model.predict(val_X)
    pred_classes = [np.argmax(_) for _ in preds]
    for i, _ in enumerate(pred_classes):
        if _ in multi_val_y[i]:
            true_positive += 1
    print("true positive rate: ", np.float(true_positive)/len(pred_classes))

##################################################

if __name__ == "__main__":

    get_data(args.base_path)
    if args.type == 'train':
        train(args)
    elif args.type == 'val':
        val()
