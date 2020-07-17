import os
import torch
import pickle
import argparse
from PIL import Image
#------------------------
import torch
import torch.nn as nn
from cnn_model import get_CNN
from decoder import RNN
from vocab import Vocabulary
from torchvision import transforms
from dataloader import DataLoader, shuffle_data
#--------------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',type=str,help="Image filename.")
    parser.add_argument('epoch',type=int,help="Number of epochs model has been trained for.")
    parser.add_argument('model_dir',type=str,help="Saved model directory, which has name of format: model + current_datetime.")
    parser.add_argument('-model',type=str,default='resnet18',help="Encoder CNN architecture.Default: 'resnet18', other option is 'inception' (Inception_v3).")
    parser.add_argument('-test_dir',type=str,default='test',help="Test dataset directory name, default: 'test'.")

    args = parser.parse_args()
    print(args)
    model_name = args.model
    model_dir = args.model_dir

    f = open(os.path.join(model_dir, 'vocab.pkl'), 'rb')
    vocab = pickle.load(f)

    transform = transforms.Compose([transforms.Resize((224, 224)),
	                                transforms.ToTensor(),
	                                transforms.Normalize((0.5, 0.5, 0.5),
	                                                     (0.5, 0.5, 0.5))
	                                ])
    image = Image.open(os.path.join(args.test_dir,args.filename))
    #image.show()
    image = transform(image)
    vocab_size = vocab.index
    embedding_dim = 512
    hidden_dim = 512

    cnn = get_CNN(architecture= model_name, embedding_dim=embedding_dim)
    lstm = RNN(embedding_dim=embedding_dim,hidden_dim=hidden_dim,vocab_size=vocab_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn.to(device)
    lstm.to(device)
    image = image.unsqueeze(0)
    image = image.to(device)

    cnn_filename = 'epoch_' + str(args.epoch) + '_cnn.pkl'
    lstm_filename = 'epoch_' + str(args.epoch) + '_lstm.pkl'

    cnn.load_state_dict(torch.load(os.path.join(model_dir, cnn_filename)))
    lstm.load_state_dict(torch.load(os.path.join(model_dir, lstm_filename)))

    cnn_output = cnn(image)
    ids_list = lstm.greedy(cnn_output)
    print(vocab.get_sentence(ids_list))
