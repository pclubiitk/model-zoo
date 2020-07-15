import os                       # Standard imports
import argparse
import nltk
import time
import datetime
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
# ---------------------------------
from vocab import Vocabulary    # Own modules import
from decoder import RNN
from preprocess import load_captions
from cnn_model import get_CNN
from dataloader import DataLoader, shuffle_data
# ---------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model',type=str,default='resnet18',help="Encoder CNN architecture.Default: 'resnet18', other option is 'inception' (Inception_v3). Model dir is automatically saved with name of model + current_datetime.")
    parser.add_argument('-dir',type=str,default='train', help="Training Directory path, default: 'train'")
    parser.add_argument('-save_epoch',type=int,default=2,help="Epochs after which model saves checkpoint, default : 2")
    parser.add_argument('-learning_rate',type=float,default=1e-3,help='Adam optimizer learning rate, default : 1e-3 (0.001)')
    parser.add_argument('-num_epoch',type=int,default=10,help="Number of epochs, default : 10")
    parser.add_argument('-hidden_dim',type=int,default=512,help='Dimensions in hidden state of LSTM decoder, default : 512')
    parser.add_argument('-embedding_dim',type=int,default=512,help='Dimensions of encoder output, default : 512')

    args = parser.parse_args()
    print(args)
    
    model_dir = args.model + str(datetime.datetime.now())    # CNN
    model_name = args.model
    train_dir = args.dir
    learning_rate = args.learning_rate
    num_epoch = args.num_epoch
    save_epoch = args.save_epoch
    threshold = 5

    nltk.download('punkt') #: uncomment if punkt is not found

    if os.path.exists(model_dir):
        print("Directory '{model_name}' already exists. Assuming vocab.pkl already dumped. If not, delete the empty directory '{model_name}' and start the program.".format(model_name=model_name))
        f = open(os.path.join(model_dir, 'vocab.pkl'), 'rb')
        vocab = pickle.load(f)
    else:
        captions_dict = load_captions(train_dir)
        vocab = Vocabulary(captions_dict, threshold)
        os.mkdir(model_dir)
        print("Directory '{model_name}' created to dump vocab.pkl.".format(model_name=model_name))
        with open(os.path.join(model_dir, 'vocab.pkl'), 'wb') as f:
            pickle.dump(vocab, f)
            print('Dictionary Dumped !')

    transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])

    dataloader = DataLoader(train_dir, vocab, transform)
    data = dataloader.gen_data()
    print(train_dir + ' loaded !')

    vocab_size = vocab.index

    cnn = get_CNN(architecture = model_name, embedding_dim = args.embedding_dim)
    lstm = RNN(embedding_dim = args.embedding_dim, hidden_dim = args.hidden_dim, vocab_size = vocab_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cnn.to(device)
    lstm.to(device)

    criterion = nn.CrossEntropyLoss()
    params = list(cnn.linear.parameters()) + list(lstm.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    tb = SummaryWriter()
    loss_list = []
    print('Training Starts!')

    for epoch in range(num_epoch):
        shuffled_imgs, shuffled_captions = shuffle_data(data, seed = epoch)
        num_captions = len(shuffled_captions)

        total_loss = 0
        start = time.time()

        for i in range(num_captions):
            image_id = shuffled_imgs[i]
            image = dataloader.get_image(image_id)
            image = image.unsqueeze(0)

            image = image.to(device)
            caption = torch.LongTensor(shuffled_captions[i]).to(device)

            caption2train = caption[:-1]
            cnn.zero_grad()
            lstm.zero_grad()

            cnn_output = cnn(image)
            lstm_output = lstm(cnn_output, caption2train)
            loss = criterion(lstm_output, caption)
            loss.backward()
            optimizer.step()

            loss_list.append(loss)
            total_loss += loss.item()

        tb.add_scalar('Training Loss', total_loss, epoch)

        end = time.time()
        avg_loss = torch.mean(torch.Tensor(loss_list))

        print('Epoch : %d , Avg_loss = %f, Time = %.2f mins'%(epoch, avg_loss, ((end-start)/60)))

        if epoch % save_epoch == 0:
            torch.save(cnn.state_dict(), os.path.join(model_dir, 'epoch_%d_cnn.pkl'%(epoch)))
            torch.save(lstm.state_dict(), os.path.join(model_dir, 'epoch_%d_lstm.pkl'%(epoch)))

torch.save(loss_list, os.path.join(model_dir, 'training_loss.pt'))
