import os                          # Standard imports
import torch
import time
import pickle
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
#-------------------------
from cnn_model import get_CNN          # Own modules imports
from decoder import RNN
from vocab import Vocabulary
from torchvision import transforms
from dataloader import DataLoader, shuffle_data
#------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir',type=str,help="Saved model directory, which has name of format: model + current_datetime.")
    parser.add_argument('--model',type=str,default='resnet18',help="Default: 'resnet18', other option is 'inception' (Inception_v3).")
    parser.add_argument('--dir', type = str, default = 'dev',help="Development Directory path, default: 'dev'")
    parser.add_argument('--save_epoch', type=int, default=2,help="Epochs after which trained model has saved checkpoint, default : 2")
    parser.add_argument('--num_epoch', type=int, default=10,help="Number of epochs model was trained for, default : 10")

    args = parser.parse_args()
    print(args)

    dir_path = args.dir
    model_name = args.model
    num_epoch = args.num_epoch
    save_epoch = args.save_epoch
    embedding_dim = 512
    hidden_dim = 512
    model_name = args.model
    model_dir = args.model_dir

    f = open(os.path.join(model_dir, 'vocab.pkl'), 'rb')
    vocab = pickle.load(f)
    vocab_size = vocab.index
    
    transform = transforms.Compose([transforms.Resize((224, 224)),
	                                transforms.ToTensor(),
	                                transforms.Normalize((0.5, 0.5, 0.5),
	                                                     (0.5, 0.5, 0.5))
	                                ])
    dataloader = DataLoader(dir_path, vocab, transform)
    data = dataloader.gen_data()
    print(dir_path + ' loaded!')

    criterion = nn.CrossEntropyLoss()
    cnn = get_CNN(architecture = model_name, embedding_dim = embedding_dim)
    lstm = RNN(embedding_dim = embedding_dim, hidden_dim = hidden_dim,
	           vocab_size = vocab_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn.to(device)
    lstm.to(device)

    tb = SummaryWriter()
    loss_list = []

    for epoch in range(0, num_epoch, save_epoch):
        cnn_filename = 'epoch_' + str(epoch) + '_cnn.pkl'
        lstm_filename = 'epoch_' + str(epoch) + '_lstm.pkl'
        cnn.load_state_dict(torch.load(os.path.join(model_dir, cnn_filename)))
        lstm.load_state_dict(torch.load(os.path.join(model_dir, lstm_filename)))

        cnn.eval()
        lstm.eval()
        total_loss = 0
        images, captions = data
        num_captions = len(captions)

		# start = time.time()
        with torch.no_grad():
            for i in range(num_captions):
                image_id = images[i]
                image = dataloader.get_image(image_id)
                image = image.unsqueeze(0)
                
                image = image.to(device)
                caption = torch.LongTensor(captions[i]).to(device)
                
                caption_train = caption[:-1]  # remove <end>
                
                loss = criterion(lstm(cnn(image), caption_train), caption)
                
                loss_list.append(loss)
                total_loss += loss.item()
        tb.add_scalar('Validation Loss', total_loss, epoch)
        avg_loss = torch.mean(torch.Tensor(loss_list))
        print('%d %f' %(epoch, avg_loss))

torch.save(loss_list, os.path.join(model_dir, 'validation_loss.pt'))
