from __future__ import print_function
import argparse
import os
import time
import datetime
from collections import Counter, defaultdict
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#-------------------------------
from glove_model import *
from utils import *
from dataloader import *
#-------------------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epoch', type=int, default=100, help='Number of epochs,default: 100')
    parser.add_argument('--save_epoch', type=int, default=10, help='Epochs after which model is saved,default: 10')

    parser.add_argument('--batch_size',type=int, default=2048,help='Batch size, default: 2048')

    parser.add_argument('--embedding_dim',type=int, default=300, help='Embedding dimension, default: 300')
    parser.add_argument('--lr',type=float,default=0.05,help='Learning rate of Adagrad optimiser, default: 0.05')
    parser.add_argument('--x_max',type=int,default=100, help='Parameter in computing weighting terms, default: 100')
    parser.add_argument('--alpha',type=float,default=0.75, help='Parameter in computing weighting terms, default: 0.75')
    parser.add_argument('--get_TSNE_plot',type=bool,default=True, help='Want to visualise high dimensional data in trained model? default: True')
    parser.add_argument('--top_k',type=int,default=300, help='Number of words you want to visualise, default: 300')

    args = parser.parse_args()
    print(args)

    model_name = "GloVe_" + str(datetime.datetime.now())
    os.mkdir(model_name)
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = GloveDataset(open(os.path.join('data', 'text8')).read(), 10000000)
    glove  = Glove(dataset._vocab_len, args.embedding_dim).to(device)

    optimizer = torch.optim.Adagrad(glove.parameters(), lr=args.lr)

    tb = SummaryWriter()
    num_batches = int(len(dataset._xij)/ args.batch_size)
    print("Number of batches : {}".format(num_batches))
    print('Training Starts...........!')

    LOSS = []
    for epoch in range(args.num_epoch):

        start = time.time()
        xbatch = 0

        for xij, i_idx, j_idx in dataset.get_batches(args.batch_size):
            xbatch += 1
            optimizer.zero_grad()

            output = glove(i_idx, j_idx)
            weights_x = weight_function(xij, args.x_max, args.alpha).to(device)
            loss = weighted_MSE_loss(weights_x, output, torch.log(xij)).to(device)

            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
            if xbatch%100 == 0:
                end = time.time()
                print('Epoch[%d/%d]\tBatch[%d/%d]\tLoss: %.4f\tTime:%.2f'% (epoch+1, args.num_epoch,xbatch,num_batches,loss.data.cpu().numpy(),((end-start)/60)))
                tb.add_scalar('Training Loss', loss, xbatch+ epoch*num_batches)
                start = time.time()

        if (epoch+1) % args.save_epoch == 0:
        	torch.save({'glove' : glove.state_dict(),'optimizer' :optimizer.state_dict(),'params' : args}, os.path.join(model_name, 'epoch_%d_model.pkl'%(epoch+1)))


    plt.plot(LOSS)

    print('Saving losses .....!')
    torch.save(LOSS, os.path.join(model_name, 'training_loss.pt'))
    print('Saved!')
    if args.get_TSNE_plot == True:
        print('Plotting TSNE space of top {} words:....'.format(args.top_k))
        emb_i = glove.wi.weight.cpu().data.numpy()
        emb_j = glove.wj.weight.cpu().data.numpy()
        emb = emb_i + emb_j
        top_k = args.top_k
        tsne = TSNE(metric='cosine', random_state=123)
        embed_tsne = tsne.fit_transform(emb[:top_k, :])
        fig, ax = plt.subplots(figsize=(14, 14))
        for idx in range(top_k):
            plt.scatter(*embed_tsne[idx, :], color='steelblue')
            plt.annotate(dataset._id2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
    plt.show()