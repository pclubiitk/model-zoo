from __future__ import print_function  # Standard Imports

import numpy as np
import argparse
import os
import time
import pickle
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as ds
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
# ------------------------------
from dataloader import get_data		# Own module Imports
from utils import *
from model import *
# -------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training hyper-parameters
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    # Learning Rate for D
    parser.add_argument('--lrD', type=float, default=0.0002)
    # Learning Rate for G
    parser.add_argument('--lrG', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float,
                        default=0.999)  # momentum2 in Adam
    parser.add_argument('--recog_weight', type=float, default=0.1)
    # misc
    parser.add_argument('--model_path', type=str,
                        default='trained_model')  # Model Save
    parser.add_argument('-save_epoch', type=int,
                        default=5)  # Saving epochs after

    args = parser.parse_args()
    print(args)

    model_name = args.model_path + str(datetime.datetime.now())
    os.mkdir(model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader = get_data(model_name, args.batch_size, args.num_workers)

    S = SharedNetwork().to(device)
    D = Discriminator().to(device)
    Q = Recogniser().to(device)
    G = Generator().to(device)
    S.apply(init_weights)
    D.apply(init_weights)
    Q.apply(init_weights)
    G.apply(init_weights)

    criterionD = nn.BCELoss().to(device)
    classifyQ = nn.CrossEntropyLoss().to(device)
    contiQ = NormalNLLLoss()

    optimD = optim.Adam([{'params': S.parameters()}, {
                        'params': D.parameters()}], lr=args.lrD, betas=(args.beta1, args.beta2))
    optimG = optim.Adam([{'params': G.parameters()}, {
                        'params': Q.parameters()}], lr=args.lrG, betas=(args.beta1, args.beta2))

    tb = SummaryWriter()

    real_im = torch.FloatTensor(args.batch_size, 1, 28, 28).to(device)
    label = torch.FloatTensor(args.batch_size, 1).to(device)
    label = Variable(label, requires_grad=False)
    dis_c = torch.FloatTensor(args.batch_size, 10).to(device)
    con_c = torch.FloatTensor(args.batch_size, 2).to(device)
    noise = torch.FloatTensor(args.batch_size, 62).to(device)

    # Fixed variables for testng
    c = np.linspace(-1, 1, 10).reshape(-1, 1)
    c = np.repeat(c, 10, 0).reshape(-1, 1)
    c1 = np.hstack([c, np.zeros_like(c)])
    c2 = np.hstack([np.zeros_like(c), c])
    idx = np.arange(10).repeat(10)
    one_hot_vec = np.zeros((100, 10))
    one_hot_vec[range(100), idx] = 1
    fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)

    D_loss_list = [0]
    G_loss_list = [0]

    print('Training Started!')

    for epoch in range(args.num_epoch):

        start = time.time()

        for i, batch_data in enumerate(train_loader):
            batch_size = batch_data[0].size(0)

            #print('hey batch size is %d'%batch_size)

            optimD.zero_grad()

    	    # Real MNIST images
            real_im.data.copy_(batch_data[0])
            first_op = S(real_im)
            is_real = D(first_op)
            label.data.fill_(0.99)
            D_real_loss = criterionD(is_real, label)
            D_real_loss.backward()

    	    # Fake generated images
            z, fake_idx = noise_sample(batch_size, dis_c, con_c, noise)
            # debug : print(z.size())
            fake_im = G(z)
            second_op = S(fake_im.detach())
            is_fake = D(second_op)
            label.data.fill_(0.01)
            D_fake_loss = criterionD(is_fake, label)
            D_fake_loss.backward()

            D_loss = D_real_loss + D_fake_loss
            optimD.step()

    	    # Training G
            optimG.zero_grad()

            third_op = S(fake_im)
            discrim_pred = D(third_op)
            label.data.fill_(0.99)
            generator_loss = criterionD(discrim_pred, label)

    	    # Mutual info maximisation
            q_logits, q_mu, q_var = Q(third_op)
            fake_idx = Variable(torch.LongTensor(fake_idx).to(device), requires_grad=False)
            digit_classify_loss = classifyQ(q_logits, fake_idx)
            conti_loss = contiQ(con_c, q_mu, q_var)*args.recog_weight

            G_loss = generator_loss + digit_classify_loss + conti_loss
            G_loss.backward()
            optimG.step()

            G_loss_list.append(G_loss)
            D_loss_list.append(D_loss)

        tb.add_scalar('Discriminator Loss', D_loss, epoch+1)
        tb.add_scalar('Generator Loss', G_loss, epoch+1)

        end = time.time()
        print('Epoch[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tTime:%.2f'
        % (epoch+1, args.num_epoch,
          D_loss.data.cpu().numpy(), G_loss.data.cpu().numpy(), ((end-start)/60)))

        noise.data.copy_(fix_noise)
        dis_c.data.copy_(torch.Tensor(one_hot_vec))

        con_c.data.copy_(torch.from_numpy(c1))
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74)
        x_save = G(z)
        save_image(x_save.data, os.path.join(model_name, 'epoch_%d_c1.png'%(epoch+1)), nrow=10)

        con_c.data.copy_(torch.from_numpy(c2))
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74)
        x_save = G(z)
        save_image(x_save.data, os.path.join(model_name, 'epoch_%d_c2.png'%(epoch+1)), nrow=10)

        if (epoch+1) % args.save_epoch == 0:
        	torch.save({
            'G' : G.state_dict(),
            'D' : D.state_dict(),
            'Q' : Q.state_dict(),
        	'S' : S.state_dict(),
            'optimD' : optimD.state_dict(),
            'optimG' : optimG.state_dict(),
            'params' : args
            }, os.path.join(model_name, 'epoch_%d_model.pkl'%(epoch+1)))

torch.save(D_loss_list, os.path.join(model_name, 'discriminator_loss.pt'))
torch.save(G_loss_list, os.path.join(model_name, 'generator_loss.pt'))
