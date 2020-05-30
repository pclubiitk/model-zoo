import os
import argparse
import glob
import time
import math

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from models import Image_Discriminator, Video_Discriminator, Generator, GRU
from util import *

def main():
    parser = argparse.ArgumentParser(description='Start trainning MoCoGAN.....')
    parser.add_argument('--cuda', type=int, default=1,
                     help='set -1 when you use cpu')
    parser.add_argument('--batch-size', type=int, default=16,
                     help='set batch_size')
    parser.add_argument('--epochs', type=int, default=60000,
                     help='set num of iterations')
    parser.add_argument('--pre-train', type=int, default=-1,
                     help='set 1 when you use pre-trained models')
    #parser.add_argument('-dir', typr=str, )

    args       = parser.parse_args()
    cuda       = args.cuda
    batch_size = args.batch_size
    pre_train  = args.pre_train
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    # Making required folder
    if not os.path.exists('./generated_videos'):
      os.makedirs('./generated_videos')
    if not os.path.exists('./trained_models'):
      os.makedirs('./trained_models')

    T = 16
    start_epoch = 1
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda == True:
        torch.cuda.set_device(0)

    videos, current_path = preprocess(args)
    num_vid = len(videos)
    img_size = 96
    channel = 3
    d_E = 10
    hidden_size = 100  
    d_C = 50
    d_M = d_E
    nz = d_C + d_M
    criterion = nn.BCELoss()

    # setup model #
    dis_i = Image_Discriminator(channel)
    dis_v = Video_Discriminator()
    gen_i = Generator(channel, nz)
    gru = GRU(d_E, hidden_size, gpu=cuda)
    gru.initWeight()
    
    # setup optimizer #
    lr = 0.0002
    betas = (0.5, 0.999)
    optim_Di = optim.Adam(dis_i.parameters(), lr=lr, betas=betas)
    optim_Dv = optim.Adam(dis_v.parameters(), lr=lr, betas=betas)
    optim_Gi = optim.Adam(gen_i.parameters(), lr=lr, betas=betas)
    optim_GRU = optim.Adam(gru.parameters(), lr=lr, betas=betas)

    if cuda == True:
        dis_i.cuda()
        dis_v.cuda()
        gen_i.cuda()
        gru.cuda()
        criterion.cuda()

    trained_path = os.path.join(current_path, 'trained_models')
    video_lengths = [video.shape[1] for video in videos]

    if pre_train == True:
      checkpoint = torch.load(trained_path+'/last_state')
      start_epoch = checkpoint['epoch']
      Gi_loss = checkpoint['Gi']
      Gv_loss = checkpoint['Gv']
      Dv_loss = checkpoint['Dv']
      Di_loss = checkpoint['Di']
      dis_i.load_state_dict(torch.load(trained_path + '/Image_Discriminator.model'))
      dis_v.load_state_dict(torch.load(trained_path + '/Video_Discriminator.model'))
      gen_i.load_state_dict(torch.load(trained_path + '/Generator.model'))
      gru.load_state_dict(torch.load(trained_path + '/GRU.model'))
      optim_Di.load_state_dict(torch.load(trained_path + '/Image_Discriminator.state'))
      optim_Dv.load_state_dict(torch.load(trained_path + '/Video_Discriminator.state'))
      optim_Gi.load_state_dict(torch.load(trained_path + '/Generator.state'))
      optim_GRU.load_state_dict(torch.load(trained_path + '/GRU.state'))
      print("Using Pre-trained model")

    def checkpoint(model, optimizer, epoch):
      state = {'epoch': epoch+1, 'Gi': Gi_loss, 'Gv': Gv_loss, 'Di': Dv_loss, 'Di': Di_loss}
      torch.save(state, os.path.join(trained_path, 'last_state'))
      filename = os.path.join(trained_path, '%s' % (model.__class__.__name__))
      torch.save(model.state_dict(), filename + '.model')
      torch.save(optimizer.state_dict(), filename + '.state')

    def generate_z(num_frame):
        eps = Variable(torch.randn(batch_size, d_E))
        z_c = Variable(torch.randn(batch_size, 1, d_C))
        z_c = z_c.repeat(1, num_frame, 1)
        if cuda == True:
            z_c, eps = z_c.cuda(), eps.cuda()
        # Initialising the hidden var for GRU
        gru.initHidden(batch_size)
        z_m = gru(eps, num_frame).transpose(1, 0)
        # print(z_m.shape)
        z = torch.cat((z_m, z_c), 2) # (batch_size, num_frame, nz)
        return  z

    if pre_train == -1:
        Gi_loss = []
        Gv_loss = []
        Di_loss = []
        Dv_loss = []

    for epoch in range(start_epoch, args.epochs+1):
        start_time = time.time()
        real_videos = Variable(randomVideo(videos, batch_size, T))  # (batch_size, channel, T, img_size, img_size)
        if cuda == True:
            real_videos = real_videos.cuda()
        real_imgs = real_videos[:, :, np.random.randint(0, T), :, :]

        num_frame = video_lengths[np.random.randint(0, num_vid)]
        # Generate Z having num_frame no. of frames

        Z = generate_z(num_frame).view(batch_size,num_frame, nz, 1, 1)
        #print(Z.shape)
        Z = sample(Z, T).contiguous().view(batch_size*T, nz, 1, 1) # So that conv layers (nz, 1, 1) noise to (channel, img_size, img_size) image frame
        fake_vid = gen_i(Z).view(batch_size, T, channel, img_size, img_size)
        fake_vid = fake_vid.transpose(2, 1)
        # sample a fake image from fake_vid frames
        fake_img = fake_vid[: , :, np.random.randint(0, T), :, :]

        r_label = Variable(torch.FloatTensor(batch_size, 1).fill_(0.9)).to(args.device)
        f_label = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0)).to(args.device)
        # Training Discriminators
        # Video Discriminator
        dis_v.zero_grad()
        outputs = dis_v(real_videos)
        loss = criterion(outputs, r_label)
        loss.backward()
        real_loss = loss
        outputs = dis_v(fake_vid.detach())
        loss = criterion(outputs, f_label)
        loss.backward()
        fake_loss = loss
        dv_loss = real_loss + fake_loss
        
        optim_Dv.step()

        # Image Discriminator
        dis_i.zero_grad()
        r_outputs = dis_i(real_imgs)
        lossi = criterion(r_outputs, r_label)
        lossi.backward()
        real_lossi = lossi
        f_outputs = dis_i(fake_img.detach())
        fake_lossi = criterion(f_outputs, f_label)
        fake_lossi.backward()
        di_loss = real_lossi + fake_lossi
        #di_loss.backward()
        optim_Di.step()

        # Training Generator and GRU
        gen_i.zero_grad()
        gru.zero_grad()
        gen_outputs = dis_v(fake_vid)
        gv_loss = criterion(gen_outputs, r_label)
        gv_loss.backward(retain_graph=True)
        gen_out = dis_i(fake_img)
        gi_loss = criterion(gen_out, r_label)
        gi_loss.backward()
        optim_Gi.step()
        optim_GRU.step()

        Gi_loss.append(gi_loss.item())
        Gv_loss.append(gv_loss.item())
        Dv_loss.append(dv_loss.item())
        Di_loss.append(di_loss.item())

        end_time = time.time()

        # Plot 
        plt.plot(Gi_loss, label='Image Generator')
        plt.plot(Gv_loss, label='Video Generator')
        plt.plot(Di_loss, label='Image Discriminator')
        plt.plot(Dv_loss, label='Video Discriminator')
        plt.legend()
        plt.savefig("plot.png")
        # #plt.show()

        if epoch % 100 == 0:
            print('[%d/%d] Time_taken: %f || Gi loss: %.3f || Gv loss: %.3f || Di loss: %.3f || Dv loss: %.3f'%(epoch, args.epochs, end_time-start_time, gi_loss, gv_loss, di_loss, dv_loss))

        if epoch % 5000 == 0:
            checkpoint(dis_i, optim_Di, epoch)
            checkpoint(dis_v, optim_Dv, epoch)
            checkpoint(gen_i, optim_Gi, epoch)
            checkpoint(gru,   optim_GRU, epoch)

        if epoch % 1000 == 0:
            save_video(fake_vid[0].data.cpu().numpy().transpose(1, 2, 3, 0), epoch, current_path)

if __name__ == '__main__':
    main()




