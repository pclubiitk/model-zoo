#standard libraries
import os
from collections import OrderedDict
import argparse
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter

# torch libraries
import torch

#import written libraries
from utils import  weights_init_normal,real_data_target,fake_data_target,log_images
from dataset import get_loader
from models import Generator,Discriminator

def parse_args():
    parser = argparse.ArgumentParser()

    # Model config
    parser.add_argument('--in_channels', type=int, default=3,help = 'number of channels in input image')
    
    # optim config
    parser.add_argument('--epochs', type=int, default=15,help='nmber of epochs to train for')
    parser.add_argument('--train_batch_size', type=int, default=64,help='batch size of train set')
    parser.add_argument('--base_g_lr', type=float, default=2.5e-4,help='learning rate for generator')
    parser.add_argument('--base_d_lr', type=float, default=1e-4,help='learning rate for discriminator')
    parser.add_argument('--beta_1', type=float, default=0.5,help='beta-1 for optmizers')
    parser.add_argument('--beta_2', type=float, default=0.999,help='beta-2 for optmizers')
    parser.add_argument('--lamda_adv', type=float, default=0.001,help='lamda_adv in the loss function')
    
    #data_config
    parser.add_argument('--dataset_name', type=str,default='img_align_celeba',help='dataset name')
    parser.add_argument('--mask_size', type=int,default=64,help='size of mask applied to images')
    
    #run_config
    parser.add_argument('--device', type=str,default='cpu',help='device on which model is to be run on')
    parser.add_argument('--num_workers', type=int, default=5,help='standard num_workers argument in dataloader')
    parser.add_argument('--test_batch_size', type=int, default=16,help='batch size of test set')
    parser.add_argument('--writer_directory', type=str,default='./runs/',help='writing directory for tensorboard')
    parser.add_argument('--label_smoothing', type=bool, default=False,help='implements label smoothing if set as "True"')
    parser.add_argument('--save_frequency', type=int, default=500,help='interval for logging images in tensorboard')
    
    args = parser.parse_args()

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('train_batch_size', args.train_batch_size),
        ('base_g_lr', args.base_g_lr),
        ('base_d_lr', args.base_d_lr),
        ('beta_1', args.beta_1),
        ('beta_2', args.beta_2),
        ('lamda_adv', args.lamda_adv),
    ])

    run_config = OrderedDict([
        ('device', args.device),
        ('num_workers', args.num_workers),
        ('label_smoothing', args.label_smoothing),
        ('test_batch_size',args.test_batch_size),
        ('writer_directory',args.writer_directory),
        ('save_frequency',args.save_frequency),
    ])

    data_config = OrderedDict([
         ('dataset_name', args.dataset_name),
         ('mask_size', args.mask_size),
        ])

    model_config = OrderedDict([
        ('in_channels',args.in_channels),
        ])
    
    config = OrderedDict([
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
        ('model_config', model_config),
    ])

    return config

config = parse_args()

run_config = config['run_config']
optim_config = config['optim_config']
device = run_config['device']
model_config = config['model_config']
dataset_name = config['data_config']['dataset_name']
nc = model_config['in_channels']
mask_size = config['data_config']['mask_size']
lamda_adv = optim_config['lamda_adv']

#get dataloaders
dataloader,test_dataloader = get_loader(optim_config['train_batch_size'],run_config['test_batch_size'],mask_size,dataset_name,run_config['num_workers'])
#loss functions
adversarial_loss = torch.nn.BCEWithLogitsLoss()
pixelwise_loss = torch.nn.MSELoss()

#models
generator = Generator(nc).to(device)
discriminator = Discriminator(nc).to(device)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

#optimizers 
optimizer_G = torch.optim.Adam(generator.parameters(), lr=optim_config['base_g_lr'], betas=(optim_config['beta_1'],optim_config['beta_2']))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=optim_config['base_d_lr'], betas=(optim_config['beta_1'],optim_config['beta_2']))

#run configurations
epochs = optim_config['epochs']
writer = SummaryWriter(run_config['writer_directory'])

#training loop
for epoch in range(epochs):
    
    #load tqdm object
    t = tqdm(dataloader, desc='epoch:{} g_loss:{:.4f} d_loss:{:.4f}'.format(epoch, 0.0, 0.0), leave=True)
    
    for i, (imgs, masked_imgs, masked_parts) in enumerate(t):

        # Adversarial ground truths
        valid = real_data_target(imgs.size(0),'cuda')
        fake = fake_data_target(imgs.size(0),'cuda')
        imgs = imgs.to(device)
        masked_imgs = masked_imgs.to(device) 
        masked_parts = masked_parts.to(device)

        #current step used for logging
        current_step = epoch*len(dataloader) + i

        #train generator
        optimizer_G.zero_grad()

        gen_parts = generator(masked_imgs)

        #calculate loss
        g_adv = adversarial_loss(discriminator(gen_parts), valid)
        g_pixel = pixelwise_loss(gen_parts, masked_parts)      
        g_loss = lamda_adv* g_adv + (1-lamda_adv)* g_pixel
       
       #update gradients
        g_loss.backward()
        optimizer_G.step()

        #train discriminator
        optimizer_D.zero_grad()

        #calculate loss
        real_loss = adversarial_loss(discriminator(masked_parts), valid)
        fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        #update weights
        d_loss.backward()
        optimizer_D.step()

        #log to tensorboard
        writer.add_scalar('real loss',real_loss,current_step)
        writer.add_scalar('fake loss',fake_loss,current_step)
        writer.add_scalar('total d loss',d_loss,current_step)
        writer.add_scalar('adversarial loss',g_adv,current_step)
        writer.add_scalar('pixelwise loss',g_pixel,current_step)
        writer.add_scalar('total g loss',g_loss,current_step)

        #update tqdm bar
        if i% 50==0:
            t.set_description('epoch:{} g_loss:{:.4f} d_loss:{:.4f}'.format(epoch, g_loss.item(), d_loss.item()))
            t.refresh()
        
        if i% run_config['save_frequency'] ==0:

          generator.eval()
          log_images(writer,test_dataloader,mask_size,device,generator,current_step)

