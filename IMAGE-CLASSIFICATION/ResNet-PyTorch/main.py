#!/usr/bin/python3

import os
import time
import importlib
import json
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random
import time
from eval import plot_accuracy_epoch,plot_loss_epoch,make_heat_map
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torch.nn.functional as F
from models import ResNet,BasicModule,BottleNeckModule

from dataloader import get_loader

def parse_args():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument('--block_type', type=str,default='basic',required=True)
    parser.add_argument('--depth', type=int,default=3,required=True)
    parser.add_argument('--option', type=str,default='A')

    # optim config
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--milestones', type=str, default='[80, 120]')
    parser.add_argument('--lr_decay', type=float, default=0.1)

    #run_config
    parser.add_argument('--device', type=str,default='cpu')
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()


    model_config = OrderedDict([
        ('block_type', args.block_type),
        ('depth', args.depth),
        ('option',args.option)
    ])

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
        ('momentum', args.momentum),
        ('milestones', json.loads(args.milestones)),
        ('lr_decay', args.lr_decay),
    ])

    data_config = OrderedDict([
        ('dataset', 'CIFAR10'),
    ])
    run_config = OrderedDict([
        ('device', args.device),
        ('num_workers', args.num_workers),
        
    ])


    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
    ])

    return config


config = parse_args()

if config['model_config']['block_type'] == 'basic':

    model = ResNet(BasicModule,filter_map=[16,32,64],n=config['model_config']['depth'],option=config['model_config']['option'])
else :

    model = ResNet(BottleNeckModule,[16,32,64],config['model_config']['depth'],config['model_config']['option'])

optimizer = torch.optim.Adam(params=model.parameters(),lr = config['optim_config']['base_lr'],weight_decay=config['optim_config']['weight_decay'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=config['optim_config']['milestones'],gamma=config['optim_config']['lr_decay'])

def train(model,epochs,trainloader,testloader,device,criterion,optimizer,scheduler):

    model.train()
    start_time = time.time()
    train_losses = np.array([])
    test_losses = np.array([])
    train_correct = np.array([])
    test_correct = np.array([])

    for epoch in range(epochs):
        trn_corr = 0
        tst_corr = 0
        running_loss = 0
    
        # Run the training batches
        t = tqdm(trainloader, desc='epoch:{} loss:{:.4f} accuracy:{}'.format(epoch, 0.0, 'NA'), leave=True)
        for b, (X_train, y_train) in enumerate(t):
        
            # Apply the model
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            running_loss+=loss.item()
 
        # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr.item()
        
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        
            # # Print interim results
            if b%50 == 0:
                t.set_description('epoch:{} loss:{:.4f} accuracy:{}'.format(epoch, loss.item(), batch_corr.item()))
                t.refresh()
        #     print(f'epoch: {i:2}  batch: {b:4} [{128*b:6}/50000]  loss: {loss.item():10.8f}   accuracy: {(batch_corr.item()*100/128):10.8f}%',end=' ')

        
        train_losses = np.append(train_losses,loss.item())
        train_correct = np.append(train_correct,trn_corr/500)
    
        
    # Run the testing batches
        with torch.no_grad():
            t = tqdm(testloader, desc="[Validation] Epoch:{}".format(epoch), leave=True)
            for (X_test, y_test) in t:
          
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                # Apply the model
                y_val = model(X_test)

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1] 
                batch_corr = (predicted == y_test).sum()  
                tst_corr+=batch_corr.item()
                # if b==40 :
                #   print(f'test accuracy :{batch_corr.item()*100/128}% ')
 
        loss = criterion(y_val, y_test)
        print('Epoch:{}  loss:{:.3f} accuracy:{:.2f}% test_accuracy:{:.2f}%'.format(epoch, running_loss, trn_corr/500, tst_corr/100))

        test_losses=np.append(test_losses,loss.item())
        test_correct=np.append(test_correct,tst_corr/100)    
    print(f'\nDuration: {(time.time() - start_time)/60} minutes') # print the time elapsed  ,minutes') # print the time elapsed
    
    return train_losses,test_losses,train_correct,test_correct
device = config['run_config']['device']

model.to(device)
criterion = nn.CrossEntropyLoss()
train_loader,test_loader = get_loader(config['optim_config']['batch_size'],config['run_config']['num_workers'])

train_loss,test_loss,train_accuracy,test_accuracy = train(model,config['optim_config']['epochs'],train_loader,test_loader,device,criterion,optimizer,scheduler)

plot_loss_epoch(train_loss,test_loss)
plot_accuracy_epoch(train_accuracy,test_accuracy)
_,test_checker = get_loader(10000,config['run_config']['num_workers'])
make_heat_map(model,test_checker,device)
