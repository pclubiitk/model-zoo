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
from eval import plot_accuracy_epoch, plot_loss_epoch, make_heat_map
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torch.nn.functional as F
from MLP_MIXER_Block import MixerBlock
from MLP import MLPMixer
from dataloader import get_loader
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_type", type=str, default="basic", required=True)
    parser.add_argument("--depth", type=int, default=3, required=True)
    parser.add_argument("--option", type=str, default="A")

    # optim config
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--base_lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--milestones", type=str, default="[80, 120]")
    parser.add_argument("--lr_decay", type=float, default=0.1)

    # run_config
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    model_config = OrderedDict(
        [
            ("input_size", args.input_size),
            ("patch", args.patch),
            ("dimension", args.dimension),
            ("in_channels", args.in_channels),
        ]
    )

    optim_config = OrderedDict(
        [
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("base_lr", args.base_lr),
            ("weight_decay", args.weight_decay),
            ("momentum", args.momentum),
            ("milestones", json.loads(args.milestones)),
            ("lr_decay", args.lr_decay),
        ]
    )

    data_config = OrderedDict(
        [
            ("dataset", "CIFAR10"),
        ]
    )
    run_config = OrderedDict(
        [
            ("device", args.device),
            ("num_workers", args.num_workers),
        ]
    )

    config = OrderedDict(
        [
            ("model_config", model_config),
            ("optim_config", optim_config),
            ("data_config", data_config),
            ("run_config", run_config),
        ]
    )

    return config


config = parse_args()

model = REPVGG(
        input_size=config["model_config"]["input_size"],
        patch=config["model_config"]["patch"],
        dim =config["model_config"]["dimension"],
        num_classes=config["model_config"]["numclasses"],
    )

optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optim_config"]["base_lr"],
        weight_decay=config["optim_config"]["weight_decay"],
    )


scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["optim_config"]["milestones"],
        gamma=config["optim_config"]["lr_decay"],
    )
criterion = nn.CrossEntropyLoss()

def train(
        model, epochs, trainloader, testloader, device, criterion, optimizer, scheduler
    ):
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                start_time = time.time()
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images).to(device)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                if (i + 1) % 250 == 0:
                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    print(
                        "Epoch {}, Step {} Loss: {:.4f} time : {:.4f}min".format(
                            epoch + 1, i + 1, loss.item(), total_time
                        )
                    )
            return train_losses


device = config["run_config"]["device"]

model.to(device)
criterion = nn.CrossEntropyLoss()
train_loader, test_loader = get_loader(
    config["optim_config"]["batch_size"], config["run_config"]["num_workers"]
)

train_loss = train(
        model,
        config["optim_config"]["epochs"],
        train_loader,
        test_loader,
        device,
        criterion,
        optimizer,
        scheduler,
    )

plot_loss_epoch(train_loss)



_, test_checker = get_loader(10000, config["run_config"]["num_workers"])
make_heat_map(model, test_checker, device)