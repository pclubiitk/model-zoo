import numpy as np
import os
import cv2
import shutil
import urllib.request
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from torchsummary import summary

from dataloader import load_cifar
from models import ConvBlock, InceptionModule, InceptionAux, GoogLeNet
from eval import plot_epoch

train_loader, val_loader, test_loader = load_cifar()

model = GoogLeNet()

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
summary(model, (3, 96, 96))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model():
    EPOCHS = 100
    nb_examples = 45000
    nb_val_examples = 5000
    train_costs, val_costs = [], []
    train_accuracy, val_accuracy = [], []

    # Training phase.

    for epoch in range(EPOCHS):

        train_loss = 0
        correct_train = 0

        model.train().cuda()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients.
            optimizer.zero_grad()

            # Forward pass.
            prediction0, prediction1, prediction2 = model(inputs)

            # Compute the loss.
            loss0 = criterion(prediction0, labels)
            loss1 = criterion(prediction1, labels)
            loss2 = criterion(prediction2, labels)

            loss = loss0 + 0.3 * loss1 + 0.3 * loss2
            # Backward pass.
            loss.backward()

            # Optimize.
            optimizer.step()

            # Compute training accuracy.
            _, predicted = torch.max(prediction0.data, 1)
            correct_train += (predicted == labels).float().sum().item()

            # Compute batch loss.
            train_loss += (loss.data.item() * inputs.shape[0])

        train_loss /= nb_examples
        train_costs.append(train_loss)
        train_acc = correct_train / nb_examples
        train_accuracy.append(train_acc)

        val_loss = 0
        correct_val = 0

        model.eval().cuda()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass.
                prediction0, prediction1, prediction2 = model(inputs)

                # Compute the loss.
                loss0 = criterion(prediction0, labels)
                loss1 = criterion(prediction1, labels)
                loss2 = criterion(prediction2, labels)

                loss = loss0 + 0.3 * loss1 + 0.3 * loss2

                # Compute training accuracy.
                _, predicted = torch.max(prediction0.data, 1)
                correct_val += (predicted == labels).float().sum().item()

                # Compute batch loss.
                val_loss += (loss.data.item() * inputs.shape[0])

            val_loss /= nb_val_examples
            val_costs.append(val_loss)
            val_acc = correct_val / nb_val_examples
            val_accuracy.append(val_acc)

        info = "[Epoch {}/{}]: train-loss = {:0.6f} | train-acc = {:0.3f} | val-loss = {:0.6f} | val-acc = {:0.3f}"
        print(info.format(epoch+1, EPOCHS, train_loss, train_acc, val_loss, val_acc))

    return train_accuracy, train_costs, val_accuracy, val_costs


train_accuracy, train_costs, val_accuracy, val_costs = train_model()

plot_epoch(train_costs, val_costs, train_accuracy, val_accuracy)

nb_test_examples = 10000
correct = 0

model.eval().cuda()

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Make predictions.
        prediction, _, _ = model(inputs)

        # Retrieve predictions indexes.
        _, predicted_class = torch.max(prediction.data, 1)

        # Compute number of correct predictions.
        correct += (predicted_class == labels).float().sum().item()

test_accuracy = correct / nb_test_examples
print('Test accuracy: {}'.format(test_accuracy))
