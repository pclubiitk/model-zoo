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
from models import inception_v3, Inception3, InceptionA, InceptionB, InceptionC, InceptionD, InceptionAux, BasicConv2d
from eval import plot_epoch

train_loader, val_loader, test_loader = load_cifar()

model = inception_v3()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
summary(model, (3, 32, 32))

LEARNING_RATE = 0.001
MOMENTUM = 0.9

cast = torch.nn.CrossEntropyLoss().to(device)
# Optimization
optimizer = torch.optim.SGD(
    model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


def train_model():

    EPOCHS = 100
    nb_examples = 45000
    nb_val_examples = 5000
    train_costs, val_costs = [], []
    train_accuracy, val_accuracy = [], []

    for epoch in range(EPOCHS):

        train_loss = 0
        correct_train = 0

        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs, aux_outputs = model(images)
            loss1 = cast(outputs, labels)
            loss2 = cast(aux_outputs, labels)

            loss = loss1 + 0.4 * loss2

            loss.backward()

            optimizer.step()

            # equal prediction and acc
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()

            train_loss += loss.item() * images.size(0)

        train_loss /= nb_examples
        train_costs.append(train_loss)
        train_acc = correct_train / nb_examples
        train_accuracy.append(train_acc)

        val_loss = 0
        correct_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs, aux_outputs = model(images)
                loss1 = cast(outputs, labels)
                loss2 = cast(aux_outputs, labels)

                loss = loss1 + 0.4 * loss2

                # equal prediction and acc
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()

                val_loss += loss.item() * images.size(0)

            val_loss /= nb_val_examples
            val_costs.append(val_loss)
            val_acc = correct_val / nb_val_examples
            val_accuracy.append(val_acc)

        info = "[Epoch {}/{}]: train-loss = {:0.6f} | train-acc = {:0.3f} | val-loss = {:0.6f} | val-acc = {:0.3f}"
        print(info.format(epoch + 1, EPOCHS,
                          train_loss, train_acc, val_loss, val_acc))

    return train_accuracy, train_costs, val_accuracy, val_costs


train_accuracy, train_costs, val_accuracy, val_costs = train_model()

plot_epoch(train_costs, val_costs, train_accuracy, val_accuracy)

nb_test_examples = 10000
correct = 0

model.eval().cuda()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        # Make predictions.
        outputs = model(images)

        # Retrieve predictions indexes.
        _, predicted_class = torch.max(outputs.data, 1)

        # Compute number of correct predictions.
        correct += (predicted_class == labels).float().sum().item()

test_accuracy = correct / nb_test_examples
print('Test accuracy: {}'.format(test_accuracy))
