#!/usr/bin/python3
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_accuracy_epoch(accuracy_train,accuracy_test):

    plt.figure(figsize=(12,7))
    plt.title("Train and Test Accuracy in ResNet")
    sns.lineplot(data = accuracy_train,label="Training accuracy",palette="muted")
    sns.lineplot(data = accuracy_test,label="Test Accuracy Model",palette="Dark")
    plt.xlabel("Epochs")    
    plt.ylabel("Accuracy")
    plt.savefig('accuracy.png')
    plt.show()
    return


def plot_loss_epoch(loss_train,loss_test):

    plt.figure(figsize=(12,7))
    plt.title("Train and Test Loss in ResNet")
    sns.lineplot(data = loss_train,label="Training Loss",palette="muted")
    sns.lineplot(data = loss_test,label="Test Loss",palette="dark")
    plt.xlabel("Epochs")    
    plt.ylabel("Loss")
    plt.show()
    plt.savefig('loss.png')
    return


def make_heat_map(model,test_loader,device='cpu') :

    with torch.no_grad():
        correct = 0
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_val = model(X_test)
            predicted = torch.max(y_val,1)[1]
            correct += (predicted == y_test).sum()
        
        print(f'Test accuracy Basic: {correct.item()}/{len(test_loader)*test_loader.batch_size} = {correct.item()*100/(len(test_loader)*test_loader.batch_size):7.3f}%')

    class_names = ['plane', '  car', ' bird', '  cat', ' deer', '  dog', ' frog', 'horse', ' ship', 'truck']    
    arr = confusion_matrix(y_test.view(-1).cpu(), predicted.view(-1).cpu())
    df_cm = pd.DataFrame(arr, class_names, class_names)
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.show()
    plt.savefig('heatmap.png')
    return