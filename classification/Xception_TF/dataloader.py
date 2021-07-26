import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tarfile
import requests
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt


# ? to load datasets using load_cifar10()
# ? This function creates a folder 'data' in the same directory which consists of a folder 'cifar10' which consists of 'train' and 'test' folder
# ? the 'train' folder consists of 50000 images and labels  for training and the 'test' folder contains 10000 images and labels for testing.
# ? Normalisation applied to the arrays to help the network to learn properly since all the pixel values are scaled down in such a way that
# ? the overall mean is 1 and standard deviation is 0. This speeds up the learning and leades to faster convergence.


def load_cifar10():
    #  dataset_url='https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz'
    #  download_url(dataset_url,'.')
    #  with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
    #     tar.extractall(path='./data')
    # dataset is opened in the folder format in order to understand the distribution by ploting a barplot.
    url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    response = requests.get(url, stream=True)
    file = tarfile.open(fileobj=response.raw, mode="r|gz")
    file.extractall(path=".")

    # ? prints a barplot showing the distribution of training data.
    data_url = "cifar10"  # dadaset directory
    img_list = []
    class_list = []  # contains the 10 classes
    train_url = os.path.join(data_url, "train")
    test_url = os.path.join(data_url, "test")
    train_classes = os.listdir(train_url)
    count_dict = {n: 0 for n in train_classes}
    for index, label in enumerate(train_classes):
        class_url = os.path.join(train_url, "{}".format(label))
        class_list.append(class_url)
        for images in os.listdir(class_url):
            img_list.append(images)
            count_dict[label] += 1
    df = pd.DataFrame(count_dict, index=[0])
    fig, axs = plt.subplots(figsize=(10, 7))
    ax = sns.barplot(data=df)
    sns.set_style("darkgrid")
    ax.set(xlabel="categories of images", ylabel="count")
    ax.set_title("Distribution of images in CIFAR_10 Dataset")

    # img_list contains the number of images present in the training sample
    # count_dict contains the key value pairs of the classes and the number of images belonging to that class in the training sample

    # ? creates the train generator and test generator on which augmetations are applied while passing it to the network.
    train_gen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
    )

    train_ds = train_gen.flow_from_directory(
        directory="cifar10/train", target_size=(299, 299), batch_size=32
    )

    test_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    test_ds = test_gen.flow_from_directory(
        directory="cifar10/test", target_size=(299, 299), batch_size=32
    )

    return train_ds, test_ds
