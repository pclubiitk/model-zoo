import matplotlib.pyplot as plt
import tensorflow as tensorflow
from tensorflow import keras
from models import Mobilenet_v1, standard_conv, depthwise_conv
from dataloader import load_cifar
from tensorflow.keras.models import load_model

train_ds, test_ds =load_cifar()
model = load_model('mobilenetv1.h5')


def evaluate(model, test_ds):
    predict = model.evaluate(test_ds)
    print("The Test Accuracy of the model is: ", predict * 100)

evaluate(test_ds, model)