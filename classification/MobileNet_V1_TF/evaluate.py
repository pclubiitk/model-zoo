import matplotlib.pyplot as plt
from dataloader import load_cifar10
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from models import Mobilenet_v1, standard_conv, depthwise_conv
from tensorflow.keras.models import load_model

train_ds, test_ds = load_cifar10()

model = load_model('mobilenetv1.h5')

def evaluate(model, test_ds):
    predict = model.evaluate(test_ds)
    print("The Test Accuracy of the model is: ", predict * 100)

evaluate(model, test_ds)

