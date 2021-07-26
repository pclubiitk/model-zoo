import tensorflow as tensorflow
from tensorflow import keras
from dataloader import load_cifar
from tensorflow.keras.models import load_model

train_ds, test_ds =load_cifar()
model = load_model('xception.h5')

# This functions tells the model to act on the test data and returns the accuracy obtained on the test data.
def evaluate(model, test_ds):
    predict = model.evaluate(test_ds)
    print("The Test Accuracy of the model is: ", predict * 100)

evaluate(model, test_ds)