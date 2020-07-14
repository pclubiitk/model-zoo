import numpy as np
from numpy import  ones 
import keras 
from keras.models import Model
from keras import backend
from matplotlib import pyplot
import argparse

import tensorflow as tf
import matplotlib.pyplot as plt
from model import *
from utils import *
############################################################
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type = int, default = 20, help = "No of epochs: default 20 ")
parser.add_argument('--latent_dim', type = int, default = 100, help = "Dimension of latent vector , default 100")
parser.add_argument('--lr', type = float, default = 0.0002, help = "Learning rate : default 0.0002 ")
parser.add_argument('--dropout', type = float, default = 0.4, help = "Dropout, default 0.4")
parser.add_argument('--beta_1', type = float, default = 0.5, help = "beta_1 : default 0.5")
parser.add_argument('--alpha', type = float, default = 0.2, help = "alpha : default 0.2")

args = parser.parse_args()
############################################################
#Global hyperparams
latent_dim = args.latent_dim
lr_ = args.lr
dropout_ = args.dropout
beta_1_ = args.beta_1
alpha_ = args.alpha
epochs_ = args.epochs
############################################################
##To log the loss and accuracy
plot_sup_loss=[]
plot_unsup_loss=[]
plot_gan_loss=[]
plot_sup_acc=[]
plot_test_acc=[]
############################################################

def train(g_model , us_model , s_model ,model , dataset , latent_dim , epochs =epochs_, n_batch=100):
  x_sup , y_sup = make_supervised_train_dataset(dataset)
  print(x_sup.shape , y_sup.shape)
  print(dataset[0].shape[0])
  batches_per_epoch = int(dataset[0].shape[0]/n_batch)
  number_steps = epochs* batches_per_epoch
  half_batch = int(n_batch/2)
  print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (epochs, n_batch, half_batch, batches_per_epoch, number_steps))
	# manually enumerate epochs
  for i in range(number_steps):
    [x_sup_real , y_sup_real] , _ = gen_real_samples([x_sup , y_sup] ,half_batch)
    s_loss , s_acc = s_model.train_on_batch(x_sup_real , y_sup_real)

    [x_us_real , _ ] , y_us_real = gen_real_samples(dataset ,half_batch)
    us_loss0  = us_model.train_on_batch(x_us_real , y_us_real)
    x_us_fake , y_us_fake = gen_fake_samples(g_model , latent_dim ,half_batch)
    us_loss1  = us_model.train_on_batch(x_us_fake , y_us_fake)

    x_gan ,y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
    gan_loss = model.train_on_batch(x_gan , y_gan)
    if ((i+1) % (batches_per_epoch) == 0): 
      summarize_performance(i, g_model, s_model, latent_dim, dataset)
    print('>%d, sup_loss && acc[%.3f,%.0f], unsup_loss[%.3f,%.3f], gen_loss[%.3f  ]' % (i+1, s_loss, s_acc*100, us_loss0, us_loss1, gan_loss ))
    plot_sup_loss.append(s_loss)
    plot_unsup_loss.append(us_loss0+ us_loss1)
    plot_gan_loss.append(gan_loss)
    plot_sup_acc.append(s_acc)
		# evaluate the model performance every so often
def summarize_performance(step, g_model,s_model, latent_dim, dataset, n_samples=100):
    # prepare fake examples
    X, _ = gen_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(100):
      # define subplot
      pyplot.subplot(10, 10, 1 + i)
      # turn off axis
      pyplot.axis('off')
      # plot raw pixel data
      pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # evaluate the classifier model
    X, y = dataset
    _, acc = s_model.evaluate(X, y, verbose=0)
    print('Classifier Accuracy: %.3f%%' % (acc * 100))
    
    # save the generator model
    filename2 = 'g_model_%04d.h5' % (step+1)
    g_model.save(filename2)
    # save the classifier model
    filename3 = 'c_model_%04d.h5' % (step+1)
    s_model.save(filename3)
    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))
    plot_test_acc.append(acc)


# creating model structure
g_model = define_generator(latent_dim,alpha_)
g_model.summary()
s_model, us_model = define_discriminator(alpha_,dropout_,lr_,beta_1_)
s_model.summary()
us_model.summary()
model = define_gan(g_model , us_model,lr_,beta_1_)
model.summary()


dataset = load_dataset()
train(g_model, us_model, s_model, model, dataset, latent_dim)
#one_example()
#graph_plot()
