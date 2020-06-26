import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
from utils import load_train,draw_loss_plot
from model import SRCNN
from tensorflow.keras.models import  model_from_json

parser = argparse.ArgumentParser()

parser.add_argument('--Image_patch_size', type=int, default=33,
                    help="size of a patch to be used for training: default 33 ")
parser.add_argument('--Epochs', type=int, default=1500,
                    help="Numberof epochs used for training : default 1500")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="Learning_rate used in the optimizing algorithm: default 0.0001")
parser.add_argument('--Scale', type=int, default=3,
                    help="Scale by which a HR image is downscaled to low spatial resolution image and then by the same scale it is upscaled using inter-cubic interpolation: default 3 ")
parser.add_argument('--BATCH_SIZE', type=int, default=128,
                    help="Batch size, default 128")
parser.add_argument('--Stride', type=int, default=14,
                    help="Stride used when selecting patches of image_patch_size from a image:default 14")
parser.add_argument('--is_training',type =bool,default= True,
                     help="decide whether to train the model on a dataset or test the pre-trained model: default True")
parser.add_argument('--dirname_train',type =str,default= '/content/drive/My Drive/train',
                     help="name of the directory where training dataset is stored")
parser.add_argument('--dirname_test',type =str,default= '/content/drive/My Drive/test',
                     help="name of the directory where testing dataset is stored")


args = parser.parse_args()

image_size = args.Image_patch_size      #image_patch_size
stride = args.Stride
scale = args.Scale
learning_rate = args.lr
batch_size = args.BATCH_SIZE
epochs = args.Epochs
is_training =args.is_training
dirname_train = args.dirname_train
dirname_test = args.dirname_test

if is_training:
   
    X_train,Y_train = load_train(image_size = image_size,stride = stride,scale = scale,dirname =dirname_train)
    srcnn = SRCNN(  image_size = image_size, learning_rate=learning_rate)
    optimizer = Adam(lr = learning_rate)
    srcnn.compile(optimizer=optimizer, loss='mean_squared_error')
    history = srcnn.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2,validation_split = 0.1 )
    draw_loss_plot(history = history)

    #Saving trained model_weights in the current workspace .
    #make a folder named srcnn in your current workspace you are working in.In that folder your weights will be stored.
    json_string = srcnn.to_json()
    open(os.path.join('./srcnn/','srcnn_model.json'),'w').write(json_string)
    srcnn.save_weights(os.path.join('./srcnn/','srcnn_weight.hdf5'))

else:

    dir_list = os.listdir(dirname_test)
    for img in dir_list:

       image = cv2.cvtColor(cv2.imread(os.path.join(dirname_test,img)),cv2.COLOR_BGR2GRAY) 
       image = image[0:image.shape[0]-np.remainder(image.shape[0],scale),0:image.shape[1]-np.remainder(image.shape[1],scale)]
    
       Y_test = image.copy() 
       X_test = cv2.resize(image, None, fx=1/scale, fy=1/scale)
       X_test = cv2.resize(X_test, None, fx= scale/1, fy= scale/1, interpolation=cv2.INTER_CUBIC)    
       X_test = X_test.reshape(1,X_test.shape[0],X_test.shape[1],1)

       weight_filename = 'srcnn_weight.hdf5'
       model = SRCNN()
       model.compile(optimizer = optimizer,loss = 'mean_squared_error')
       model.load_weights(os.path.join('./srcnn/',weight_filename))
       Y_pred = model.predict(X_test)

       X_test = X_test.reshape(X_test.shape[1],X_test.shape[2])
       Y_pred = Y_pred.reshape(Y_pred.shape[1],Y_pred.shape[2])
   
       cv2.imshow(X_test)
       cv2.imshow(Y_test)
       cv2.imshow(Y_pred)