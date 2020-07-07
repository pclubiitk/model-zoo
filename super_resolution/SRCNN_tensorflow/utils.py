import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_train(image_size = 33,stride = 14,scale = 3,dirname = '/content/drive/My Drive/train'):
    dirname = dirname
    dir_list = os.listdir(dirname)
    images = [cv2.cvtColor(cv2.imread(os.path.join(dirname,img)),cv2.COLOR_BGR2GRAY) for img in dir_list]
    images = [img[0:img.shape[0]-np.remainder(img.shape[0],scale),0:img.shape[1]-np.remainder(img.shape[1],scale)] for img in images]

    trains = images.copy()
    ground_truth = images.copy()
    
    trains = [cv2.resize(img, None, fx=1/scale, fy=1/scale) for img in trains]
    trains = [cv2.resize(img, None, fx=scale/1, fy=scale/1, interpolation=cv2.INTER_CUBIC) for img in trains]

    sub_trains = []
    sub_ground_truth = []

    for train, label in zip(trains, ground_truth):
        v, h = train.shape
        for x in range(0,v-image_size+1,stride):
            for y in range(0,h-image_size+1,stride):
                sub_train = train[x:x+image_size,y:y+image_size]
                sub_label = label[x:x+image_size,y:y+image_size]
                sub_train = sub_train.reshape(image_size,image_size,1)
                sub_label = sub_label.reshape(image_size,image_size,1)
                sub_trains.append(sub_train)
                sub_ground_truth.append(sub_label)
    
    X_train = np.array(sub_trains)
    Y_train = np.array(sub_ground_truth)
    return X_train,Y_train

def draw_loss_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()