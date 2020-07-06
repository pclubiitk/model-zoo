import torch.utils.data as data
import torch
import h5py
import os
import math
import numpy as np
from PIL import Image

class prepareDataset(data.Dataset):
    def __init__(self, path):
        super(prepareDataset, self).__init__()
        file = h5py.File(path, 'r')
        self.data = file.get('data')
        self.target = file.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]

def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

def computePSNR(gt, pred, border=1):
    height, width = pred.shape [:2]
    pred = pred[border:height-border, border:width-border]
    gt = gt[border:height-border, border:width-border]
    rmse = math.sqrt(np.mean((pred-gt)**2))
    if rmse == 0:
        return 100
    return 20*math.log10(255.0/rmse)

def colorize(y, ycbcr):
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 0:sz[1]]
    return img

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // args.step))
    return lr