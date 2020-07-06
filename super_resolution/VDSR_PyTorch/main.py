import argparse
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
from model import VDSR
from util import *

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batchSize", type=int, default=128, help="Training batch size. Default 128")
    parser.add_argument("--Epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
    parser.add_argument("--step", type=int, default=10,
                        help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number")
    parser.add_argument("--cuda", action="store_true", help="Use cuda?")
    parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
    parser.add_argument("--pretrained", default='', type=str, help="Path to pretrained model")
    parser.add_argument("--train_data", default="train.h5", type=str, help="Path to preprocessed train dataset")
    parser.add_argument("--test_data", default="./assets/", type=str, help="Path to file containing test images")
    args = parser.parse_args()

    cuda = args.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(0))
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    cudnn.benchmark = True

    train_set = prepareDataset(args.train_data)
    train_data = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)

    model = VDSR()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.MSELoss(size_average=False)
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading model '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            args.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint['model'].state_dict())
        else:
            print("No model found at '{}'".format(opt.pretrained))

    train(args.start_epoch, train_data, optimizer, model, criterion, args.Epochs, args)
    eval(model, args)


def train(start_epoch, dataloader, optimizer, model, criterion, Epoch, args):
    for epoch in range(start_epoch, Epoch+1):
        lr = adjust_learning_rate(optimizer, epoch - 1, args)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        for i, batch in enumerate(dataloader, 1):
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

            if args.cuda:
                input = input.cuda()
                target = target.cuda()

            loss = criterion(model(input), target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if i % 100 == 0:
                print("Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, i, len(dataloader), loss.item()))

        if epoch % 2 == 0:
            save_checkpoint(model, epoch)

def eval( model, args):
    im_gt = Image.open(args.test_data+"butterfly_GT.bmp").convert("RGB")
    im_b = Image.open(args.test_data+"butterfly_GT_scale_4.bmp").convert("RGB")
    # Convert the images into YCbCr mode and extraction the Y channel (for PSNR calculation)
    im_gt_ycbcr = np.array(im_gt.convert("YCbCr"))
    im_b_ycbcr = np.array(im_b.convert("YCbCr"))
    im_gt_y = im_gt_ycbcr[:, :, 0].astype(float)
    im_b_y = im_b_ycbcr[:, :, 0].astype(float)

    im_b = Variable(torch.from_numpy(im_b_y/255.).float())
    im_b = im_b.view(1, -1, im_b.shape[0], im_b.shape[1]).cuda()

    start_time = time.time()
    out = model(im_b)
    print("Time taken: ", time.time()-start_time)

    out = out.cpu()
    out_y = out.data[0].numpy().astype(np.float32)
    out_y = out_y * 255.
    out_y[out_y > 255.] = 255.
    out_y[out_y < 0] = 0
    out_y = out_y[0,:,:]
    
    psnr_org = computePSNR(im_gt_y, im_b_y)
    psnr_score = computePSNR(im_gt_y, out_y)
    print(" PSNR score for our predicted image is ", psnr_score)
    print(" Improvement from blur image is ",psnr_score-psnr_org)

    out_img = colorize(out_y, im_b_ycbcr)
    im_gt = Image.fromarray(im_gt_ycbcr, "YCbCr").convert("RGB")
    im_b = Image.fromarray(im_b_ycbcr, "YCbCr").convert("RGB")

    plt.imshow(im_gt)
    plt.savefig("GT.png")
    plt.show()
    
    plt.imshow(im_b)
    plt.savefig("Input.png")
    plt.show()
    
    plt.imshow(out_img)
    plt.savefig("img.png")
    plt.show()

if __name__ == '__main__':
    main()