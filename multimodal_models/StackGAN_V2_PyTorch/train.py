import torch
import numpy as np
from discriminator_model import D64, D512, D256, D128, D1024
import time, cv2, models

from postprocessing import postprocessing
from datasets import TextDataset
import helper_functions.config as cfg
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')
# from helper_functions.losses import GenLoss

class StackGAN():
    def __init__(self, eval_ = False, 
                inn_channels = cfg.channels, 
                generatorLR = cfg.generatorLR, 
                discriminatorLR = cfg.discriminatorLR, 
                StageNum = 4, beta1 = 0.5, beta2 = 0.999,
                zDim = 100):
        self.zDim = zDim
        self.inn_channels = inn_channels
        self.eval_ = eval_
        self.StageNum = StageNum
        if cfg.inception:
            from losses import INCEPTION_V3, compute_inception_score
            self.inception_model = INCEPTION_V3()
            inception_model = inception_model.cuda()
            inception_model.eval()

        image_transform = transforms.Compose([
                          transforms.Scale(int((64 * 4) * 76 / 64)),
                          transforms.RandomCrop(64 * 4),
                          transforms.RandomHorizontalFlip()])
        self.dataset = TextDataset('birds', 'train',
                                    base_size=64,
                                    transform=image_transform,
                                    StageNum=StageNum)

        self.generator = models.G_NET(StageNum=StageNum, zDim=zDim).cuda()
        self.discriminator = []
        if StageNum == 1:
            self.discriminator.append(D_NET64(self.inn_channels).cuda())
        elif StageNum == 2:
            self.discriminator.append(D64(self.inn_channels).cuda())
            self.discriminator.append(D128(self.inn_channels).cuda())
        elif StageNum == 3:
            self.discriminator.append(D64(self.inn_channels).cuda())
            self.discriminator.append(D128(self.inn_channels).cuda())
            self.discriminator.append(D256(self.inn_channels).cuda())
        elif StageNum == 4:
            self.discriminator.append(D64(self.inn_channels).cuda())
            self.discriminator.append(D128(self.inn_channels).cuda())
            self.discriminator.append(D256(self.inn_channels).cuda())
            self.discriminator.append(D512(self.inn_channels).cuda())
        elif StageNum == 5:
            self.discriminator.append(D64(self.inn_channels).cuda())
            self.discriminator.append(D128(self.inn_channels).cuda())
            self.discriminator.append(D256(self.inn_channels).cuda())
            self.discriminator.append(D512(self.inn_channels).cuda())
            self.discriminator.append(D1024(self.inn_channels).cuda())
        self.generator.apply(models.weights_init)
        for i in range(len(self.discriminator)):
            self.discriminator[i].apply(models.weights_init)
        self.loss = torch.nn.BCELoss().cuda()
        from torch.optim import Adam
        self.gOptimizer = Adam(self.generator.parameters(), lr=generatorLR, betas=(beta1, beta2))
        self.disOptimizer = []
        for i in range(len(self.discriminator)):
            opt = Adam(self.discriminator[i].parameters(), lr=discriminatorLR, betas=(beta1, beta2))
            self.disOptimizer.append(opt)

    def train(self, epochs, batchSize, saveInterval):
        self.trainData = DataLoader(self.dataset, 
                                    batch_size=batchSize, 
                                    shuffle=True, drop_last=True, 
                                    num_workers=0)
        rc = cfg.rowsColums
        imgs = []
        embs = []
        nums = [1992, 1992, 1992, 1992,
                5881, 5881, 5881, 5881,
                7561, 7561, 7561, 7561,
                1225, 1225, 1225, 1225]

        for i in range(rc * rc):
            imgs.append(self.dataset[nums[i]][0][2].reshape(1, 3, 256, 256))
            embs.append(torch.Tensor(self.dataset[nums[i]][2]))

        imgs = torch.Tensor(np.concatenate(imgs))
        embs = torch.stack(embs).cuda()
        embs = self.tile(embs[:rc * rc], rc)
        fixedData =  (imgs, embs)
        noise = torch.Tensor(batchSize, self.zDim).cuda()
        fixedNoise = torch.Tensor(cfg.rowsColums * cfg.rowsColums, self.zDim).normal_(0, 1).cuda()

        real = torch.Tensor(batchSize).fill_(1).cuda()
        fake = torch.Tensor(batchSize).fill_(0).cuda()
        sizes = []
        base = 64
        for i in range(self.StageNum):
            sizes.append(base)
            base = base * 2
        batches = self.trainData.__len__()
        predictions = []
        for epoch in range(epochs):
            totalGenLoss = 0.0
            totalKLloss = 0.0
            totalDisLoss = 0.0
            start = time.time()
            for batch, data in enumerate(self.trainData):
                images = [0, 0, 0, 0, 0]
                for i in range(len(self.discriminator)):
                    images[i] = data[0][i].cuda()
                embeddings = data[2].cuda()

                noise.data.normal_(0, 1)
                genImgs, mu, logvar = self.generator(noise, embeddings)
                mean = mu.detach()
                for i in range(len(self.discriminator)):
                    self.discriminator[i].zero_grad()
                    imgs = images[i]
                    logits, uncondLogits = self.discriminator[i](imgs, mean)
                    realLoss = self.loss(logits, real) + self.loss(uncondLogits, real)
                    logits, uncondLogits = self.discriminator[i](torch.roll(imgs, 1, 0), mean)
                    wrongLoss = self.loss(logits, fake) + self.loss(uncondLogits, real)
                    logits, uncondLogits = self.discriminator[i](genImgs[i].detach(), mean)
                    fakeLoss = self.loss(logits, fake) + self.loss(uncondLogits, fake)
                    disLoss = realLoss + wrongLoss + fakeLoss
                    totalDisLoss += disLoss

                    disLoss.backward()
                    self.disOptimizer[i].step()

                self.generator.zero_grad()
                gLoss = 0
                for i in range(len(self.discriminator)):
                    logits = self.discriminator[i](genImgs[i], mean)
                    gLoss += self.loss(logits[0], real) + self.loss(logits[1], real)
                totalGenLoss += gLoss

                KLloss = models.KLloss(mu, logvar) * cfg.KL
                totalKLloss += KLloss
                gLoss = gLoss + KLloss
                gLoss.backward()
                self.gOptimizer.step()
                if cfg.inception:
                    pred = self.inception_model(genImgs[-1].detach())
                    predictions.append(pred.data.cpu().numpy())
                    if len(predictions) > 100:
                        predictions = np.concatenate(predictions, 0)
                        mean, std = compute_inception_score(predictions, 10)
                        predictions = []
            end = time.time()
            duration = round(end - start, 1)
            print (f"{epoch+1} / {epochs} epoch, Discriminator Loss: {totalDisLoss / batches}, Generator loss: {totalGenLoss / batches}, duration: {duration}s")
            if self.eval_:
                if epoch % saveInterval == 0:
                    self.sampleImages(epoch, fixedNoise, fixedData)
                    torch.save(self.generator.state_dict(), f"models/stackGAN-V2_Generator{epoch}.pyt")
            

    def tile(self, x, n):
        for i in range(n):
            for j in range(1, n):
                x[i * n + j] = x[i * n]
        return x

    def sampleImages(self, epoch, noise, data):
        rc = cfg.rowsColums
        genImgs, mu, logvar = self.generator(noise, data[1])

        for i in range(self.StageNum):
            genImgs[i] = genImgs[i].detach()
        self.saveImages(genImgs, data[0], rc, "Train_", epoch)
    
    def saveImages(self, genImgs, trainImgs, rc, name, epoch):
        gap = 10
        res = cfg.stage3Res
        canvasSizeY = res * rc + (rc * gap)
        canvasSizeX = canvasSizeY * 3 + (res + gap) + gap
        canvas = torch.zeros((canvasSizeY + gap, canvasSizeX, 3), dtype=torch.uint8).cuda()
        genImgs[0] = torch.nn.functional.interpolate(genImgs[0], scale_factor=4, mode="nearest")
        genImgs[1] = torch.nn.functional.interpolate(genImgs[1], scale_factor=2, mode="nearest")
        trainImgs = postprocessing(trainImgs)
        for i in range(self.StageNum):
            genImgs[i] = postprocessing(genImgs[i])
        gapX = gap
        gapY = gap
        cnt = 0
        for i in range(rc):
            canvas[gapY:gapY+res, gapX:gapX+res] = trainImgs[i * rc]

            for j in range(rc):
                for l in range(self.StageNum):
                    gapX += res + gap
                    canvas[gapY:gapY+res, gapX:gapX+res] = genImgs[l][cnt]
                cnt += 1
            gapY += res + gap
            gapX = gap

        cv2.imwrite(f"images/{name}{epoch}.png", canvas.cpu().numpy())