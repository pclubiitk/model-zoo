import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.exposure import match_histograms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from utils import *


def train_st(config, dataloader, style_img, sample_img, stnet, lossnet, optimizer):
    a_S = lossnet(style_img)
    steps = 0
    net_style_loss, net_content_loss, net_reg_loss = [0.0], [0.0], [0.0]

    stnet.train()
    while True:
        for x, _ in dataloader:
            steps += 1

            x = x.to(config.device)
            y = stnet(x)

            a_C = lossnet(x)
            a_G = lossnet(y)

            content_loss = config.content_weight * fr_loss(a_C[1], a_G[1])
            style_loss = 0.0
            for i, w in enumerate(config.style_weights):
                style_loss += w * sr_loss(a_S[i], a_G[i])
            reg_loss = config.reg_weight * tvr_loss(y)
            loss = content_loss + style_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net_style_loss[-1] += style_loss.item()
            net_content_loss[-1] += content_loss.item()
            net_reg_loss[-1] += reg_loss.item()

            if steps % config.log_step == 0:
                print(
                    "[{}/{}] content: {:.4f} style: {:.4f} reg: {:.4f} total_loss: {:.4f}".format(
                        steps,
                        config.iterations,
                        net_content_loss[-1] / config.log_step,
                        net_style_loss[-1] / config.log_step,
                        net_reg_loss[-1] / config.log_step,
                        (net_content_loss[-1] + net_style_loss[-1] + net_reg_loss[-1])
                        / config.log_step,
                    )
                )
                net_content_loss.append(0.0)
                net_style_loss.append(0.0)
                net_reg_loss.append(0.0)

            if steps % config.save_step == 0:
                torch.save(stnet.state_dict(), config.model_path + "model_st.pth")
                stnet.eval()
                with torch.no_grad():
                    y = stnet(sample_img)
                    Image.Image.save(
                        Image.fromarray(tensor_to_image(y.data.cpu().numpy())[0]),
                        config.sample_path + "sample_{}.jpg".format(steps),
                    )
                stnet.train()

            if steps > config.iterations:
                losses = (net_content_loss, net_style_loss, net_reg_loss)
                return losses


def stylize(config, dataloader, stnet, samples=5):
    stnet.eval()
    fig = plt.figure(figsize=(40.0, 40.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(samples, 8), axes_pad=0.1)

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            x = x.to(config.device)
            y = stnet(x)

            grid[8 * i].imshow(tensor_to_image(x.data.cpu().numpy())[0])
            grid[8 * i + 1].imshow(tensor_to_image(y.data.cpu().numpy())[0])
            grid[8 * i + 2].imshow(tensor_to_image(x.data.cpu().numpy())[1])
            grid[8 * i + 3].imshow(tensor_to_image(y.data.cpu().numpy())[1])
            grid[8 * i + 4].imshow(tensor_to_image(x.data.cpu().numpy())[2])
            grid[8 * i + 5].imshow(tensor_to_image(y.data.cpu().numpy())[2])
            grid[8 * i + 6].imshow(tensor_to_image(x.data.cpu().numpy())[3])
            grid[8 * i + 7].imshow(tensor_to_image(y.data.cpu().numpy())[3])

            if i == samples - 1:
                break

    plt.show()


def train_sr(config, dataloader, srnet, lossnet, optimizer, c_x):
    steps = 0
    net_content_loss, net_reg_loss = [0.0], [0.0]

    srnet.train()
    while True:
        for x, y in dataloader:
            steps += 1

            x = x.to(config.device)
            y = y.to(config.device)
            y_hat = srnet(x)

            a_C = lossnet(y)
            a_G = lossnet(y_hat)

            content_loss = config.content_weight * fr_loss(a_C[1], a_G[1])
            reg_loss = config.reg_weight * tvr_loss(y_hat)
            loss = content_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net_content_loss[-1] += content_loss
            net_reg_loss[-1] += reg_loss

            if steps % config.log_step == 0:
                print(
                    "[{}/{}] content: {:.4f} reg: {:.4f} total_loss: {:.4f}".format(
                        steps,
                        config.iterations,
                        net_content_loss[-1] / config.log_step,
                        net_reg_loss[-1] / config.log_step,
                        (net_content_loss[-1] + net_reg_loss[-1]) / config.log_step,
                    )
                )
                net_content_loss.append(0.0)
                net_reg_loss.append(0.0)

            if steps % config.save_step == 0:
                torch.save(srnet.state_dict(), config.model_path + 'model.pth')
                srnet.eval()
                with torch.no_grad():
                    y = srnet(c_x)
                    c_x_resized = transforms.Resize(config.image_size)(c_x)
                    sample_img = match_histograms(tensor_to_image(y.data.cpu().numpy())[0], 
                                                tensor_to_image(c_x_resized.data.cpu().numpy())[0], 
                                                multichannel=True)
                    Image.Image.save(
                        Image.fromarray(sample_img),
                        config.sample_path + "sample_{}.jpg".format(steps),
                    )
                srnet.train()

            if steps > config.iterations:
                losses = (net_content_loss, net_reg_loss)
                return losses


def enhance(config, dataloader, srnet, samples=5):
    samples = 5
    srnet.eval()

    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(fig, 111, nrows_ncols=(samples, 4), axes_pad=0.1)

    with torch.no_grad():
        for i, (x, orig) in enumerate(dataloader):
            x = x.to(config.device)
            y = srnet(x)
            x_resized = transforms.Resize(config.image_size)(x)
            x_resized_bc = transforms.Resize(config.image_size, interpolation=InterpolationMode.BICUBIC)(x)
            sample_img = match_histograms(tensor_to_image(y.data.cpu().numpy())[0], 
                                           tensor_to_image(x_resized_bc.data.cpu().numpy())[0], 
                                           multichannel=True)
            grid[4*i].imshow(tensor_to_image(x_resized.data.cpu().numpy())[0])
            grid[4*i+1].imshow(tensor_to_image(x_resized_bc.data.cpu().numpy())[0])
            grid[4*i+2].imshow(sample_img)
            grid[4*i+3].imshow(tensor_to_image(orig.data.cpu().numpy())[0])
            
            if i == samples - 1:
                break
    
    plt.show()
