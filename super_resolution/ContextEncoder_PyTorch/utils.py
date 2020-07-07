import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
import torch

inv_transform = transforms.Normalize((-1,-1,-1),(2,2,2)) # For restoring images to normal range [0,1]

def weights_init_normal(m):
  '''
  initializes weights of layers according to guide lines from DCGAN paper(Radford et. al,2015)
  '''
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
      torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find("BatchNorm2d") != -1:
      torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
      torch.nn.init.constant_(m.bias.data, 0.0)

def show(x,nrow=8):
  '''
  Shows an image
  '''
  for i in range(x.size(0)):
    x[i] = inv_transform(x[i])

  img = make_grid(x.cpu().detach(),nrow=nrow)

  plt.figure(figsize=(50,25))
  plt.imshow(img.permute(1,2,0))
  
def real_data_target(size,device,label_smoothing=False):
    '''
    Tensor containing ones, with shape = size
    '''
    num  = np.random.uniform(0.7,1)
    data = torch.ones((1,1))
    if label_smoothing :
      data = data.new_full((size,1),num)
    else :   
      data = data.new_full((size,1),1)

    return data.to(device)

def fake_data_target(size,device,label_smoothing=False):
    '''
    Tensor containing zeros, with shape = size
    '''
    num  = np.random.uniform(0,0.3)
    data = torch.ones((1,1))
    if label_smoothing :
      data = data.new_full((size,1),num)
    else :   
      data = data.new_full((size,1),0)

    return data.to(device)  

def log_images(writer,dataloader,mask_size,device,generator,current_step):
  '''
  logs image to a summary writer
  '''
  samples,masked_samples,i = next(iter(dataloader))
  i = i[0].item() # gets the coordinate for top left pixel
  masked_samples = masked_samples.to(device)

  gene_image = generator(masked_samples)
  clone_sample = masked_samples.clone().cpu()
  clone_sample[:,:,i:i+mask_size,i:i+mask_size] = gene_image
  for i in range(clone_sample.size(0)):
    clone_sample[i] = inv_transform(clone_sample[i])
    samples[i] = inv_transform(samples[i])  

  grid_image = make_grid(clone_sample,nrow=4)
  grid_samples = make_grid(samples,nrow=4)
  writer.add_image('original images',grid_samples,current_step)
  writer.add_image('test images',grid_image,current_step)


