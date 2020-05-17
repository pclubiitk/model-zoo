import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torch.nn.functional as F

class BasicModule(nn.Module):
    """
    basic block with identity maps in shortcuts
    """
    def __init__(self, in_planes, out_planes, stride=1, option='A'):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        if stride==1 and in_planes==out_planes:
            if option != 'C': 
                self.shortcut = nn.Sequential()
            else:
                self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride = stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            if option == 'A':
                self.shortcut = lambda x: F.pad(x[:,:,::2,::2], (0,0,0,0,(out_planes-in_planes)//2,(out_planes-in_planes)//2))
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride = stride, bias=False),
                    nn.BatchNorm2d(out_planes)
                )
    
    def forward(self, x):
        x_short = x
        x = F.celu(self.bn1(self.conv1(x)),alpha=0.075)
        x = self.bn2(self.conv2(x))
        x += self.shortcut(x_short)
        return F.celu(x,alpha= 0.075)

class BottleNeckModule(nn.Module):
    """
    basic block with identity maps in shortcuts
    """
    def __init__(self, in_planes, out_planes, stride = 1, option='A'):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes, 1, padding=0, bias=False, stride=stride)
        self.bn3 = nn.BatchNorm2d(out_planes)

        if stride==1 and in_planes==out_planes:
            if option != 'C': 
                self.shortcut = nn.Sequential()
            else:
                self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride = stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            if option == 'A':
                self.shortcut = lambda x: F.pad(x[:,:,::2,::2], (0,0,0,0,(out_planes-in_planes)//2,(out_planes-in_planes)//2))
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride = stride, bias=False),
                    nn.BatchNorm2d(out_planes)
                )

    def forward(self, x):
        x_short = x
        x = F.celu(self.bn1(self.conv1(x)),alpha = 0.075)
        x = F.celu(self.bn2(self.conv2(x)),alpha = 0.075)
        x = self.bn3(self.conv3(x))
        x += self.shortcut(x_short)
        x = F.celu(x,alpha = 0.075)
        return x


class ResNet(nn.Module):

  def __init__(self,block,filter_map,n,num_classes=10,option='A'):

    super().__init__()

    self.conv1  = nn.Conv2d(in_channels=3,out_channels=filter_map[0],kernel_size=3,padding=1,bias=False)
    self.bn1 = nn.BatchNorm2d(filter_map[0])
    self.block1 = self.MakeResNetLayer(block,filter_map[0],n,stride=1,option=option)
    #self.drop1  = nn.Dropout2d(0.3)
    self.block2 = self.MakeResNetLayer(block,(filter_map[0],filter_map[1]),n,stride=2,option=option)
    #self.drop2  = nn.Dropout2d(0.2)
    self.block3 = self.MakeResNetLayer(block,(filter_map[1],filter_map[2]),n,stride=2,option=option)
    self.drop3  = nn.Dropout2d(0.25)
    self.globalavgpool = nn.AdaptiveAvgPool2d(2)

    #self.drop1  = nn.Dropout(0.3)
    self.fc = nn.Linear(2*2*filter_map[2],num_classes)  

  def MakeResNetLayer(self,block,filters,n,stride,option='A'):

    if stride!=1 :
      in_planes,out_planes = filters
    else :
      in_planes,out_planes = filters,filters

    layer = []
    layer.append(block(in_planes,out_planes,stride, option=option))

    for i in range(n-1):

      layer.append(block(out_planes, out_planes, option=option))

    SubBlock = nn.Sequential(*layer)

    return SubBlock

  def  forward(self,x):

    x = F.relu(self.bn1(self.conv1(x)))
        
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
        
    x = self.globalavgpool(x)
    x = x.view(-1, self.find_shape(x))
    x = self.fc(x)
    return x

  def find_shape(self, x):
    res = 1
    for dim in x[0].shape:
        res *= dim
    return res          
