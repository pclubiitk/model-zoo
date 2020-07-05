import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CA(nn.Module):
  def __init__(self,indim,outdim):
    super().__init__()
    self.outdim=outdim
    self.fc1=nn.Linear(indim,outdim*2)
    self.relu=nn.ReLU()
  
  def forward(self,x):
    outdim=self.outdim
    out=self.relu(self.fc1(x))
    mean=out[:,:outdim]
    logvar=out[:,outdim:]
    epsilon=torch.randn(x.size()[0],outdim).to(device)
    std= logvar.mul(0.5).exp_()
    output=mean+epsilon*std

    return output,mean,logvar
    
class G1(nn.Module):
  def __init__(self,indim,zdim,imsize,inchnls):
    super().__init__()
    self.ca=CA(1024,indim)
    self.indim=indim
    self.zdim=zdim
    self.imsize=imsize
    self.inchnls=(indim+zdim)*8
    self.fc1=nn.Linear(indim+zdim,(indim+zdim)*4*4*8)
    self.bn=nn.BatchNorm1d((indim+zdim)*4*4*8)
    self.relu=nn.ReLU()

    self.up1=nn.UpsamplingNearest2d(scale_factor=2)
    self.conv1=nn.Conv2d(inchnls,inchnls//2,stride=1,padding=1,kernel_size=3)
    self.bn1=nn.BatchNorm2d(inchnls//2)

    self.up2=nn.UpsamplingNearest2d(scale_factor=2)
    self.conv2=nn.Conv2d(inchnls//2,inchnls//4,stride=1,padding=1,kernel_size=3)
    self.bn2=nn.BatchNorm2d(inchnls//4)

    self.up3=nn.UpsamplingNearest2d(scale_factor=2)
    self.conv3=nn.Conv2d(inchnls//4,inchnls//8,stride=1,padding=1,kernel_size=3)
    self.bn3=nn.BatchNorm2d(inchnls//8)

    self.up4=nn.UpsamplingNearest2d(scale_factor=2)
    self.conv4=nn.Conv2d(inchnls//8,3,stride=1,padding=1,kernel_size=3)
    self.bn4=nn.BatchNorm2d(3)
    self.tanh=nn.Tanh()


  def forward(self,x,z):
    out,mean,logvar=self.ca(x)
    out=torch.cat((out,z),1)

    out=self.relu(self.bn(self.fc1(out)))

    out=out.view(-1,(self.indim+self.zdim)*8,4,4)

    out=self.relu(self.bn1(self.conv1(self.up1(out))))
    out=self.relu(self.bn2(self.conv2(self.up2(out))))
    out=self.relu(self.bn3(self.conv3(self.up3(out))))
    out=self.tanh(self.bn4(self.conv4(self.up4(out))))
    
    return out,mean,logvar

class EmbedComp(nn.Module):
  def __init__(self,insize,outsize,md):
    super().__init__()
    self.fc1=nn.Linear(insize,outsize)
    self.outsize=outsize
    self.md=md
  def forward(self,x):
    out=self.fc1(x)
    out=out.view(-1,self.outsize,1,1)
    out=out.repeat(1,1,self.md,self.md)
    return out

class D1(nn.Module):
  def __init__(self,imsize,dim,emdim):
    super().__init__()
    self.conv1=nn.Conv2d(3,dim//8,kernel_size=4,stride=2,padding=1)
    self.lrelu=nn.LeakyReLU(0.2)
    self.conv2=nn.Conv2d(dim//8,dim//4,kernel_size=4,stride=2,padding=1)
    self.bn2=nn.BatchNorm2d(dim//4)
    self.conv3=nn.Conv2d(dim//4,dim//2,kernel_size=4,stride=2,padding=1)
    self.bn3=nn.BatchNorm2d(dim//2)
    self.conv4=nn.Conv2d(dim//2,dim,kernel_size=4,stride=2,padding=1)
    self.bn4=nn.BatchNorm2d(dim)

    self.embedc=EmbedComp(1024,emdim,4).to(device)
    self.conv5=nn.Conv2d(dim+emdim,dim,kernel_size=1)
    self.bn5=nn.BatchNorm2d(dim)

    self.conv6=nn.Conv2d(dim,1,kernel_size=4,stride=4)
    self.sigmoid=nn.Sigmoid()


  def forward(self,img,embed):

    out=self.lrelu(self.conv1(img))
    out=self.lrelu(self.bn2(self.conv2(out)))
    out=self.lrelu(self.bn3(self.conv3(out)))
    out=self.lrelu(self.bn4(self.conv4(out)))

    em=self.embedc(embed)
    out=torch.cat((out,em),1)
    out=self.lrelu(self.bn5(self.conv5(out)))

    out=self.sigmoid(self.conv6(out) )
    out=out.view(-1)

    return out

class ResBlock(nn.Module):
  def __init__(self,inchnls):
    super().__init__()
    self.conv1=nn.Conv2d(inchnls,inchnls,kernel_size=3,stride=1,padding=1)
    self.bn1=nn.BatchNorm2d(inchnls)
    self.relu=nn.ReLU()
    self.conv2=nn.Conv2d(inchnls,inchnls,kernel_size=3,stride=1,padding=1)
    self.bn2=nn.BatchNorm2d(inchnls)
  
  def forward(self,x):
    x1=x
    out=self.relu(self.bn1(self.conv1(x)))
    out=(self.bn2(self.conv2(out)))

    out=self.relu(out+x1)
    return out

class G2(nn.Module):
  def __init__(self,imsize,chnls,outsize,inchnls,embed_dim):
    super().__init__()
    self.ca=CA(1024,embed_dim)
    self.embed_dim=embed_dim
    self.conv1=nn.Conv2d(3,chnls//4,kernel_size=4,stride=2,padding=1)
    self.bn1=nn.BatchNorm2d(chnls//4)
    self.lrelu=nn.LeakyReLU(0.2)
    self.conv2=nn.Conv2d(chnls//4,chnls,kernel_size=4,stride=2,padding=1)
    self.bn2=nn.BatchNorm2d(chnls)
    self.res1=ResBlock(chnls+embed_dim)
    self.res2=ResBlock(chnls+embed_dim)
    
    self.relu=nn.ReLU()
    self.up1=nn.UpsamplingNearest2d(scale_factor=2)
    self.conva=nn.Conv2d(inchnls,inchnls//2,stride=1,padding=1,kernel_size=3)
    self.bna=nn.BatchNorm2d(inchnls//2)

    self.up2=nn.UpsamplingNearest2d(scale_factor=2)
    self.convb=nn.Conv2d(inchnls//2,inchnls//4,stride=1,padding=1,kernel_size=3)
    self.bnb=nn.BatchNorm2d(inchnls//4)

    self.up3=nn.UpsamplingNearest2d(scale_factor=2)
    self.convc=nn.Conv2d(inchnls//4,inchnls//8,stride=1,padding=1,kernel_size=3)
    self.bnc=nn.BatchNorm2d(inchnls//8)

    self.up4=nn.UpsamplingNearest2d(scale_factor=2)
    self.convd=nn.Conv2d(inchnls//8,3,stride=1,padding=1,kernel_size=3)
    self.bnd=nn.BatchNorm2d(3)
    self.tanh=nn.Tanh()


  def forward(self,x,img):
    out,mean,logvar=self.ca(x)
    out=out.view(-1,self.embed_dim,1,1)
    out=out.repeat(1,1,16,16)
    img=self.lrelu(self.bn1(self.conv1(img)))
    img=self.lrelu(self.bn2(self.conv2(img)))
    out=torch.cat((img,out),1)

    out=self.res1(out)
    out=self.res2(out)

    out=self.relu(self.bna(self.conva(self.up1(out))))
    out=self.relu(self.bnb(self.convb(self.up2(out))))
    out=self.relu(self.bnc(self.convc(self.up3(out))))
    out=self.tanh(self.bnd(self.convd(self.up4(out))))
    
    return out,mean,logvar

class D2(nn.Module):
  def __init__(self,imsize,dim,emdim):
    super().__init__()
    self.conv1=nn.Conv2d(3,dim//32,kernel_size=4,stride=2,padding=1)
    self.lrelu=nn.LeakyReLU(0.2)
    self.conv2=nn.Conv2d(dim//32,dim//16,kernel_size=4,stride=2,padding=1)
    self.bn2=nn.BatchNorm2d(dim//16)
    self.conv3=nn.Conv2d(dim//16,dim//8,kernel_size=4,stride=2,padding=1)
    self.bn3=nn.BatchNorm2d(dim//8)
    self.conv4=nn.Conv2d(dim//8,dim//4,kernel_size=4,stride=2,padding=1)
    self.bn4=nn.BatchNorm2d(dim//4)
    self.conv5=nn.Conv2d(dim//4,dim//2,kernel_size=4,stride=2,padding=1)
    self.bn5=nn.BatchNorm2d(dim//2)
    self.conv6=nn.Conv2d(dim//2,dim,kernel_size=4,stride=2,padding=1)
    self.bn6=nn.BatchNorm2d(dim)

    self.embedc=EmbedComp(1024,emdim,4).to(device)
    self.conv7=nn.Conv2d(dim+emdim,dim,kernel_size=1)
    self.bn7=nn.BatchNorm2d(dim)

    self.conv8=nn.Conv2d(dim,1,kernel_size=4,stride=4)
    self.sigmoid=nn.Sigmoid()


  def forward(self,img,embed):

    out=self.lrelu(self.conv1(img))
    out=self.lrelu(self.bn2(self.conv2(out)))
    out=self.lrelu(self.bn3(self.conv3(out)))
    out=self.lrelu(self.bn4(self.conv4(out)))
    out=self.lrelu(self.bn5(self.conv5(out)))
    out=self.lrelu(self.bn6(self.conv6(out)))

    em=self.embedc(embed)
    out=torch.cat((out,em),1)
    out=self.lrelu(self.bn7(self.conv7(out)))

    out=self.sigmoid(self.conv8(out) )
    out=out.view(-1)

    return out



