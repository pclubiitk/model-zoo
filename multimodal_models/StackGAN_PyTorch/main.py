import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataloader import TextDataset
import models
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--image_size1', type=int, default=64,help='stage1 image size')
parser.add_argument('--image_size2', type=int, default=256,help='stage2 image size')
parser.add_argument('--z_dim', type=int, default=100,help='noise dimension')
parser.add_argument('--embed_dim', type=int, default=128,help='embedding compressed dim')
parser.add_argument('--D1_dim', type=int, default=512,)
parser.add_argument('--G2_dim', type=int, default=512)
parser.add_argument('--data_dir', type=str, required=True,help='path to embedding directory containing pickle files')
parser.add_argument('--image_dir', type=str, required=True,help='path to folder containing jpg folder which contains images')
parser.add_argument('--epochs1', type=int, default=100)
parser.add_argument('--epochs2', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lrG1', type=float, default=0.0002)
parser.add_argument('--lrG2', type=float, default=0.0002)
parser.add_argument('--lrD1', type=float, default=0.0002)
parser.add_argument('--lrD2', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_transform1 = transforms.Compose([
            transforms.RandomCrop(args.image_size1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
image_transform2 = transforms.Compose([
            transforms.RandomCrop(args.image_size2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = TextDataset(args.image_dir,args.data_dir,image_transform1,image_transform2 , 'train', args.image_size1,args.image_size2)            

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def show_and_save(file_name,img):
    npimg = np.transpose(img.numpy(),(1,2,0))
    f = "/content/drive/My Drive/%s.png" % file_name
    fig = plt.figure(dpi=250)
    fig.suptitle(file_name, fontsize=14, fontweight='bold')
    plt.imshow(npimg)
    plt.imsave(f,npimg)            

'''
STAGE 1
'''

netG1=models.G1(args.embed_dim,args.z_dim,args.image_size1,(args.embed_dim+args.z_dim)*8).to(device)
netG1.apply(weights_init)
netD1=models.D1(args.image_size1,args.D1_dim,args.embed_dim).to(device)
netD1.apply(weights_init)

netG1_para = []
for p in netG1.parameters():
  if p.requires_grad:
    netG1_para.append(p)

optimG1=torch.optim.Adam(netG1_para,lr=args.lrG1,betas=(args.beta1, args.beta2))
optimD1=torch.optim.Adam(netD1.parameters(),lr=args.lrD1,betas=(args.beta1, args.beta2))
criterion=nn.BCELoss().to(device)    

def KL(logvar,mu):
   KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
   KLD = torch.mean(KLD_element).mul_(-0.5)
   return KLD

ones=torch.ones(args.batch_size).to(device)
zeros=torch.zeros(args.batch_size).to(device)   
z_fixed=torch.randn(args.batch_size,args.z_dim).to(device)

for epoch in range(args.epochs1):
  G1_loss=[]
  D1_loss=[]
  for i,data in enumerate(dataloader):
    data[0]=data[0].to(device)
    data[2]=data[2].to(device)
    z=torch.randn(args.batch_size,args.z_dim).to(device)
    gen_img,mean,logvar=netG1(data[2],z)
    out_fake=netD1(gen_img,data[2])
    out_real=netD1(data[0],data[2])
    lossD=criterion(out_fake,zeros)+criterion(out_real,ones)
    optimD1.zero_grad()
    lossD.backward(retain_graph=True)
    optimD1.step()
    D1_loss.append(lossD.item())


    gen_img,mean,logvar=netG1(data[2],z)
    out_fake=netD1(gen_img,data[2])
    kl=KL(logvar,mean).to(device)
    lossG=criterion(out_fake,zeros)+KL(logvar,mean).to(device)
    G1_loss.append(lossG.item())
    optimG1.zero_grad()
    lossG.backward()
    optimG1.step()
    if i%10==0:
      print("Epoch [%d/%d] Step[%d/%d] G_loss:%f D_loss:%f KL: %f"%(epoch+1,args.epochs1,i+1,len(dataloader),lossG.item(),lossD.item(),kl.item()) )
      gen,_,_=netG(data[2],z_fixed)
      gen=gen.detach()
      show_and_save("%d"%(i) ,make_grid((gen*0.5+0.5).cpu(),8))



'''
STAGE 2
'''

gen_img64=[]
for i,data in enumerate(dataloader):
  data[2]=data[2].to(device)
  z=torch.randn(args.batch_size,args.z_dim).to(device)
  gen_img,mean,logvar=netG1(data[2],z) 
  gen_img=gen_img.detach()
  gen_img64.append(gen_img)

netG2=models.G2(64,512,256,512+128).to(device)
netG2.apply(weights_init)
netD2=models.D2(256,512,128).to(device)
netD2.apply(weights_init)  

optimG2=torch.optim.Adam(netG2.parameters(),lr=args.lrG2,betas=(args.beta1,args.beta2))
optimD2=torch.optim.Adam(netD2.parameters(),lr=args.lrD2,betas=(args.beta1,args.beta2))
criterion=nn.BCELoss().to(device)

for epoch in range(args.epochs2):
  G2_loss=[]
  D2_loss=[]
  for i,data in enumerate(dataloader):
    data[2]=data[2].to(device)
    data[1]=data[1].to(device)
    gen_img64[i]=gen_img64[i].to(device)
    gen_img,mean,logvar=netG2(data[2],gen_img64[i])
    out_fake=netD2(gen_img,data[2])
    out_real=netD2(data[1],data[2])
    lossD=criterion(out_fake,zeros)+criterion(out_real,ones)
    optimD2.zero_grad()
    lossD.backward(retain_graph=True)
    optimD2.step()
    D2_loss.append(lossD.item())


    gen_img,mean,logvar=netG2(data[2],gen_img64[i])
    out_fake=netD2(gen_img,data[2])
    kl=KL(logvar,mean).to(device)
    lossG=criterion(out_fake,zeros)+KL(logvar,mean).to(device)
    G2_loss.append(lossG.item())
    optimG2.zero_grad()
    lossG.backward()
    optimG2.step()
    
    if i%10==0:
      print("Epoch [%d/%d] Step[%d/%d] G_loss:%f D_loss:%f KL: %f"%(epoch+1,args.epochs2,i+1,len(dataloader),lossG.item(),lossD.item(),kl.item()) )
