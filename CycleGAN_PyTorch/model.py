import itertools
import functools

import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from torch.optim import lr_scheduler

'''
Class for CycleGAN with train() as a member function

'''
class cycleGAN(object):
    def __init__(self,args):

        # Define the network 
        #####################################################
        self.Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)

        utils.print_networks([self.Gab,self.Gba,self.Da,self.Db], ['Gab','Gba','Da','Db'])

        # Define Loss criterias

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # Optimizers
        #####################################################
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(),self.Gba.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(),self.Db.parameters()), lr=args.lr, betas=(0.5, 0.999))
        

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0



    def train(self,args):
        # For transforming the input image
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((args.load_height,args.load_width)),
             transforms.RandomCrop((args.crop_height,args.crop_width)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        dataset_dirs = utils.get_traindata_link(args.dataset_dir)

        # Pytorch dataloader
        a_loader = torch.utils.data.DataLoader(dsets.ImageFolder(dataset_dirs['trainA'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)
        b_loader = torch.utils.data.DataLoader(dsets.ImageFolder(dataset_dirs['trainB'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
                # step
                step = epoch * min(len(a_loader), len(b_loader)) + i + 1

                # Generator Computations
                ##################################################

                set_grad([self.Da, self.Db], False)
                self.g_optimizer.zero_grad()

                a_real = Variable(a_real[0])
                b_real = Variable(b_real[0])
                a_real, b_real = utils.cuda([a_real, b_real])

                # Forward pass through generators
                ##################################################
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)

                # Identity losses
                ###################################################
                a_idt_loss = self.L1(a_idt, a_real) * args.lamda * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.lamda * args.idt_coef

                # Adversarial losses
                ###################################################
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                real_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))

                a_gen_loss = self.MSE(a_fake_dis, real_label)
                b_gen_loss = self.MSE(b_fake_dis, real_label)

                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda

                # Total generators losses
                ###################################################
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                # Update generators
                ###################################################
                gen_loss.backward()
                self.g_optimizer.step()


                # Discriminator Computations
                #################################################

                set_grad([self.Da, self.Db], True)
                self.d_optimizer.zero_grad()

                # Sample from history of generated images
                #################################################
                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = utils.cuda([a_fake, b_fake])

                # Forward pass through discriminators
                ################################################# 
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                real_label = utils.cuda(Variable(torch.ones(a_real_dis.size())))
                fake_label = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))

                # Discriminator losses
                ##################################################
                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

                # Total discriminators losses
                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss)*0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5

                # Update discriminators
                ##################################################
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" % 
                                            (epoch, i + 1, min(len(a_loader), len(b_loader)),
                                                            gen_loss,a_dis_loss+b_dis_loss))

            # Override the latest checkpoint
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
    def test(args):
    
        transform = transforms.Compose(
            [transforms.Resize((args.crop_height,args.crop_width)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        dataset_dirs = utils.get_testdata_link(args.dataset_dir)

        a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=transform)
        b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=transform)


        a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

        Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG='resnet_9blocks', norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG='resnet_9blocks', norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)

        utils.print_networks([Gab,Gba], ['Gab','Gba'])

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            Gab.load_state_dict(ckpt['Gab'])
            Gba.load_state_dict(ckpt['Gba'])
        except:
            print(' [*] No checkpoint!')


        """ run """
        a_real_test = Variable(iter(a_test_loader).next()[0], requires_grad=True)
        b_real_test = Variable(iter(b_test_loader).next()[0], requires_grad=True)
        a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
            

        Gab.eval()
        Gba.eval()

        with torch.no_grad():
            a_fake_test = Gab(b_real_test)
            b_fake_test = Gba(a_real_test)
            a_recon_test = Gab(b_fake_test)
            b_recon_test = Gba(a_fake_test)

        pic = (torch.cat([a_real_test, b_fake_test, a_recon_test, b_real_test, a_fake_test, b_recon_test], dim=0).data + 1) / 2.0

        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        torchvision.utils.save_image(pic, args.results_dir+'/sample.jpg', nrow=3)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('Network initialized with weights sampled from N(0,0.02).')
    net.apply(init_func)


def init_network(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net)
    return net


def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.LeakyReLU(0.2,True))

def conv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.ReLU(True))

def dconv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0, output_padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias = bias),
        norm_layer(out_dim), nn.ReLU(True))


class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1),
                     conv_norm_relu(dim, dim, kernel_size=3, 
                     norm_layer= norm_layer, bias=use_bias)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                      norm_layer(dim)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

###DISCRIMINATOR

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(NLayerDiscriminator, self).__init__()
        dis_model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2,
                                               norm_layer= norm_layer, padding=1, bias=use_bias)]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1,
                                               norm_layer= norm_layer, padding=1, bias=use_bias)]
        dis_model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, input):
        return self.dis_model(input)

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(PixelDiscriminator, self).__init__()
        dis_model = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, input):
        return self.dis_model(input)



def define_Dis(input_nc, ndf, netD, n_layers_D=3, norm='batch', gpu_ids=[0]):
    dis_net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    if netD == 'n_layers':
        dis_net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_bias=use_bias)
    elif netD == 'pixel':
        dis_net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_bias=use_bias)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

    return init_network(dis_net, gpu_ids)

###GENERATORS

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, 
                                innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [nn.ReLU(True), upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.unet_model = unet_block

    def forward(self, input):
        return self.unet_model(input)



class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, num_blocks=6):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        res_model = [nn.ReflectionPad2d(3),
                    conv_norm_relu(input_nc, ngf * 1, 7, norm_layer=norm_layer, bias=use_bias),
                    conv_norm_relu(ngf * 1, ngf * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias),
                    conv_norm_relu(ngf * 2, ngf * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)]

        for i in range(num_blocks):
            res_model += [ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)]

        res_model += [dconv_norm_relu(ngf * 4, ngf * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      dconv_norm_relu(ngf * 2, ngf * 1, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      nn.ReflectionPad2d(3),
                      nn.Conv2d(ngf, output_nc, 7),
                      nn.Tanh()]
        self.res_model = nn.Sequential(*res_model)

    def forward(self, x):
        return self.res_model(x)




def define_Gen(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, gpu_ids=[0]):
    gen_net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        gen_net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_blocks=9)
    elif netG == 'resnet_6blocks':
        gen_net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_blocks=6)
    elif netG == 'unet_128':
        gen_net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        gen_net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_network(gen_net, gpu_ids)