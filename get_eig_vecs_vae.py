
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import DCGAN_VAE_pixel as DVAE
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from hessian3 import hessian
from density_plot2 import get_esd_plot


def KL_div(mu,logvar,reduction = 'none'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1) 
        return KL


def store_NLL(x, recon, mu, logvar, z):
    with torch.no_grad():
        b = x.size(0)
        target = Variable(x.data.view(-1) * 255).long()
        recon = recon.contiguous()
        recon = recon.view(-1,256)
        cross_entropy = F.cross_entropy(recon, target, reduction='none')
        log_p_x_z = -torch.sum(cross_entropy.view(b ,-1), 1)
      
        log_p_z = -torch.sum(z**2/2+np.log(2*np.pi)/2,1)
        z_eps = z - mu
        z_eps = z_eps.view(opt.repeat,-1)
        log_q_z_x = -torch.sum(z_eps**2/2 + np.log(2*np.pi)/2 + logvar/2, 1)
        
        weights = log_p_x_z+log_p_z-log_q_z_x
        
    return weights

def compute_NLL(weights):
    
    with torch.no_grad():
        NLL_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max())))+weights.max()) 
        
    return NLL_loss

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data', help='path to dataset')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

    parser.add_argument('--repeat', type=int, default=200)
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    
    parser.add_argument('--state_E', default='./models/netE_pixel.pth', help='path to encoder checkpoint')
    parser.add_argument('--state_G', default='./models/netE_pixel.pth', help='path to encoder checkpoint')

    parser.add_argument('--state_E_bg', default='./saved_models/cifar/netE_pixel_bg.pth', help='path to encoder checkpoint')
    parser.add_argument('--state_G_bg', default='./saved_models/cifar/netE_pixel_bg.pth', help='path to encoder checkpoint')

    opt = parser.parse_args()
    
    cudnn.benchmark = True
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    '''
    dataset_fmnist = dset.FashionMNIST(root=opt.dataroot, train=Fpytalse, download=True, transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.ToTensor()
                            ]))
    dataloader_fmnist = torch.utils.data.DataLoader(dataset_fmnist, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=int(opt.workers))

    dataset_mnist = dset.MNIST(root=opt.dataroot, train=False, download=True, transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.ToTensor()
                            ]))

    dataloader_mnist = torch.utils.data.DataLoader(dataset_mnist, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=int(opt.workers))
    '''
        
    
    transform_train = transforms.Compose([
                            transforms.Resize(32),
                            
                            transforms.ToTensor()
                        ])

    dataset_cifar_train = dset.CIFAR10(root=opt.dataroot, download=True, train=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(32),
                                           transforms.ToTensor()
                                       ]))
    #dataset_fmnist_train = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_train)
    dataset_fmnist_test = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform_train)
    testloader = torch.utils.data.DataLoader(dataset_fmnist_test, batch_size=32,
                                        shuffle=False, num_workers=6)

    train_cifar100 = torchvision.datasets.CIFAR100(root='data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_cifar100, batch_size=32,
                                              shuffle=True, num_workers=6)

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = 3
    print('Building models...')
    netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu)
    state_G = torch.load(opt.state_G, map_location = device)
    netG.load_state_dict(state_G)
    
    netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu)
    state_E = torch.load(opt.state_E, map_location = device)
    netE.load_state_dict(state_E)
    
    
    netG.to(device)
    netG.eval()
    netE.to(device)
    netE.eval()
    
    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    
    print('Building complete...')
    '''
    First run through the VAE and record the ELBOs of each image in fmnist and mnist
    '''
    NLL_test_indist = []
    NLL_test_indist_bg = []
        
    hessian_comp = hessian(netE, netG, loss_fn, data=None, dataloader = trainloader, cuda=True)

    #density_eigen, density_weight = hessian_comp.density()
    #density_eigen  = np.load('density_eigen.npy')
    #density_weight = np.load('density_weight.npy')
    #get_esd_plot(density_eigen, density_weight)

    eigenvalues, eigenvectors =  hessian_comp.eigenvalues(maxIter=100, tol=1e-3, top_n=50)
    ev = []
    evals = []
    for i in range(len(eigenvectors)):
        ev.append(torch.cat([p.flatten() for p in eigenvectors[i]]).cpu().detach().numpy() )
    np.save('eigenvalues_50_vae_cifar100.npy', np.asarray(eigenvalues) )
    np.save('eigenvectors_50_vae_cifar100.npy', ev )

            
    

      




    
        
    