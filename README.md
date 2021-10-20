# LMPBT
Supplementary code for the Paper entitled ``Locally Most Powerful Bayesian Test for Out-of-Distribution Detection using Deep Generative Models"

1. Specification of dependencies 
Pytorch 1.7.1 torchvision 
0.8.2 Scikit-learn 
0.22 PyHessian (See, https://github.com/amirgholami/PyHessian.) 
pip install torch==1.7.1 torchvision==0.8.2 pip install pyhessian pip install scikit-learn 

2. Dataset MNIST, FASHION-MNIS, CIFAR-10, CIFAR-100, and SVHN : downloaded from torchvision.datasets 
CelebA : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 

3. Training code We refer to the following code for model training : VAE : https://github.com/XavierXiao/Likelihood-Regret 

To train the VAEs, use appropriate arguments and run this command: python train_vae.py 

We refer to the following code for computing the low-rank-approximation of Hessian : https://github.com/amirgholami/PyHessian 

To compute the top eigenvalues and eigenvectors of VAE models, run this command: python get_eig_vecs_vae.py 

4. Evaluation code To comput the LMPBT-scores using the VAE models, run this command: python get_lmpbt_score.py 

5. We provided the LMPBT socres of a VAE trained on CIFAR-100 and tested on CIFAR-100 as an in-distribution dataset, SVHN and CelebA as an OOD dataset. 
To comput the OOD performance metrics (AUROC, AUPR and FPR80) of these experiments, run this command: python get_metrics.py 

5. Pre-trained models We provided the pretrained VAE, trained on CIFAR-100. 

The model can be loaded using the following code. 

parser.add_argument('--state_E', default='./models/cifar100_netE.pth', help='path to encoder checkpoint') 

parser.add_argument('--state_G', default='./models/cifar100_netG.pth', help='path to encoder checkpoint') 

netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu) 

state_G = torch.load(opt.state_G, map_location=device) 

netG.load_state_dict(state_G) 

netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu) 

state_E = torch.load(opt.state_E, map_location=device) 

netE.load_state_dict(state_E)
