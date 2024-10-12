import argparse
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import time
import random
import os
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch import autograd
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import math
import dp_optimizer
import analysis
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
parser = argparse.ArgumentParser()
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn import metrics
# experimentName is the current file name without extension
experimentName = os.path.splitext(os.path.basename(__file__))[0]
experimentName = 'rdp'

# parser.add_argument("--DATASETPATH", type=str,
#                     default=os.path.expanduser('~/data/MIMIC/processed/out_binary.matrix'),
#                     help="Dataset file")
parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('D:/DP-GAN/cardio_train.csv'),
                    help="Dataset file")

parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--n_epochs_pretrain", type=int, default=50,
                    help="number of epochs of pretraining the autoencoder")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.00001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument('--n_iter_D', type=int, default=5, help='number of D iters per each G iter')

# Check the details
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

parser.add_argument("--cuda", type=bool, default=True,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=True,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")

parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent noise space")
parser.add_argument("--feature_size", type=int, default=1071, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between batches")
parser.add_argument("--epoch_time_show", type=bool, default=True, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=1, help="number of epops per model save")
parser.add_argument("--minibatch_averaging", type=bool, default=False, help="Minibatch averaging")

#### Privacy
parser.add_argument('--noise_multiplier', type=float, default=0.5)
parser.add_argument('--max_per_sample_grad_norm', type=float, default=1.0)
parser.add_argument('--delta', type=float, default=1e-5, help="Target delta (default: 1e-5)")

# Training/Testing
parser.add_argument("--pretrained_status", type=bool, default=False, help="If want to use ae pretrained weights")
parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=True, help="Generating Sythetic Data")
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")
# parser.add_argument("--expPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/' + experimentName),help="Experiment path")
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('wadi/'),
                    help="Experiment path")
parser.add_argument("--modelPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/' + experimentName + '/model'),
                    help="Model path")
opt = parser.parse_args()
print(opt)

# Create experiments DIR
if not os.path.exists(opt.expPATH):
    os.system('mkdir -p {0}'.format(opt.expPATH))

# Random seed for pytorch
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
cudnn.benchmark = True

# Check cuda
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device BUT it is not in use...")

# Activate CUDA
device = torch.device("cuda:0" if opt.cuda else "cpu")


##########################
### Dataset Processing ###
##########################


trainData=pd.read_csv(os.path.join(opt.expPATH, "dataTrain.csv"))
testData=pd.read_csv(os.path.join(opt.expPATH, "dataTest.csv"))
print(trainData.shape)
pd.set_option('display.max_columns', None)
trainData = trainData.to_numpy()
testData = testData.to_numpy()
trainData = trainData.astype(np.float32)
testData = testData.astype(np.float32)
sc = MinMaxScaler()
trainData = sc.fit_transform(trainData)
testData = sc.transform(testData)
# trainData[trainData >= 0.5] = 1.0
# trainData[trainData < 0.5] = 0.0
class Dataset:
    def __init__(self, data, transform=None):

        # Transform
        self.transform = transform

        # load data here
        self.data = data
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]

    def return_data(self):
        return self.data
        # return np.clip(self.data, 0, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]


        if self.transform:
            pass

        return torch.from_numpy(sample)


# Train data loader
dataset_train_object = Dataset(data=trainData, transform=False)

samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
dataloader_train = DataLoader(dataset_train_object, batch_size=opt.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)

# Test data loader
dataset_test_object = Dataset(data=testData, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
dataloader_test = DataLoader(dataset_test_object, batch_size=opt.batch_size,
                             shuffle=True, num_workers=0, drop_last=True)

# Generate random samples for test
random_samples = next(iter(dataloader_test))
feature_size = random_samples.size()[1]

###########################
## Privacy Calculation ####
###########################
totalsamples = len(dataset_train_object)

num_batches = len(dataloader_train)
iterations = opt.n_epochs * num_batches
print('Achieves ({}, {})-DP'.format(
        analysis.epsilon(
            totalsamples,
            opt.batch_size,
            opt.noise_multiplier,
            iterations,
            opt.delta
        ),
        opt.delta,
    ))

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        n_channels_base = 8

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_channels_base, kernel_size=8, stride=2, padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            # nn.LeakyReLU(1 / 5.5, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base, out_channels=2 * n_channels_base, kernel_size=7, stride=2,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.PReLU(),
            nn.Conv1d(in_channels=n_channels_base*2, out_channels=4 * n_channels_base, kernel_size=7, stride=2,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base * 4, out_channels=8 * n_channels_base, kernel_size=7, stride=2,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base * 8, out_channels=n_channels_base * 16, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=n_channels_base * 16, out_channels=8 * n_channels_base, kernel_size=7,
                               stride=1, padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=n_channels_base * 8, out_channels=4 * n_channels_base, kernel_size=8,
                               stride=2, padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=4 * n_channels_base, out_channels=2*n_channels_base, kernel_size=7,
                               stride=3, padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),

            nn.BatchNorm1d(n_channels_base*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=2 * n_channels_base, out_channels=n_channels_base, kernel_size=6,
                               stride=2, padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),

            nn.BatchNorm1d(n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.PReLU(),
            nn.ConvTranspose1d(in_channels=n_channels_base, out_channels=1, kernel_size=3, stride=1,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.Sigmoid(),
            # nn.PReLU(),
        )

    def forward(self, x):
        x = self.encoder(x.view(-1, 1, x.shape[1]))
        # x=x.view(-1, 1, x.shape[1])
        # for i in range(len(self.encoder)):
        #     x = self.encoder[i](x)
        #     print(f'enc{i}', x.shape)
        # for i in range(len(self.decoder)):
        #     x = self.decoder[i](x)
        #     print(f'dec{i}', x.shape)
        x = self.decoder(x)
        return torch.squeeze(x, dim=1)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #ConvTranspose1d(1, 1, kernel_size=3, stride=3, padding=1, output_padding=1)
        ngf = 8
        self.main = nn.Sequential(
            nn.ConvTranspose1d(opt.latent_dim, ngf * 8, 8, 1, 0),
            nn.BatchNorm1d(ngf * 8, eps=0.0001, momentum=0.01),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.PReLU(),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 7, 3, 0),
            nn.BatchNorm1d(ngf * 4, eps=0.0001, momentum=0.01),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ngf * 4, ngf*2, 7, 2, 0),
            nn.BatchNorm1d(ngf*2, eps=0.0001, momentum=0.01),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ngf*2, ngf, 6, 2, 0),
            nn.BatchNorm1d(ngf, eps=0.0001, momentum=0.01),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ngf, 1, 3, 1, 0),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(-1,x.shape[1] ,1)

        out = self.main(x)
        return torch.squeeze(out, dim=1)
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        n_channels_base = 8

        self.main = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_channels_base, kernel_size=8, stride=2, padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            # nn.LeakyReLU(1 / 5.5, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base, out_channels=2 * n_channels_base, kernel_size=7, stride=2,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.PReLU(),
            nn.Conv1d(in_channels=n_channels_base * 2, out_channels=4 * n_channels_base, kernel_size=7, stride=2,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base * 4, out_channels=8 * n_channels_base, kernel_size=7, stride=2,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base * 8, out_channels=n_channels_base * 16, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.Tanh(),
        )
    def forward(self, x):
        x = self.main(x.view(-1, 1, x.shape[1]))
        return torch.squeeze(x, dim=2)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 8
        # batchnorm位置需要注意
        self.conv1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv1d(1, ndf*2, 30, 5, 0),
            nn.BatchNorm1d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 2, ndf * 4, 17, 5, 0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 2*ndf, 25, 5, 0),
            nn.BatchNorm1d(2*ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf*2, ndf * 4, 18, 4, 0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv1d(1, ndf * 4, 15, 4, 0),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 4, 1, 10, 4, 0),
            nn.Sigmoid()
        )

    def forward(self, x,z):
        outx = self.conv1(x.view(-1, 1, x.shape[1]))
        outx=torch.squeeze(outx, dim=2)
        outz = self.conv2(z.view(-1, 1, z.shape[1]))
        outz=torch.squeeze(outz, dim=2)
        out=torch.cat([outx, outz], dim=1)
        out = self.conv4(out.view(-1, 1, out.shape[1]))
        return torch.squeeze(out,dim=2)

###############
### Lossess ###
###############

criterion = nn.BCELoss()

def mse_loss(x_output, y_target):
    loss=nn.MSELoss(reduction='sum')
    l=loss(x_output,y_target)/opt.batch_size
    return l
def distance(X, Y, sqrt):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX, -1).cuda()
    X2 = (X * X).sum(1).reshape(nX, 1)
    Y = Y.view(nY, -1).cuda()
    Y2 = (Y * Y).sum(1).reshape(nY, 1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) - 2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M

def mmd(Mxx, Mxy, Myy, sigma) :
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx/(scale*2*sigma*sigma))
    Mxy = torch.exp(-Mxy/(scale*2*sigma*sigma))
    Myy = torch.exp(-Myy/(scale*2*sigma*sigma))
    a = Mxx.mean()+Myy.mean()-2*Mxy.mean()
    if a>0:
        mmd=a.sqrt()
    else:
        mmd=0
    # mmd = math.sqrt(max(a, 0))
    return mmd

def mmd_test(Mxx, Mxy, Myy, sigma) :
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx/(scale*2*sigma*sigma))
    Mxy = torch.exp(-Mxy/(scale*2*sigma*sigma))
    Myy = torch.exp(-Myy/(scale*2*sigma*sigma))
    a = Mxx.mean()+Myy.mean()-2*Mxy.mean()
    mmd = math.sqrt(max(a, 0))
    return mmd

#################
### Functions ###
#################

def sample_transform(sample):
    """
    Transform samples to their nearest integer
    :param sample: Rounded vector.
    :return:
    """
    sample[sample >= 0.5] = 1
    sample[sample < 0.5] = 0
    return sample


def weights_init(m):
    """
    Custom weight initialization.
    NOTE: Bad initialization may lead to dead model and can prevent training!
    :param m: Input argument to extract layer type
    :return: Initialized architecture
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


#############
### Model ###
#############

# Initialize generator and discriminator
discriminatorModel = Discriminator()
generatorModel = Generator()
autoencoderModel = Autoencoder()
autoencoderDecoder = autoencoderModel.decoder
EncoderModel=Encoder()

# Define cuda Tensors
# BE careful about torch.FloatTensor([1])!!!!
# I once defined it as torch.FloatTensor(1) without brackets around 1 and everything was messed hiddenly!!
Tensor = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1

if torch.cuda.device_count() > 1 and opt.multiplegpu:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    generatorModel = nn.DataParallel(generatorModel, list(range(opt.num_gpu)))
    discriminatorModel = nn.DataParallel(discriminatorModel, list(range(opt.num_gpu)))
    autoencoderModel = nn.DataParallel(autoencoderModel, list(range(opt.num_gpu)))
    autoencoderDecoder = nn.DataParallel(autoencoderDecoder, list(range(opt.num_gpu)))
    EncoderModel = nn.DataParallel(EncoderModel, list(range(opt.num_gpu)))

if opt.cuda:
    """
    model.cuda() will change the model inplace while input.cuda() 
    will not change input inplace and you need to do input = input.cuda()
    ref: https://discuss.pytorch.org/t/when-the-parameters-are-set-on-cuda-the-backpropagation-doesnt-work/35318
    """
    generatorModel.cuda()
    discriminatorModel.cuda()
    autoencoderModel.cuda()
    autoencoderDecoder.cuda()
    EncoderModel.cuda()
    one, mone = one.cuda(), mone.cuda()
    Tensor = torch.cuda.FloatTensor

# Weight initialization
generatorModel.apply(weights_init)
discriminatorModel.apply(weights_init)
autoencoderModel.apply(weights_init)
EncoderModel.apply(weights_init)
# Optimizers
g_params = [{'params': generatorModel.parameters()},
            {'params': autoencoderDecoder.parameters(), 'lr': 1e-4}]

optimizer_G = torch.optim.Adam(g_params, lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

optimizer_D = torch.optim.Adam(
        params=discriminatorModel.parameters(),
        lr=opt.lr,
        betas=(opt.b1, opt.b2),
        weight_decay=opt.weight_decay,
    )

optimizer_E = torch.optim.Adam(
        params=EncoderModel.parameters(),
        lr=0.001,
        betas=(opt.b1, opt.b2),
        weight_decay=opt.weight_decay,
    )
optimizer_A = torch.optim.Adam(
        params=autoencoderModel.parameters(),
        lr=opt.lr,
        betas=(opt.b1, opt.b2),
        weight_decay=opt.weight_decay,
    )

################
### TRAINING ###
################
if opt.training:

    if opt.resume:
        #####################################
        #### Load model and optimizer #######
        #####################################

        # Loading the checkpoint
        checkpoint = torch.load(os.path.join(opt.modelPATH, "model_epoch_50.pth"))

        # Load models
        generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
        discriminatorModel.load_state_dict(checkpoint['Discriminator_state_dict'])
        autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])
        autoencoderDecoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])

        # Load optimizers
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])

        # Load epoch number
        epoch_resume = checkpoint['epoch']

        generatorModel.eval()
        discriminatorModel.eval()
        autoencoderModel.eval()
        autoencoderDecoder.eval()

    else:
        epoch_resume = 0
        print('Training from scratch...')

    if not opt.pretrained_status:
        print('开始预训练：')
        for epoch_pre in range(opt.n_epochs_pretrain):
            for i_batch, samples in enumerate(dataloader_train):

                # Configure input
                real_samples = Variable(samples.type(Tensor))

                # # Reset gradients (if you comment below line, it would be a mess. Think why?!!!!!!!!!)
                optimizer_A.zero_grad()

                # Extract microbatch
                micro_batch = real_samples


                # Generate a batch of images
                recons_samples = autoencoderModel(micro_batch)
                # Loss measures generator's ability to fool the discriminator
                a_loss = mse_loss(recons_samples, micro_batch)
                # Backward
                a_loss.backward()
                optimizer_A.step()

                batches_done = epoch_pre * len(dataloader_train) + i_batch + 1
                if batches_done % opt.sample_interval == 0:
                    print(
                        "[Epoch %d/%d of pretraining] [Batch %d/%d] [A loss: %.3f]"
                        % (epoch_pre + 1, opt.n_epochs_pretrain, i_batch + 1, len(dataloader_train), a_loss.item())
                        , flush=True)

    else:
        print('loading pretrained autoencoder...')

        # Loading the checkpoint
        checkpoint = torch.load(os.path.join(opt.expPATH, "aepretrained.pth"))

        # Load models
        autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])

        # Load optimizers
        optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])

        # Load weights
        autoencoderModel.eval()

    gen_iterations = 0
    for epoch in range(epoch_resume, opt.n_epochs):
        epoch_start = time.time()
        for i_batch, samples in enumerate(dataloader_train):
            real_samples = Variable(samples.type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(samples.shape[0]).fill_(1.0), requires_grad=False)
            fake_truths = Variable(Tensor(samples.shape[0]).fill_(0.0), requires_grad=False)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # reset gradients of discriminator
            optimizer_D.zero_grad()

            # Microbatch processing
            # for i in range(opt.batch_size):
            # Extract microbatch
            micro_batch = real_samples

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = True

                # Error on real samples
                # print('micro_batch',micro_batch)
            z_real = EncoderModel(micro_batch)
            out_real = discriminatorModel(micro_batch, z_real.detach()).view(-1)
            real_loss = criterion(out_real, valid)

            # Sample noise as generator input torch.Size([30, 128])
            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)
            # z = torch.randn(micro_batch.shape[0], opt.latent_dim, device=device)
            # Generate a batch of images
            fake = generatorModel(z)

            fake_decoded = torch.squeeze(autoencoderDecoder(fake.unsqueeze(dim=2)), dim=1)

            out_fake = discriminatorModel(fake_decoded.detach(), z).view(-1)
            fake_loss = criterion(out_fake, fake_truths)

            # total loss and calculate the backprop
            d_loss = (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = False

            # Zero grads
            optimizer_G.zero_grad()

            # Sample noise as generator input

            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)
            # Generate a batch of images
            fake = generatorModel(z)
            # uncomment if there is no autoencoder
            fake_decoded = torch.squeeze(autoencoderDecoder(fake.unsqueeze(dim=2)), dim=1)
            # Loss measures generator's ability to fool the discriminator
            g_loss = criterion(discriminatorModel(fake_decoded, z).view(-1), valid)

            # read more at https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/4

            optimizer_E.zero_grad()
            z_real = EncoderModel(micro_batch)
            e_loss = criterion(discriminatorModel(micro_batch, z_real).view(-1), fake_truths)

            # 重构误差
            fake1 = generatorModel(z_real)
            fake_decoded1 = torch.squeeze(autoencoderDecoder(fake1.unsqueeze(dim=2)), dim=1)

            rec_loss = (mse_loss(fake_decoded, micro_batch))*0.1
            #重构mmd误差
            Mxx = distance(micro_batch, micro_batch, False)
            Mxy = distance(micro_batch, fake_decoded, False)
            Myy = distance(fake_decoded, fake_decoded, False)
            sigma=1

            Mxx1 = distance(micro_batch, micro_batch, False)
            Mxy1 = distance(micro_batch, fake_decoded1, False)
            Myy1 = distance(fake_decoded1, fake_decoded1, False)
            mmd_loss=mmd(Mxx, Mxy, Myy, sigma)+mmd(Mxx1, Mxy1, Myy1, sigma)

            loss = 14*mmd_loss +e_loss + g_loss-rec_loss
            loss.backward()

            optimizer_E.step()
            optimizer_G.step()
            gen_iterations += 1
            batches_done = epoch * len(dataloader_train) + i_batch + 1
            if batches_done % opt.sample_interval == 0:
                print(
                    'TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss_D: %.6f Loss_G: %.6f Loss_E: %.6f Loss_MMD: %.6f Loss_REC: %.6f'
                    % (epoch + 1, opt.n_epochs, i_batch + 1, len(dataloader_train),d_loss.item(), g_loss.item(),e_loss.item(),mmd_loss.item(),rec_loss.item()),flush=True)
        # End of epoch
        epoch_end = time.time()
        if opt.epoch_time_show:
            print("It has been {0} seconds for this epoch".format(epoch_end - epoch_start), flush=True)

        if (epoch + 1) % opt.epoch_save_model_freq == 0:
            # Refer to https://pytorch.org/tutorials/beginner/saving_loading_models.html
            torch.save({
                'epoch': epoch + 1,
                'Generator_state_dict': generatorModel.state_dict(),
                'Discriminator_state_dict': discriminatorModel.state_dict(),
                'Autoencoder_state_dict': autoencoderModel.state_dict(),
                'Autoencoder_Decoder_state_dict': autoencoderDecoder.state_dict(),
                'Encoder_state_dict': EncoderModel.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'optimizer_A_state_dict': optimizer_A.state_dict(),
                'optimizer_E_state_dict': optimizer_E.state_dict(),
            }, os.path.join(opt.expPATH, "model_epoch_%d.pth" % (epoch + 1)))

            # keep only the most recent 10 saved models
            # ls -d -1tr /home/sina/experiments/pytorch/model/* | head -n -10 | xargs -d '\n' rm -f
            # call("ls -d -1tr " + opt.expPATH + "/*" + " | head -n -10 | xargs -d '\n' rm -f", shell=True)

trainData = pd.read_csv(os.path.join(opt.expPATH, "dataTrain.csv"))
testData = pd.read_csv(os.path.join(opt.expPATH, "dataTest.csv"))

feature=trainData.columns
feature=feature.copy().drop('attack')

# pd.set_option('display.max_columns', None)
trainData = trainData.to_numpy()
testData = testData.to_numpy()
real_train = trainData.astype(np.float32)
real_test = testData.astype(np.float32)


# if opt.generate:
for j in range(50,51):
    # Check cuda
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device BUT it is not in use...")

    # Activate CUDA
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    #####################################
    #### Load model and optimizer #######
    #####################################

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.expPATH, f'model_epoch_{j}.pth'))


    # # Load models
    generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
    autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])
    autoencoderDecoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])
    EncoderModel.load_state_dict(checkpoint['Encoder_state_dict'])
    # insert weights [required]
    generatorModel.eval()
    autoencoderModel.eval()
    autoencoderDecoder.eval()
    EncoderModel.eval()
    print(f'合成数据：{j}')
    # #######################################################
    # #### Load real data and generate synthetic data #######
    # #######################################################
    #
    # Load real data
    real_samples = dataset_train_object.return_data()
    # num_fake_samples = len(dataset_train_object)
    num_fake_samples = 70000
    # Generate a batch of samples
    # gen_samples = np.zeros_like(real_samples, dtype=type(real_samples))
    gen_samples=np.zeros([70000, 134])
    n_batches = int(num_fake_samples / opt.batch_size)

    for i in range(n_batches):
        # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
        z = torch.randn(opt.batch_size, opt.latent_dim, device=device)
        # z = torch.random(1, opt.latent_dim, device=device)
        gen_samples_tensor = generatorModel(z)
        gen_samples_decoded = torch.squeeze(
            autoencoderDecoder(gen_samples_tensor.view(-1, gen_samples_tensor.shape[1], 1)))
        gen_samples[i * opt.batch_size:(i + 1) * opt.batch_size, :] = gen_samples_decoded.cpu().data.numpy()

        # Check to see if there is any nan
        assert (gen_samples[i, :] != gen_samples[i, :]).any() == False

    gen_samples = np.delete(gen_samples, np.s_[(i + 1) * opt.batch_size:], 0)

    for i in range(66):
        gen_samples[gen_samples[:, i] >= 0.5, i] = 1
        gen_samples[gen_samples[:, i] < 0.5, i] = 0
    gen_samples[gen_samples[:, -1] >= 0.5, -1] = 1
    gen_samples[gen_samples[:, -1] < 0.5, -1] = 0
    # 12开始到40
    # for i in range(12,39,2):
    #     gen_samples = gen_samples[((gen_samples[:, i] == 0) & (gen_samples[:, i+1] == 1)) | ((gen_samples[:, i] == 1) & (gen_samples[:, i+1] == 0)), :]
    # for i in range(0,10,3):
    #     gen_samples = gen_samples[((gen_samples[:, i] == 1) & (gen_samples[:, i+1] == 0)&(gen_samples[:, i+2] == 0)) | (
    #             (gen_samples[:, i] == 0) & (gen_samples[:, i+1] == 1)&(gen_samples[:, i+2] == 0))|((gen_samples[:, i] == 0) &(gen_samples[:, i+1] == 0)&(gen_samples[:,i+2] == 1)),:]
    # # 41开始-64
    # for i in range(40,62,3):
    #     gen_samples = gen_samples[((gen_samples[:, i] == 1) & (gen_samples[:, i+1] == 0)&(gen_samples[:, i+2] == 0)) | (
    #             (gen_samples[:, i] == 0) & (gen_samples[:, i+1] == 1)&(gen_samples[:, i+2] == 0))|((gen_samples[:, i] == 0) &(gen_samples[:, i+1] == 0)&(gen_samples[:,i+2] == 1)),:]
    # gen_samples = gen_samples[((gen_samples[:, 64] == 0) & (gen_samples[:, 65] == 1)) | ((gen_samples[:, 64] == 1) & (gen_samples[:, 65] == 0)), :]

    if (gen_samples.shape[0] < trainData.shape[0]):
        print("合成数据少于", trainData.shape[0])
        continue
    x = trainData.shape[0]
    gen_samples = gen_samples[0:x]
    print(gen_samples.shape[0])
    # Trasnform Object array to float
    gen_samples = gen_samples.astype(np.float32)
    np.save(os.path.join(opt.expPATH, f'syntheticlabel{j}.npy'), gen_samples, allow_pickle=False)

    train_f = np.load(f'wadi/syntheticlabel{j}.npy', allow_pickle=False)

    X_train_f, y_train_f = train_f[:, :-1], train_f[:, -1]
    if((y_train_f==np.zeros_like(y_train_f, dtype=type(y_train_f))).all() or (y_train_f==np.ones_like(y_train_f, dtype=type(y_train_f))).all()):
        print("标签只有一类")
        continue
    # X_train_f, y_train_f, X_test_f, y_test_f = train_f[:,:-1], train_f[:,-1], test_f[:,:-1], test_f[:,-1]
    X_train_r, y_train_r, X_test_r, y_test_r = real_train[:, :-1], real_train[:, -1], real_test[:, :-1], real_test[:,-1]

    sc = MinMaxScaler()

    X_train_r = sc.fit_transform(X_train_r)
    X_test_r = sc.transform(X_test_r)

    ###############################
    ######## Classifier ###########
    ###############################

    ########## Train: Real Test: Real ########

    # Supervised transformation based on random forests
    # Good to know about feature transformation
    n_estimator = 10
    # cls = RandomForestClassifier(max_depth=5, n_estimators=n_estimator)
    cls = GradientBoostingClassifier(n_estimators=n_estimator)
    cls.fit(X_train_r, y_train_r)
    relevants=cls.feature_importances_
    indices = np.argsort(relevants)[::-1]
    feature1=feature.copy()
    plt.barh(feature1[indices[0:20]],relevants[indices[0:20]])
    plt.savefig(f'pictures/real{j}.png',pad_inches = 1,bbox_inches ="tight")
    plt.close()
    y_pred_rf = cls.predict_proba(X_test_r)[:, 1]
    # ROC
    fpr_rf_lm, tpr_rf_lm, _ = metrics.roc_curve(y_test_r, y_pred_rf)
    print('AUROC: ', metrics.auc(fpr_rf_lm, tpr_rf_lm))
    # PR
    precision, recall, thresholds = metrics.precision_recall_curve(y_test_r, y_pred_rf)
    AUPRC = metrics.auc(recall, precision)
    # print('AP: ', metrics.average_precision_score(y_test_r, y_pred_rf))
    print('AUPRC: ', AUPRC)

    ########## linear_regression ############
    print("linear_regression:")
    a = X_train_r[:, :-2]
    b = X_train_r[:, -1:]
    x_train_lr = np.concatenate((a, b), axis=1)
    y_train_lr = X_train_r[:, -2]

    a = X_test_r[:, :-2]
    b = X_test_r[:, -1:]
    x_test_lr = np.concatenate((a, b), axis=1)
    y_test_lr = X_test_r[:, -2]
    linear_regressor = LinearRegression()
    linear_regressor.fit(x_train_lr, y_train_lr)
    y_pre = linear_regressor.predict(x_test_lr)
    loss = nn.MSELoss(reduction='sum')
    l = loss(torch.from_numpy(y_pre), torch.from_numpy(y_test_lr))
    print("mse:", l.item())
    ########## test Synthetic ############

    print('Train: Synthetic Test: Real')
    ########## Train: Synthetic Test: Real ########

    # Supervised transformation based on random forests
    # Good to know about feature transformation

    n_estimator = 10
    # X_train_f = sc.fit_transform(X_train_f)
    # cls = RandomForestClassifier(max_depth=5, n_estimators=n_estimator)
    cls = GradientBoostingClassifier(n_estimators=n_estimator)
    cls.fit(X_train_f, y_train_f)
    relevants = cls.feature_importances_
    indices = np.argsort(relevants)[::-1]
    feature2 = feature.copy()
    plt.barh(feature2[indices[0:20]], relevants[indices[0:20]])
    plt.savefig(f'pictures/syn{j}.png',pad_inches = 1,bbox_inches ="tight")
    plt.close()
    y_pred_rf = cls.predict_proba(X_test_r)[:, 1]

    # ROC
    fpr_rf_lm, tpr_rf_lm, _ = metrics.roc_curve(y_test_r, y_pred_rf)
    print('AUROC: ', metrics.auc(fpr_rf_lm, tpr_rf_lm))

    # PR
    precision, recall, thresholds = metrics.precision_recall_curve(y_test_r, y_pred_rf)
    AUPRC = metrics.auc(recall, precision)
    # print('AP: ', metrics.average_precision_score(y_test_r, y_pred_rf))
    print('AUPRC: ', AUPRC)

    ########## linear_regression ############
    print("linear_regression:")
    a = X_train_f[:, :-2]
    b = X_train_f[:, -1:]
    x_train_lr = np.concatenate((a, b), axis=1)
    y_train_lr = X_train_f[:, -2]
    linear_regressor = LinearRegression()
    linear_regressor.fit(x_train_lr, y_train_lr)
    y_pre = linear_regressor.predict(x_test_lr)

    loss = nn.MSELoss(reduction='sum')
    l = loss(torch.from_numpy(y_pre), torch.from_numpy(y_test_lr))
    print("mse:", l.item())

    ########## mmd ############
    sc = MinMaxScaler()
    trainData_mmd = sc.fit_transform(real_train)
    real_samples_mmd = torch.from_numpy(trainData_mmd).float().to(device)
    gen_samples_mmd = torch.from_numpy(train_f).float().to(device)

    real = real_samples_mmd[0:10000]
    fake = gen_samples_mmd[0:10000]
    Mxx = distance(real, real, False)
    Mxy = distance(real, fake, False)
    Myy = distance(fake, fake, False)
    sigma=1
    print('Manual MMD: ', mmd_test(Mxx, Mxy, Myy, sigma))

    ########## privacy 方差 ############
    loss = nn.MSELoss(reduction='sum')
    l = loss(gen_samples_mmd, real_samples_mmd)
    print('方差：',l)

