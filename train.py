### python lib
import os, sys, argparse, glob, re, math, copy, pickle, random, logging
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

### torch lib
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

### ode solver
from torchdiffeq import odeint

from model import LatentODEfunc, RecognitionRNN, Decoder
from dataset import indDataset_train,indDataset_test

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Trajectory Forecasting")

    ### model options
    parser.add_argument('-latent_dim',      type=int,     default=128,               help='dimention of latent vector')
    

    ### dataset options
    parser.add_argument('-train_data_dir',  type=str,     default='../data/inD/data/', help='path to training dataset')
    parser.add_argument('-model_dir',       type=str,     default='checkpoints',    help='path to model folder')


    ### training options
    parser.add_argument('-solver',          type=str,     default="ADAM",           choices=["SGD", "ADAM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=0,                help='weight decay')
    parser.add_argument('-batch_size',      type=int,     default=32,                help='training batch size')
    parser.add_argument('-train_epoch_size',type=int,     default=1000,             help='train epoch size')
    parser.add_argument('-valid_epoch_size',type=int,     default=100,              help='valid epoch size')
    parser.add_argument('-epoch_max',       type=int,     default=2000,              help='max #epochs')
    parser.add_argument('-loss',            type=str,     default="L2",             help="optimizer [Options: SGD, ADAM]")

    ### learning rate options
    parser.add_argument('-lr_init',         type=float,   default=1e-4,             help='initial learning Rate')
    parser.add_argument('-lr_offset',       type=int,     default=20,               help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('-lr_step',         type=int,     default=20,               help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=1,              help='learning rate drop ratio')
    parser.add_argument('-lr_min_m',        type=float,   default=0.1,              help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')

    ### distributed training
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument('--gpu', type=str, default='0')


    opts = parser.parse_args()


    torch.cuda.set_device(opts.local_rank)
    torch.distributed.init_process_group(backend='nccl')

    ### adjust options
    opts.lr_min = opts.lr_init * opts.lr_min_m

    ### model saving directory
    print("========================================================")
    print("===> Save model to %s" %opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)

    print('===Initializing model===')

    device = torch.device(f'cuda:{opts.local_rank}')




    func = LatentODEfunc(opts.latent_dim, 128).to(device)
    rec = RecognitionRNN(opts.latent_dim, 4, 128, opts.batch_size).to(device)
    dec = Decoder(opts.latent_dim, 4, 128).to(device)
    loss_meter = RunningAverageMeter()

    func = torch.nn.parallel.DistributedDataParallel(func)
    rec = torch.nn.parallel.DistributedDataParallel(rec) 
    dec = torch.nn.parallel.DistributedDataParallel(dec)

    mse_loss = nn.MSELoss()


    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
        
    ### initialize optimizer
    if opts.solver == 'SGD':
        optimizer = optim.SGD(params, lr=opts.lr_init, momentum=opts.momentum, weight_decay=opts.weight_decay)
    elif opts.solver == 'ADAM':
        optimizer = optim.Adam(params, lr=1e-4, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))
        optimizer_D = optim.SGD(params, lr=1e-4, momentum=0.9, weight_decay=opts.weight_decay)
    else:
        raise Exception("Not supported solver (%s)" % opts.solver)

    func.train()
    rec.train()
    dec.train()


    samp_ts = torch.from_numpy(np.linspace(0, 8, num=20)).float().to(device)
    for epoch in range(1, opts.epoch_max + 1):
        ### create dataset
        dataset = indDataset_train(opts.train_data_dir)

        data_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4, drop_last=True)

        optimizer.zero_grad()

        for iteration, (tracks, x_min, x_max, y_min, y_max) in enumerate(data_loader, 0):
            x_min, x_max, y_min, y_max = x_min[0], x_max[0], y_min[0], y_max[0]
            tracks = tracks.to(device)
            tracks[:,:,0] = (tracks[:,:,0] - x_min) / (x_max - x_min)
            tracks[:,:,1] = (tracks[:,:,1] - y_min) / (y_max - y_min)
            tracks[:,:,2] = (tracks[:,:,2] - x_min) / (x_max - x_min)
            tracks[:,:,3] = (tracks[:,:,3] - y_min) / (y_max - y_min)

            h = rec.module.initHidden().to(device)
            for t in reversed(range(int(tracks.size(1)/2))):
                obs = tracks[:, t, :]
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :opts.latent_dim], out[:, opts.latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
            pred_x = dec(pred_z)

            # compute loss
            noise_std_ = torch.zeros(pred_x.size()).to(device) + 0.1
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = log_normal_pdf(
                tracks, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                    pz0_mean, pz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())


            print('Iter: {}, running avg elbo: {:.4f}'.format(epoch, -loss_meter.avg))
            
        if epoch % 100 == 0:
            dict = {'func': func.state_dict(),'rec': rec.state_dict(),'dec': dec.state_dict}
            torch.save(dict, os.path.join(opts.model_dir, 'model_%d.pt' % epoch))
    
