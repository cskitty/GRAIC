'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-05-30 14:20:04
LastEditTime: 2020-09-14 23:09:19
@Description: 
'''

import numpy as np
import torch.nn.init as init
import torch
import torch.nn as nn
from torch.autograd import Variable


def CPU(var):
    return var.detach().cpu().numpy()


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def normal(x, mu, sigma_sq):
    pi = CUDA(Variable(torch.FloatTensor([np.pi])))
    a = (-1*(CUDA(Variable(x))-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b
