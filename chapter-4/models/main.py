from __future__ import print_function, absolute_import, division


import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd

print('library loaded!')

from opt import Options

import model as nnmodel

import GCN_model as modelnn

print('libraries imported from others local folder!')

def main(opt):
    print('Hello Gaurav!')
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()
    


    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n
    sample_rate = opt.sample_rate

    upJ = np.array([10,11,12,13,14,15,16,17,18,19,20,21])
    downJ = np.array([0,1,2,3,4,5,6,7,8,9])
    dim_up = np.concatenate((upJ * 3, upJ * 3 + 1, upJ * 3 + 2))
    dim_down = np.concatenate((downJ * 3, downJ * 3 + 1, downJ * 3 + 2))
    n_up = dim_up.shape[0]
    n_down = dim_down.shape[0]
    part_sep = (dim_up, dim_down, n_up, n_down)

    model = nnmodel.GCN(in_d=dct_n, hid_d=opt.linear_size, p_dropout=opt.dropout,
                        num_stage=opt.num_stage, node_n=66, J=opt.J, part_sep=part_sep,
                        W_pg=opt.W_pg, W_p=opt.W_p)

    if is_cuda:
        model.cuda()

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('---------First model !SPGSN!completed----------------------------------')
    print('Now the second model !OOD model!------------------------------------------------')

    print(">>> creating model")
    model = modelnn.GCN(input_feature=dct_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout,
                        num_stage=opt.num_stage, node_n=48, variational=opt.variational, n_z=opt.n_z, num_decoder_stage=opt.num_decoder_stage)
    #methods = MODEL_METHODS(model, is_cuda)
    #if opt.is_load:
    #  start_epoch, err_best, lr_now = methods.load_weights(opt.load_path)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    
if __name__ == "__main__":
    option = Options().parse()
    main(option)