#!/usr/bin/python3

import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
# F parameter
parser.add_argument('--n_his', type=int, default=12)
# T parameter. TODO not sure what this should start out as? what is this?
parser.add_argument('--n_temporal', type=int, default=8)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')

args = parser.parse_args()
print(f'Training configs: {args}')


# Load weighted adjacency matrix W
if args.graph == 'default':
    W = weight_matrix(pjoin('./dataset', f'PeMSD7_W_228.csv'))
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('./dataset', args.graph))

# Data Preprocessing
data_file = f'PeMSD7_V_228.csv'
# Splits to apply to data. TODO make configurable?
splits = np.array([.6, .2, .2])
PeMS = datagen(pjoin('./dataset', data_file), splits, (args.n_temporal, args.n_his), args.n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    print("hello")
    # model_train(PeMS, blocks, args)
    # model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)
