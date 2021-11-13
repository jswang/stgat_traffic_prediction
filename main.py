#!/usr/bin/python3

# @Time     : 11/7/21
# @Author   : Julie Wang
# @FileName : data_utils.py

import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.math_graph import *
from data_loader.traffic_dataset import TrafficDataset, get_splits
from models.trainer import model_train
from models.tester import model_test

import torch
import argparse

parser = argparse.ArgumentParser()
# F parameter
parser.add_argument('--n_his', type=int, default=12)
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

# Config that's passed around
config = {
    'C_BATCH_SIZE': 50,
    'C_EPOCHS': 150,
    'C_WEIGHT_DECAY': 5e-4,
    'C_INITIAL_LR': 2e-4
}

# Load weighted adjacency matrix W
if args.graph == 'default':
    W = weight_matrix(os.path.join('./dataset', f'PeMSD7_W_228.csv'))
else:
    # load customized graph weight matrix
    W = weight_matrix(os.path.join('./dataset', args.graph))

# Data Preprocessing and loading
# TODO make filename configurable
data = pd.read_csv('./dataset/PeMSD7_V_228.csv', header=None).values
dataset = TrafficDataset(data, args.n_his, args.n_pred)
(train, val, test) = get_splits(dataset)
train_dataloader = torch.utils.data.DataLoader(train, batch_size=config['C_BATCH_SIZE'], shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=config['C_BATCH_SIZE'], shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=config['C_BATCH_SIZE'], shuffle=True)

model_train(train_dataloader, config)
model_test(test_dataloader, config)
