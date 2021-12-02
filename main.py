#!/usr/bin/python3

# @Time     : 11/7/21
# @Author   : Julie Wang
# @FileName : main.py

import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.math_graph import *
from data_loader.dataloader import TrafficDataset, get_splits
from models.trainer import model_train
from models.tester import model_test
from torch_geometric.loader import DataLoader

import torch
import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    # F parameter
    parser.add_argument('--n_hist', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--opt', type=str, default='RMSProp')
    parser.add_argument('--graph', type=str, default='PeMSD7_W_228', help='Graph name defaults to PeMSD7_W_228')
    parser.add_argument('--graph_data', type=str, default='PeMSD7_V_228', help='Graph dataset name defaults to PeMSD7_V_228')
    parser.add_argument('--inf_mode', type=str, default='merge')

    return parser.parse_args()

# Constant config to use througout
config = {
    'C_BATCH_SIZE': 50,
    'C_EPOCHS': 150,
    'C_WEIGHT_DECAY': 5e-4,
    'C_INITIAL_LR': 2e-4
}

def main():
    """
    Main function to train and test a model.
    """
    args = parse_args()

    # Load weighted adjacency matrix W
    W_pt_path = os.path.join('./dataset', args.graph + '.pt')
    if os.path.exists(W_pt_path):
        W = torch.load(W_pt_path)
    else:
        W = weight_matrix(os.path.join('./dataset', args.graph + '.csv'))
        torch.save(W, W_pt_path)

    # Data Preprocessing and loading
    data_pt_path = os.path.join('./dataset', args.graph_data + '.pt')
    if os.path.exists(data_pt_path):
        data = torch.load(data_pt_path)
    else:
        data = pd.read_csv(os.path.join('./dataset', args.graph_data + '.csv'), header=None).values
        torch.save(data, data_pt_path)

    # right now just one big dataset
    dataset = TrafficDataset(data, W, args.n_hist, args.n_pred)
    # Transfrom the dataset into train, validation, and test
    (train, val, test) = get_splits(dataset, (0.6, 0.2, 0.2))
    train_dataloader = DataLoader(train, batch_size=50, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=50, shuffle=True)
    # Train model
    config['n_pred'] = args.n_pred
    config['n_hist'] = args.n_hist
    config['n_node'] = 228 #TODO parameterize
    model_train(train_dataloader, val_dataloader, config)
    # Test model
    model_test(test_dataloader, config)

if __name__ == "__main__":
    main()
