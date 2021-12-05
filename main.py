#!/usr/bin/python3

# @Time     : 11/7/21
# @Author   : Julie Wang
# @FileName : main.py

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from data_loader.dataloader import TrafficDataset, get_splits
from models.trainer import model_train, model_test
from torch_geometric.loader import DataLoader

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

    # Get the train/val/test datasets
    dataset = TrafficDataset(args.n_hist, args.n_pred)
    train, val, test = get_splits(dataset, (0.6, 0.2, 0.2))
    train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=args.batch_size, shuffle=True)

    # Configure and train model
    config['n_pred'] = args.n_pred
    config['n_hist'] = args.n_hist
    config['n_node'] = dataset.n_node
    model = model_train(train_dataloader, val_dataloader, config)

    # Test Model
    model_test(model, test_dataloader)

if __name__ == "__main__":
    main()
