#!/usr/bin/python3

# @Time     : 11/7/21
# @Author   : Julie Wang
# @FileName : main.py

import torch
import pandas as pd

from models.trainer import load_from_checkpoint, model_train, model_test
from torch_geometric.loader import DataLoader
from data_loader.dataloader import TrafficDataset, get_splits, distance_to_weight

def main():
    """
    Main function to train and test a model.
    """

    # Constant config to use througout
    config = {
        'BATCH_SIZE': 50,
        'EPOCHS': 200,
        'WEIGHT_DECAY': 5e-5,
        'INITIAL_LR': 3e-4,
        'CHECKPOINT_DIR': './runs',
        'N_PRED': 9,
        'N_HIST': 12,
        'DROPOUT': 0.2,
        # number of possible 5 minute measurements per day
        'N_DAY_SLOT': 288,
        # number of days worth of data in the dataset
        'N_DAYS': 44,
        # If false, use GCN paper weight matrix, if true, use GAT paper weight matrix
        'USE_GAT_WEIGHTS': True,
        'N_NODE': 228,
    }
    # Number of possible windows in a day
    config['N_SLOT']= config['N_DAY_SLOT'] - (config['N_PRED']+config['N_HIST']) + 1

    # Load the weight matrix
    distances = pd.read_csv('./dataset/PeMSD7_W_228.csv', header=None).values
    W = distance_to_weight(distances, gat_version=config['USE_GAT_WEIGHTS'])
    # Load the dataset
    dataset = TrafficDataset(config, W)

    # total of 44 days in the dataset, use 34 for training, 5 for val, 5 for test
    train, val, test = get_splits(dataset, config['N_SLOT'], (34, 5, 5))
    train_dataloader = DataLoader(train, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_dataloader = DataLoader(val, batch_size=config['BATCH_SIZE'], shuffle=True)
    test_dataloader = DataLoader(test, batch_size=config['BATCH_SIZE'], shuffle=False)

    # Get gpu if you can
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    # Configure and train model
    config['N_NODE'] = dataset.n_node
    model = model_train(train_dataloader, val_dataloader, config, device)
    # Or, load from a saved checkpoint
    # model = load_from_checkpoint('./runs/model_final_60epochs.pt', config)
    # Test Model
    model_test(model, test_dataloader, device, config)


if __name__ == "__main__":
    main()
