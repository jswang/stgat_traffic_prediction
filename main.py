#!/usr/bin/python3

# @Time     : 11/7/21
# @Author   : Julie Wang
# @FileName : main.py

import argparse
import torch

from data_loader.dataloader import TrafficDataset, get_splits
from models.trainer import load_from_checkpoint, model_train, model_test
from torch_geometric.loader import DataLoader


def main():
    """
    Main function to train and test a model.
    """

    # Constant config to use througout
    config = {
        'BATCH_SIZE': 50,
        'EPOCHS': 150,
        'WEIGHT_DECAY': 5e-4,
        'INITIAL_LR': 2e-4,
        'CHECKPOINT_DIR': './runs',
        'N_PRED': 9,
        'N_HIST': 12,
        # number of possible 5 minute measurements per day
        'N_DAY_SLOT': 288,
        # number of days worth of data in the dataset
        'N_DAYS': 44,
    }
    # Number of possible windows in a day
    config['N_SLOT']= config['N_DAY_SLOT'] - (config['N_PRED']+config['N_HIST']) + 1

    # Get the train/val/test datasets
    dataset = TrafficDataset(config)

    # total of 44 days in the dataset, use 34 for training, 5 for val, 5 for test
    train, val, test = get_splits(dataset, config['N_SLOT'], (34, 5, 5))
    train_dataloader = DataLoader(train, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_dataloader = DataLoader(val, batch_size=config['BATCH_SIZE'], shuffle=True)
    test_dataloader = DataLoader(test, batch_size=config['BATCH_SIZE'], shuffle=True)

    # # Get gpu if you can
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    # # Configure and train model
    config['N_NODE'] = dataset.n_node
    model = model_train(train_dataloader, val_dataloader, config, device)

    # Test Model
    model_test(model, test_dataloader, device)

    # Example: Test using pretrained model
    #trained_model = load_from_checkpoint('./runs/model_12-05-142102.pt', config)
    #model_test(trained_model, test_dataloader, device)


if __name__ == "__main__":
    main()
