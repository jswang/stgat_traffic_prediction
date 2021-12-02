import os
import numpy as np
import pandas as pd
import torch
from dataloader import TrafficDataset, get_splits
from tqdm import tqdm

# Config that's passed around
config = {
    'C_BATCH_SIZE': 50,
    'C_EPOCHS': 150,
    'C_WEIGHT_DECAY': 5e-4,
    'C_INITIAL_LR': 2e-4
}

# Data Preprocessing and loading
data = pd.read_csv('./dataset/PeMSD7_V_228.csv', header=None).values
dataset = TrafficDataset(data, 12, 9)
(train, val, test) = get_splits(dataset, (0.6, 0.2, 0.2))
train_dataloader = torch.utils.data.DataLoader(train, batch_size=config['C_BATCH_SIZE'], shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=config['C_BATCH_SIZE'], shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=config['C_BATCH_SIZE'], shuffle=True)
