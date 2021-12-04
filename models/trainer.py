# @Time     : 11/12/21
# @Author   : Julie Wang
# @FileName : trainer.py

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.st_gat import ST_GAT
from utils.math_utils import *

def train(model, device, dataloader, optimizer, loss_fn):
    """
    Forward pass and backward pass on a model
    """
    model.train()
    loss=0

    for _, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()
        y_pred = torch.squeeze(model(batch))
        loss = loss_fn()(y_pred.float(), torch.squeeze(batch.y).float())
        loss.backward()
        optimizer.step()

    return loss.item()

@torch.no_grad()
def eval(model, device, dataloader):
    """
    Evaluation function to evaluate model on data

    """
    model.eval()
    y_true = []
    y_pred = []

    # Run model on all data
    for _, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    # TODO return something


def model_train(train_dataloader, val_dataloader, config):
    """
    Train the ST-GAT model. Evaluate on validation dataset as you go.
    """
    # Get GPU if you can
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Make the model. TODO add RNN here
    # each datapoint in the graph is 228 x12: N x F (N = # nodes, F = time window)
    # TODO pass in n_hist and n_pred better
    #model = ST_GAT(in_channels=config['n_hist'], out_channels=config['n_pred'], num_nodes=config['n_node'])
    model = ST_GAT(in_channels=config['n_hist'], out_channels=228, num_nodes=config['n_node'])
    optimizer = optim.Adam(model.parameters(), lr=config['C_INITIAL_LR'], weight_decay=config['C_WEIGHT_DECAY'])
    loss_fn = torch.nn.MSELoss

    # For every epoch, train the model on training dataset. Evaluate model on validation dataset
    for epoch in range(config['C_EPOCHS']):
        loss = train(model, device, train_dataloader, optimizer, loss_fn)
        train_result = eval(model, device, train_dataloader)
        val_result = eval(model, device, val_dataloader)
        # TODO add tensorboard to visualize training over time
        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_acc:.2f}%, Valid: {100 * valid_acc:.2f}%')

def model_test(dataset, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If you use GPU, the device should be cuda
    print('Device: {}'.format(device))
    data = dataset.to(device)
    split_idx = dataset.get_idx_split()
    model = ST_GAT(in_channels=dataset.shape[1::], out_channels=dataset.shape[1::])
    train_acc, valid_acc, test_acc = eval(model, data, split_idx)

    print(f'Test:, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')