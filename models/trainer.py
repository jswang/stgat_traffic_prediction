# @Time     : 11/12/21
# @Author   : Julie Wang
# @FileName : trainer.py

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

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
def eval(model, device, dataloader, type=''):
    """
    Evaluation function to evaluate model on data
    """
    model.eval()

    mae = 0
    rmse = 0
    mape = 0
    n = 0


    # Evaluate model on all data
    for _, batch in enumerate(dataloader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            truth = batch.y.view(pred.shape)
            rmse += mean_squared_error(truth, pred, squared=False)
            mae += mean_absolute_error(truth, pred)
            mape += mean_absolute_percentage_error(truth, pred)
            n += 1
    rmse, mae, mape = rmse / n, mae / n, mape / n

    print(f'{type}, mae: {mae}, rmse: {rmse}, mape: {mape}')

    #get the average score for each metric in each batch
    return rmse, mae, mape


def model_train(train_dataloader, val_dataloader, config):
    """
    Train the ST-GAT model. Evaluate on validation dataset as you go.
    """
    print("Training Model")
    # Get GPU if you can
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Make the model.
    # each datapoint in the graph is 228x12: N x F (N = # nodes, F = time window)
    # TODO pass in n_hist and n_pred better
    model = ST_GAT(in_channels=config['n_hist'], out_channels=config['n_pred'], num_nodes=config['n_node'])
    optimizer = optim.Adam(model.parameters(), lr=config['C_INITIAL_LR'], weight_decay=config['C_WEIGHT_DECAY'])
    loss_fn = torch.nn.MSELoss

    # For every epoch, train the model on training dataset. Evaluate model on validation dataset
    for epoch in range(1):#config['C_EPOCHS']): # only do a couple of epochs for now to see what's happening
        loss = train(model, device, train_dataloader, optimizer, loss_fn)

        if epoch %10 == 0:
            print(f"Epoch {epoch}, loss: {loss}")
            eval(model, device, train_dataloader, 'Train')
            eval(model, device, val_dataloader, 'Valid')

    return model

def model_test(model, test_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval(model, device, test_dataloader, 'Test')