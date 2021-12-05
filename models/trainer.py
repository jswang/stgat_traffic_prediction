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
from torch.utils.tensorboard import SummaryWriter

# Make a tensorboard writer
writer = SummaryWriter()

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
    # Get GPU if you can TODO put dataset onto GPU?
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Make the model. Each datapoint in the graph is 228x12: N x F (N = # nodes, F = time window)
    model = ST_GAT(in_channels=config['N_HIST'], out_channels=config['N_PRED'], num_nodes=config['N_NODE'])
    optimizer = optim.Adam(model.parameters(), lr=config['C_INITIAL_LR'], weight_decay=config['C_WEIGHT_DECAY'])
    loss_fn = torch.nn.MSELoss

    model.train()

    # For every epoch, train the model on training dataset. Evaluate model on validation dataset
    for epoch in range(5):#config['C_EPOCHS']): # only do a couple of epochs for now to see what's happening

        for _, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = batch.to(device)
            optimizer.zero_grad()
            y_pred = torch.squeeze(model(batch))
            loss = loss_fn()(y_pred.float(), torch.squeeze(batch.y).float())
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print(f"Epoch {epoch}, loss: {loss}")
            train_mae, train_rmse, train_mape = eval(model, device, train_dataloader, 'Train')
            val_mae, val_rmse, val_mape = eval(model, device, val_dataloader, 'Valid')
            writer.add_scalar("MAE/train", train_mae, epoch)
            writer.add_scalar("RMSE/train", train_rmse, epoch)
            writer.add_scalar("MAPE/train", train_mape, epoch)

            writer.add_scalar("MAE/val", val_mae, epoch)
            writer.add_scalar("RMSE/val", val_rmse, epoch)
            writer.add_scalar("MAPE/val", val_mape, epoch)

    writer.flush()

    return model

def model_test(model, test_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval(model, device, test_dataloader, 'Test')