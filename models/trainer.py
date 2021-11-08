# @Time     : Jan. 13, 2019 20:16
# @Author   : Veritas YIN
# @FileName : trainer.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from data_loader.data_utils import gen_batch
from os.path import join as pjoin
from models.st_gat import ST_GAT
import torch
import torch.optim as optim
import numpy as np
import time
import pandas as pd

# Training Constants
# TODO Maybe move these to a config file?
C_WEIGHT_DECAY = 5e-4
C_INITIAL_LR = 2e-4
C_BATCH_SIZE = 50
C_EPOCHS = 150

def train(model, data, train_idx, optimizer, loss_fn):
    """
    Forward pass and backward pass on a model
    """
    model.train()
    loss =0

    optimizer.zero_grad()
    y_pred= model(data.x)
    loss = loss_fn(y_pred[train_idx], data.y[train_idx])

    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()
    out = None
    out = model(data.x)
    y_pred = out

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

def model_train(dataset, args):
    """
    Train the ST-GAT model.
    """
    # Load and preprocess the data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If you use GPU, the device should be cuda
    print('Device: {}'.format(device))
    data = dataset.to(device)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    # dataset shape: T x F x N
    model = ST_GAT(in_channels=dataset.shape[1::], out_channels=dataset.shape[1::])
    optimizer = optim.Adam(args, lr=C_INITIAL_LR, weight_decay=C_WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss

    model.train()
    for epoch in range(C_EPOCHS):
        loss = train(model, data, train_idx, optimizer, loss_fn)
        # TODO define the evaluator
        train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)
        # TODO add tensorboard to visualize training over time
        print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')

def model_test(dataset, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If you use GPU, the device should be cuda
    print('Device: {}'.format(device))
    data = dataset.to(device)
    split_idx = dataset.get_idx_split()
    model = ST_GAT(in_channels=dataset.shape[1::], out_channels=dataset.shape[1::])
    train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)

    print(f'Test:, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')