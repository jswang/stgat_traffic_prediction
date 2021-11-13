# @Time     : Jan. 13, 2019 20:16
# @Author   : Veritas YIN
# @FileName : trainer.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from os.path import join as pjoin
from models.st_gat import ST_GAT
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

def train(model, device, dataloader, optimizer, loss_fn):
    """
    Forward pass and backward pass on a model
    """
    model.train()
    loss=0

    for _, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            y_pred = torch.squeeze(model(batch))
            loss = loss_fn(y_pred, torch.squeeze(batch.y).float())
            loss.backward()
            optimizer.step()

    return loss.item()



@torch.no_grad()
def eval(model, device, dataloader, evaluator):
    """
    Evaluation function to evaluate model on data
    """
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

def model_train(train_dataloader, val_dataloader, config):
    """
    Train the ST-GAT model. Evaluate on validation dataset as you go.
    """
    # Get GPU if you can
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Make the model. TODO add RNN here
    model = ST_GAT(in_channels=train_dataloader.shape[1::], out_channels=train_dataloader.shape[1::])
    optimizer = optim.Adam(lr=config.C_INITIAL_LR, weight_decay=config.C_WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss

    # For every epoch, train the model on training dataset. Evaluate model on validation dataset
    for epoch in range(config.C_EPOCHS):
        loss = train(model, device, train_dataloader, optimizer, loss_fn)
        train_result = eval(model, device, train_dataloader, evaluator)
        val_result = eval(model, device, val_dataloader, evaluator)
        # TODO add tensorboard to visualize training over time
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_acc:.2f}%, Valid: {100 * valid_acc:.2f}%')

def model_test(dataset, config):
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