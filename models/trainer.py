import torch
import torch.optim as optim
from tqdm import tqdm
import time
import os

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
    model.to(device)

    mae = torch.zeros(3, dtype=torch.float)
    rmse = torch.zeros(3, dtype=torch.float)
    mape = torch.zeros(3, dtype=torch.float)
    n = 0

    # Evaluate model on all data
    for _, batch in enumerate(dataloader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch, device)

            truth = batch.y.view(pred.shape)
            truth = un_z_score(truth, dataloader.dataset.mean, dataloader.dataset.std_dev)
            pred = un_z_score(pred, dataloader.dataset.mean, dataloader.dataset.std_dev)
            rmse += torch.tensor([RMSE(truth[0:3], pred[0:3]), RMSE(truth[0:6], pred[0:6]), RMSE(truth[0:9], pred[0:9])])
            mae += torch.tensor([MAE(truth[0:3], pred[0:3]), MAE(truth[0:6], pred[0:6]), MAE(truth[0:9], pred[0:9])])
            mape += torch.tensor([MAPE(truth[0:3], pred[0:3]), MAPE(truth[0:6], pred[0:6]), MAPE(truth[0:9], pred[0:9])])
            n += 1
    rmse, mae, mape = rmse / n, mae / n, mape / n

    print(f'{type}, MAE 15/30/45: {mae}, RMSE 15/30/45: {rmse}, MAPE 15/30/45: {mape}')

    #get the average score for each metric in each batch
    return rmse, mae, mape

def train(model, device, train_dataloader, optimizer, loss_fn, epoch):
    model.train()
    for _, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
        batch = batch.to(device)
        optimizer.zero_grad()
        y_pred = torch.squeeze(model(batch, device))
        loss = loss_fn()(y_pred.float(), torch.squeeze(batch.y).float())
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

    return loss

def model_train(train_dataloader, val_dataloader, config, device):
    """
    Train the ST-GAT model. Evaluate on validation dataset as you go.
    """

    # Make the model. Each datapoint in the graph is 228x12: N x F (N = # nodes, F = time window)
    model = ST_GAT(in_channels=config['N_HIST'], out_channels=config['N_PRED'], n_nodes=config['N_NODE'])
    optimizer = optim.Adam(model.parameters(), lr=config['INITIAL_LR'], weight_decay=config['WEIGHT_DECAY'])
    loss_fn = torch.nn.MSELoss

    model.to(device)

    # For every epoch, train the model on training dataset. Evaluate model on validation dataset
    for epoch in range(config['EPOCHS']):
        loss = train(model, device, train_dataloader, optimizer, loss_fn, epoch)
        print(f"Loss: {loss:.3f}")
        if epoch % 5 == 0:

            train_mae, train_rmse, train_mape = eval(model, device, train_dataloader, 'Train')
            val_mae, val_rmse, val_mape = eval(model, device, val_dataloader, 'Valid')
            writer.add_scalar("MAE/train", train_mae, epoch)
            writer.add_scalar("RMSE/train", train_rmse, epoch)
            writer.add_scalar("MAPE/train", train_mape, epoch)

            writer.add_scalar("MAE/val", val_mae, epoch)
            writer.add_scalar("RMSE/val", val_rmse, epoch)
            writer.add_scalar("MAPE/val", val_mape, epoch)

    writer.flush()
    # Save the model
    timestr = time.strftime("%m-%d-%H%M%S")
    torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            }, os.path.join(config["CHECKPOINT_DIR"], f"model_{timestr}.pt"))

    return model

def model_test(model, test_dataloader, device):
    eval(model, device, test_dataloader, 'Test')

def load_from_checkpoint(checkpoint_path, config):
    model = ST_GAT(in_channels=config['N_HIST'], out_channels=config['N_PRED'], num_nodes=config['N_NODE'])

    #checkpoint = torch.load(checkpoint_path)
    #gitmodel.load_state_dict(checkpoint['model_state_dict'])

    return model

