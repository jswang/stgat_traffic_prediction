#!/usr/bin/python3

# @Time     : 11/7/21
# @Author   : Julie Wang
# @FileName : main.py

import argparse
import torch

from data_loader.dataloader import TrafficDataset, get_splits
from models.trainer import load_from_checkpoint, model_train, model_test
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
    parser.add_argument('--graph', type=str, default='PeMSD7_W_228', help='Graph name defaults to PeMSD7_W_228')
    parser.add_argument('--graph_data', type=str, default='PeMSD7_V_228', help='Graph dataset name defaults to PeMSD7_V_228')
    parser.add_argument('--inf_mode', type=str, default='merge')

    return parser.parse_args()

def main():
    """
    Main function to train and test a model.
    """
    args = parse_args()

    # Constant config to use througout
    config = {
        'C_BATCH_SIZE': args.batch_size,
        'C_EPOCHS': 150,
        'C_WEIGHT_DECAY': 5e-4,
        'C_INITIAL_LR': 2e-4,
        'C_CHECKPOINT_DIR': './runs',
        'N_PRED': args.n_pred,
        'N_HIST': args.n_hist,
    }

    # Get the train/val/test datasets
    dataset = TrafficDataset(args.n_hist, args.n_pred)
    train, val, test = get_splits(dataset, (0.6, 0.2, 0.2))
    train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=args.batch_size, shuffle=True)

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
