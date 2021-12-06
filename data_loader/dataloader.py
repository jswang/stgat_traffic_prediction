import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.data import InMemoryDataset, Data
from shutil import copyfile

from utils.math_utils import *

def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    """
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_node, n_node].
    """
    W = pd.read_csv(file_path, header=None).values

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq. 20 in Graph Attention Network
        W = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
        # Add self loop
        W += np.identity(n)

    return W

#Given the original data, come up with one big dataset.
class TrafficDataset(InMemoryDataset):
    def __init__(self, n_hist, n_pred, root='', transform=None, pre_transform=None):
        self.n_hist = n_hist
        self.n_pred = n_pred
        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.n_node, self.mean, self.std_dev = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return [os.path.join(self.raw_dir, 'PeMSD7_W_228.csv'), os.path.join(self.raw_dir, 'PeMSD7_V_228.csv')]

    @property
    def processed_file_names(self):
        return ['./weight.pt', './data.pt']

    def download(self):
        #TODO: need to implement for other data types
        copyfile('./dataset/PeMSD7_W_228.csv', os.path.join(self.raw_dir, 'PeMSD7_W_228.csv'))
        copyfile('./dataset/PeMSD7_V_228.csv', os.path.join(self.raw_dir, 'PeMSD7_V_228.csv'))

    def process(self):
        """
        Process the raw datasets into saved .pt dataset for later use.
        Note that any self.fields here wont exist if loading straight from the .pt file
        """
        # Load weighted adjacency matrix W, save it because it's been processed
        W = weight_matrix(self.raw_file_names[0])
        torch.save(W, self.processed_paths[0])

        # Data Preprocessing and loading
        data = pd.read_csv(self.raw_file_names[1], header=None).values
        # Technically using the validation and test datasets here, but it's fine, would normally get the
        # mean and std_dev from a large dataset
        mean =  np.mean(data)
        std_dev = np.std(data)
        data = z_score(data, np.mean(data), np.std(data))

        n_datapoints, n_node = data.shape
        n_window = self.n_hist + self.n_pred
        # The number of actual sequences you can make
        n_sequences = n_datapoints - n_window + 1

        # manipulate nxn matrix into 2xnum_edges
        edge_index = torch.zeros((2, n_node**2), dtype=torch.long)
        # create an edge_attr matrix with our weights  (num_edges x 1) --> our edge features are dim 1
        edge_attr = torch.zeros((n_node**2, 1))
        num_edges = 0
        for i in range(n_node):
            for j in range(n_node):
                if W[i, j] != 0.:
                    edge_index[0, num_edges] = i
                    edge_index[1, num_edges] = j
                    edge_attr[num_edges] = W[i, j]
                    num_edges += 1
        # using resize_ to just keep the first num_edges entries
        edge_index = edge_index.resize_(2, num_edges)
        edge_attr = edge_attr.resize_(num_edges, 1)

        sequences = []
        # T x F x N
        for t in range(n_sequences):
            # for each time point construct a different graph with data object
            # Docs here: https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
            g = Data()
            g.__num_nodes__ = n_node

            g.edge_index = edge_index
            g.edge_attr  = edge_attr

            # (F,N) switched to (N,F)
            full_window = np.swapaxes(data[t:t+n_window, :], 0, 1)
            g.x = torch.FloatTensor(full_window[:, 0:self.n_hist])
            g.y = torch.FloatTensor(full_window[:, self.n_hist::])
            sequences += [g]

        # Make the actual dataset
        data, slices = self.collate(sequences)
        torch.save((data, slices, n_node, mean, std_dev), self.processed_paths[1])

def get_splits(dataset: TrafficDataset, splits):
    """
    Given the data, split it into random subsets of train, val, and test as given by splits
    :param dataset: TrafficDataset object to split
    :param splits: (train, val, test) ratios
    """
    split_train, split_val, _ = splits
    dataset = dataset.shuffle()
    i = int(split_train * len(dataset))
    j = int(split_val * len(dataset))
    train = dataset[:i]
    val = dataset[i:i+j]
    test = dataset[i+j:]

    return train, val, test
