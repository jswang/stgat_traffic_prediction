import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split

#Given the original data, come up with one big dataset.
class TrafficDataset():
    def __init__(self, data, W, n_hist, n_pred):
        self.mean = np.mean(data)
        self.std_dev = np.std(data)
        self.data = self.speed2vec(W, data, n_hist, n_pred)

    def speed2vec(self, W, data, n_hist, n_pred):
        """
        Given some data, figure out T, F, and N and return graphs for F timewindow
        :param W: Weight matrix
        :param data: Raw data to process
        :param n_hist: Number of timesteps in historical window to consider
        :param n_pred: Number of timestemps into the future to predict (ground truth)
        """
        self.n_datapoints, self.n_node = data.shape
        self.n_window = n_hist + n_pred
        # The number of actual sequences you can make
        n_sequences = self.n_datapoints - self.n_window + 1

        # manipulate nxn matrix into 2xnum_edges
        edge_index = np.zeros((2, self.n_node**2))
        # create an edge_attr matrix with our weights  (num_edges x 1) --> our edge features are dim 1
        edge_attr = np.zeros((self.n_node**2, 1)) # TODO should this be 1-dim?
        num_edges = 0
        for i in range(self.n_node):
            for j in range(self.n_node):
                if W[i, j] != 0.:
                    edge_index[0, num_edges] = i
                    edge_index[1, num_edges] = j
                    edge_attr[num_edges] = W[i, j]
                    num_edges += 1
        edge_index = np.resize(edge_index, (2, num_edges))
        edge_attr = np.resize(edge_attr, (num_edges, 1))

        sequences = []
        # T x F x N
        for t in range(n_sequences):
            # for each time point construct a different graph with data object
            # Docs here: https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
            g = Data()
            g.__num_nodes__ = self.n_node

            g.edge_index = torch.from_numpy(edge_index)
            g.edge_attr  = torch.from_numpy(edge_attr)

            # (F,N) switched to (N,F)
            full_window = np.swapaxes(data[t:t+self.n_window, :], 0, 1)
            g.x = full_window[:, 0:n_hist]
            g.y = full_window[:, n_hist::]
            sequences += [g]

        return sequences


    def __len__(self):
        """
        Total number of samples
        """
        len(self.data)


    def __get_item__(self, index):
        """
        Generates one sample of data
        :param index: Index of data to retreive
        """
        data = self.sequences[index]
        data = (data - self.mean) / self.std_dev
        X = data[0:self.n_hist]
        y = data[self.n_hist::]
        return X,y


def get_splits(dataset: TrafficDataset, splits):
    """
    Given the data, split it into random subsets of train, val, and test as given by splits
    :param dataset: TrafficeDataset object
    :param splits: (train, val, test) ratios
    """
    split_train, split_val, split_test = splits
    train, test = train_test_split(dataset.data, test_size=split_train, random_state=1)
    val, test = train_test_split(test, test_size=(split_val)/(1-split_train), random_state=1)

    return (train, val, test)
