import torch
import numpy as np

class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, data, n_hist, n_pred):
        self.mean = torch.mean(data)
        self.std_dev = torch.std(data)
        self.speed2vec(data, n_hist, n_pred)

    def speed2vec(self, data, n_hist, n_pred):
        """
        Given some data, figure out T, F, and N and populate self.sequences

        Keyword arguments:
        data -- Raw data to process
        n_hist -- Number of timesteps in historical window to consider
        n_pred -- Number of timestemps into the future to predict (ground truth)
        """
        self.n_datapoints, self.n_node = data.shape
        self.n_window = n_hist + n_pred

        # The number of actual sequences you can make
        n_sequences = self.n_datapoints - self.n_window + 1
        # T x F x N
        sequences = np.zeros((n_sequences, self.n_window, self.n_node))
        for i in range(n_sequences):
            sequences[i] = data[i:i+self.n_window, :]

        self.sequences = sequences


    def __len__(self):
        """
        Total number of samples
        """
        len(self.sequences)


    def __get_item__(self, index):
        """
        Generates one sample of data

        Keyword arguments:
        index -- Index of data to retreive
        """
        data = self.sequences[index]
        data = (data - self.mean) / self.std_dev
        X = data[0:self.n_hist]
        y = data[self.n_hist::]
        return X,y