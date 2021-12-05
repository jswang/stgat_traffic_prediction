import numpy as np
import pandas as pd
import torch

def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_node, n_node].
    '''
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

def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: torch array, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: torch array, z-score normalized array.
    '''
    return (x - mean) / std

def un_z_score(x_normed, mean, std):
    return x_normed * std  + mean


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, MAPE averages on all elements of input.
    '''
    #return torch.mean(torch.abs(v_ - v) / (v + 1e-15))
    # make MAPE a percentage score; also include denominator inside absolute value to fit with the equation given in paper
    return torch.mean(torch.abs((v_ - v)/ (v + 1) * 100))


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, RMSE averages on all elements of input.
    '''
    return torch.sqrt(torch.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, MAE averages on all elements of input.
    '''
    return torch.mean(torch.abs(v_ - v))
