# @Time     : Jan. 10, 2019 15:26
# @Author   : Veritas YIN
# @FileName : data_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from utils.math_utils import z_score

import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


def speed2vec(len_seq, data_seq, offset, n_frame, n_node, day_slot, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_node: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_node, C_0].
    '''
    # total number of slots that you can make out of the day.
    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_node, C_0))
    # for every target date
    for i in range(len_seq):
        # for every target slot
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_node, C_0])
    return tmp_seq

# TODO figure out how to use the pytorch dataloader instead
def datagen(file_path, splits, n_his, n_pred, day_slot=288):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param splits: array, dataset splits ratios: train, validation, test.
    :param n_his: F, or the number of history timeslices
    :param n_pred: int, Number of timeslots to predict into future. n_pred = 9 (45 min into future).
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default). 24 hours * 60 min/hour / 5 min/slot
    :return: dict, dataset that contains training, validation and test with stats.
    '''

    # generate training, validation and test data
    # The dataset is already linearly interpolated and cleaned up for us.
    # (all timepoints) x N
    data_seq = pd.read_csv(file_path, header=None).values

    n_datapoints, n_node = data_seq.shape

    # The number of actual sequences you can make
    n_sequences = n_datapoints - (n_his + n_pred) + 1
    # T x F x N
    sequences = np.zeros((n_sequences, n_his + n_pred, n_node))
    for i in range(n_sequences):
        sequences[i] = data_seq[i:i+n_his + n_pred, :]
    # Split up in training, validation, and test.
    splits = (splits*n_sequences).astype(int)

    # Randomize
    sequences = np.random.permutation(sequences)
    train_dataset = sequences[0:splits[0]]
    val_dataset = sequences[splits[0]: splits[0] + splits[1]]
    test_dataset = sequences[splits[0] + splits[1]: -1]

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    train_stats = {'mean': np.mean(train_dataset), 'std': np.std(train_dataset)}

    # Calculate the zscore
    x_train = z_score(train_dataset, train_stats['mean'], train_stats['std'])
    x_val = z_score(val_dataset, train_stats['mean'], train_stats['std'])
    x_test = z_score(test_dataset, train_stats['mean'], train_stats['std'])

    dataset = Dataset( {'train': x_train, 'val': x_val, 'test': x_test}, train_stats)
    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_node, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]
