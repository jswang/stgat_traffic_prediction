# @Time     : 11/7/21
# @Author   : Julie Wang
# @FileName : data_utils.py

from sklearn.model_selection import train_test_split

def get_splits(data):
    """
    Given the data, split it into 60% training, 20% validation, 20% test
    TODO: make splits configurable
    """
    train, test = train_test_split(data, test_size=0.4, random_state=1)
    test, val = train_test_split(test, test_size=0.5, random_state=1)

    return (train, val, test)