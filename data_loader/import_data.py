import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from datetime import datetime



def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates distance between two lat/lon points using haversine formulat
    """
    R = 6372.8 # km.

    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))

    return R * c

# TRACY: this is how you make the heatmap. can you modify the axes and title to clean it up?
def visualize(data):
    ax = sns.heatmap(data, annot=True)
    plt.savefig("W.png", bbox_inches='tight', dpi=100)
    plt.show()

def generate_weight_matrix(filename='../dataset/d07_text_meta_2012_05_02.txt'):
    df = pd.read_csv(filename, delimiter='\t')
    df = df[["ID", "Latitude", "Longitude"]].dropna()
    df_sample = df.sample(228, random_state=0)

    ids = df_sample["ID"].to_numpy()
    lat = df_sample["Latitude"].to_numpy()
    lon = df_sample["Longitude"].to_numpy()

    x = np.meshgrid(lat, lat)
    lat1, lat2 = x[0], x[1]
    x = np.meshgrid(lon, lon)
    lon1, lon2 = x[0], x[1]

    distances = haversine(lat1, lon1, lat2, lon2)

    W = distance_to_weight(distances)

    return ids, W


def distance_to_weight(distances, sigma2=100, epsilon=0.5, use_ints=False):
    """
    Load weight matrix function.
    :param W: str, weight matrix
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param use_ints: bool, if True, rounds all weights to 0s and 1s
    :return: np.ndarray, [n_node, n_node].
    """
    n = distances.shape[0]
    mask = np.ones([n, n]) - np.identity(n)
    # refer to Eq. 20 in Graph Attention Network
    W = np.exp(-distances*distances / sigma2) * mask
    W[W < epsilon] = 0

    if use_ints:
      W[W>0] = 1
      # Add self loop
      W += np.identity(n)

    return W

def parse_traffic_data(ids, filename='../dataset/d07_text_station_5min_2012_05_01.txt'):
    df = pd.read_csv(filename, delimiter=',', header=None, usecols=[0, 1, 11])
    # For every node, get all of the traffic points associated with it
    for id in ids:
        id_df = df.loc[df[1] == id]
        times = id_df[0].to_numpy()
        values = id_df[11].to_numpy()
        datetime_object = [datetime.strptime(t, '/%Y %I:%M%p') for t in times]

    print("here")

ids, W = generate_weight_matrix()
parse_traffic_data([715898, 715900, 715903], '../dataset/test.txt')