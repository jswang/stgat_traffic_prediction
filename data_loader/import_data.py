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

# def generate_weight_matrix(filename='./dataset/d07_text_meta_2012_05_02.txt'):
    # df = pd.read_csv(filename, delimiter='\t')
    # df = df[["ID", "Latitude", "Longitude"]].dropna()
    # # df_sample = df.sample(228, random_state=0)
    # # df_sample = df.loc(df[1] == [715898, 715900, 715905, 715906, 715907, 715912, 715913, 715915, 715916, 715917, 715918, 715919, 715920, 715921, 715922, 715923, 715924, 715925, 715926, 715927, 715928, 715929, 715930, 715932, 715933, 715935, 715938, 715941, 715944, 715947, 715949, 715950, 715957, 715958, 715959, 715961, 715963, 715964, 715966, 715969, 715970, 715971, 715972, 715973, 715974, 715977, 715979, 715980, 715981, 715992, 715994, 715996, 716007, 716008, 716009, 716010, 716011, 716012, 716013, 716014, 716015, 716016, 716017, 716020, 716021, 716023, 716026, 716028, 716029, 716030, 716032, 716033, 716035, 716036, 716038, 716039, 716040, 716041, 716043, 716044, 716045, 716046, 716047, 716050, 716054, 716055, 716057, 716058, 716061, 716063, 716064, 716065, 716066, 716067, 716069, 716072, 716075, 716076, 716078, 716081, 716084, 716087, 716088, 716090, 716091, 716092, 716096, 716101, 716116, 716126, 716130, 716140, 716141, 716142, 716143, 716145, 716146, 716148, 716149, 716150, 716151, 716152, 716153, 716154, 716155, 716156, 716157, 716158, 716159, 716160, 716161, 716162, 716163, 716165, 716166, 716167, 716168, 716170, 716171, 716172, 716174, 716176, 716178, 716181, 716182, 716183, 716184, 716185, 716187, 716188, 716189, 716190, 716191, 716192, 716193, 716195, 716196, 716197, 716199, 716200, 716203, 716204, 716205, 716206, 716207, 716208, 716209, 716210, 716211, 716213, 716214, 716215, 716216, 716217, 716222, 716223, 716224, 716225, 716226, 716227, 716228, 716229, 716230, 716231, 716232, 716233, 716234, 716235, 716237, 716238, 716240, 716241, 716243, 716244, 716245, 716246, 716249, 716250, 716252, 716253, 716254, 716255, 716256, 716258, 716259, 716260, 716261, 716262, 716263, 716264, 716265, 716267, 716268, 716269, 716271, 716272, 716273, 716274, 716275, 716276, 716279, 716281, 716282, 716283, 716284, 716285, 716287, 716289]

    # # ids = df_sample["ID"].to_numpy()
    # # lat = df_sample["Latitude"].to_numpy()
    # # lon = df_sample["Longitude"].to_numpy()

    # x = np.meshgrid(lat, lat)
    # lat1, lat2 = x[0], x[1]
    # x = np.meshgrid(lon, lon)
    # lon1, lon2 = x[0], x[1]

    # distances = haversine(lat1, lon1, lat2, lon2)

    # W = distance_to_weight(distances)

    # return ids, W


def distance_to_weight(W, sigma2=0.1, epsilon=0.5, gat_version=False):
    n = W.shape[0]
    W = W / 10000.
    W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
    # refer to Eq.10
    W = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask

    # If using the gat version of this, round to 0/1 and include self loops
    if gat_version:
        W[W>0] = 1
        W += np.identity(n)

    return W

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def parse_traffic_data(ids, filename='./dataset/d07_text_station_5min_2012_05_01.txt'):
    df = pd.read_csv(filename, delimiter=',', header=None, usecols=[0, 1, 11])
    # For every node, get all of the traffic points associated with it
    for id in ids:
        id_df = df.loc[df[1] == id]
        times = id_df[0].to_numpy()
        y = id_df[11].to_numpy()
        datetimes = [datetime.strptime(t, '%m/%d/%Y %H:%M:%S') for t in times]
        # TODO check that datetimes are all 5 minutes apart
        nans, x = nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])


# distances = pd.read_csv('./dataset/PeMSD7_W_228.csv', header=None).values
# W = distance_to_weight(distances, gat_version=False)
# ax = sns.heatmap(W, cmap="YlGnBu")
# plt.savefig("W.png", bbox_inches='tight', dpi=100)
# plt.show()

# W = distance_to_weight(distances, gat_version=True)
# ax = sns.heatmap(W, cmap="YlGnBu")
# plt.savefig("W.png", bbox_inches='tight', dpi=100)
# plt.show()

