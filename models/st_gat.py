import tensorflow as tf
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ST_GAT(torch.nn.Module):

    def __init__(self, in_channels, out_channels, heads=8, dropout=0.6):
        super(ST_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        # single graph attentional layer with 8 attention heads
        self.gat = GATConv(in_channels, heads, heads, dropout, dropout)

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(self.out_channels, 32)
        self.lstm2 = torch.nn.LSTM(32, 128)

        # fully-connected neural network
        self.linear = torch.nn.Linear(128, self.out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # apply dropout
        x = torch.FloatTensor(x) # TODO probably should have x's as tensor in dataset
        x = F.dropout(x, self.dropout, training=self.training)
        # gat layer
        x = self.gat(x, edge_index)
        # apply Leaky RELU then softmax
        x = F.leaky_relu(x)
        x = F.log_softmax(x, dim=1)

        # RNN: 2 LSTM
        x = self.lstm1(x)
        x = self.lstm2(x)

        # final output layer
        x = self.linear(x)

        return x



# class STGAT: # legacy  -- to remove (we are going to use the pytorch implementation)
#     '''
#     STGAT block as described in Zhang et al's ST-GAT: Deep Learning Approach for Traffic Forecasting.
#     Multihead Attention adapted from the graph attentional operator from the `"Graph Attention Networks".
#     Ref a PyG Pytorch equivalent source code: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv
#     Experimental settings gigven are: ST-GAT uses dropout and early stopping.
#     :param x: tensor, [T, N, F] where T  = lenght of temporal sequence, N = num nodes in batch, F = size of fixed historical window.
#     :param channels: list, [in_channels, out_channels]
#     :param K: number of heads for multi-head attention (8 heads used in paper)
#     '''
#     def  __init__(self, channels, K=8,  dropout=0.0, edge_dim=None, add_self_loops=True, has_bias=True):
#         # initialize inputs
#         self.in_channels, self.out_channels = channels
#         self.heads = K
#         self.concat = concat
#         self.dropout = dropout
#         self.edge_dim = edge_dim
#         self.has_bias =  bias
#         self.add_self_loops = add_self_loops
#         # assume concat is true since that is explicitly given in the ST-GAT paper
#         # negative slope  and fill-value -- might need to add this?
#         self.bias = self.heads * out_channels if self.has_bias else None
#         # we do not consider a bipartite case, given our data is not bipartite (unlike PyG implementation)

#         # initialize linear transformations for source and target nodes
#         self.lin_src = tf.keras.models.Sequential()
#         self.lin_src.add(tf.keras.Input(in_channels[0]))
#         self.lin_src.add(tf.keras.Dense(self.heads * out_channels), bias=self.bias)

#         self.lin_dst = tf.keras.models.Sequential()
#         self.lin_dst.add(tf.keras.Input(in_channels[1]))
#         self.lin_dst.add(tf.kerasDense(self.heads * out_channels), bias=self.bias)

#         # initialize attention matrices for source and target nodes
#         self.attn_src = tf.random.normal(shape=(1, self.heads, self.out_channels))
#         self.attn_dst  = tf.random.normal(shape=(1, self.heads, self.out_channels))


#         if edge_dim is not None:
#             self.lin_edge = tf.keras.models.Sequential()
#             self.lin_edge.add(tf.keras.Input(edge_dim)
#             self.line_edge.add(tf.keras.Dense(heads*out_channels), bias=self.bias)

#             self.att_edge = tf.random.normal(shape=(1, self.heads, self.out_channels))
#             (edge_dim, heads * out_channels, bias=False,
#                                    weight_initializer='glorot')
#             self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
#         else:
#             self.lin_edge = None


#         #self.reset_parameters()

#     def reset_parameters(self):
#         '''
#         Reset parameters.
#         '''
#         # need to figure out if there is a TF equivalent for this?
#         pass

#     def forward(self, x, edge_index, edge_attr, size, return_attention_weights):
#         '''
#         Perform forward pass of multi-layer attention. Either mean or
#         :param x: Speed2Vec produced input, shape=[T,N,F]
#         :param edge_index: adjacency matrix, shape=[N,N]
#         :param edge_attr: edge features.
#         '''
#         # need to fully understand and implement this.

#         # Mapping the PyG implementation

#         # (1) adapt dimensionality
#         # (2) apply the attention coefficients to x
#         # (3) deal with self loops
#         # (4) conduct concatenation and then mean

#         # Current Questions:
#         # ---- What does size refer to here?
#         # ---- PyG code suggests you do either concatenation OR mean, NOT both; need to clarify how this works for our ST-GAT
#         # ---- What is the alpha term here?
#         # ---- return attention weights in addition to next iteration of H embeddings (before activation)


#     def message(self):
#         '''
#         Message unit.
#         '''

#         # applies activation functions to get the final embedding outputs in addition to dropout
