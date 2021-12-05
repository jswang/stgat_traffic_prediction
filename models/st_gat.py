import tensorflow as tf
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ST_GAT(torch.nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes, heads=8, dropout=0.6):
        super(ST_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.num_nodes = num_nodes

        # single graph attentional layer with 8 attention heads
        self.gat = GATConv(in_channels=self.in_channels, out_channels=12,
            heads=heads, dropout=self.dropout, concat=False) # use the number of output channels equivalent to width of data for predictions (9)

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size=self.num_nodes, hidden_size=32, num_layers=1)
        self.lstm2 = torch.nn.LSTM(input_size=32, hidden_size=128, num_layers=1)

        # fully-connected neural network
        self.linear = torch.nn.Linear(128, self.out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # apply dropout
        x = torch.FloatTensor(x)
        # gat layer
        x = self.gat(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.log_softmax(x, dim=1)

        # RNN: 2 LSTM
        # incoming: x is (batchsize*num_nodes, seq_length), change into (batch_size, num_nodes, seq_length)
        x = torch.reshape(x, (data.num_graphs, int(data.num_nodes/data.num_graphs), data.num_features)) #TODO: should this be batch then nodes or nodes then batch?
        # for lstm: x should be (seq_length, batch_size, num_nodes)
        # sequence length = 12, batch_size = 50, input_dim = 228
        
        #x = torch.reshape(x, (data.num_graphs, int(data.num_nodes/data.num_graphs), 9)) # also currently hard codeds
        
        x = torch.movedim(x, 2, 0)
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # final output layer, x coming in is 12,50,128.
        x = self.linear(x)
        x = torch.reshape(x, (data.num_features, data.num_nodes)).T
        
        return x
