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

        self.bn1  = torch.nn.BatchNorm1d(12)

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size=self.num_nodes, hidden_size=32, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_normal(param)
        self.lstm2 = torch.nn.LSTM(input_size=32, hidden_size=128, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_normal(param)
        # fully-connected neural network
        self.linear = torch.nn.Linear(128, self.num_nodes)
        torch.nn.init.xavier_normal(self.linear.weight)

    def forward(self, data, device):
        x, edge_index = data.x, data.edge_index
        # apply dropout
        if device == 'cpu':
            x = torch.FloatTensor(x)
        else:
            x = torch.cuda.FloatTensor(x)

        # gat layer
        x = self.gat(x, edge_index)
        x = self.bn1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.log_softmax(x, dim=1)
        
        # output of gat: [11400, 12]

        # RNN: 2 LSTM
        # incoming: x is (batchsize*num_nodes, seq_length), change into (batch_size, num_nodes, seq_length)
        x = torch.reshape(x, (data.num_graphs, int(data.num_nodes/data.num_graphs), data.num_features))
        # for lstm: x should be (seq_length, batch_size, num_nodes)
        # sequence length = 12, batch_size = 50, input_dim = 228
        x = torch.movedim(x, 2, 0)
        # [12, 50, 228] -> [12, 50, 32]
        x, _ = self.lstm1(x)
        # [12, 50, 32] -> [12, 50, 128]
        x, _ = self.lstm2(x)
        # [12,50,128] -> [12, 50, 228]
        x = self.linear(x)
        # Then, select the last 9 outputs for the prediction into the future
        # [12, 50, 228] -> [9, 50, 228]
        x = x[-self.out_channels:, :, :]
        # [9, 50, 228] ->  [11400, 9]


        x = torch.movedim(x, 0, 2)
        s = x.shape
        x = torch.reshape(x, (s[0]*s[1], s[2]))

        return x
