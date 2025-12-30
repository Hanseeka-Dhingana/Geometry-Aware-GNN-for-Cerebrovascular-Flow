import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class VesselGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(4, 32)
        self.conv2 = GCNConv(32, 16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
