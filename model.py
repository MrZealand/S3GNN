import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv

class StructureAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(structureAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=0)

        return (beta * z).sum(0)

class S3GNNLayer(nn.Module):

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(S3GNNLayer, self).__init__()
        
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.structure_attention = StructureAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        structure_embeddings = []

        for i, g in enumerate(gs):
            structure_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        structure_embeddings = torch.stack(structure_embeddings, dim=1)                  # (N, M, D * K)

        return self.structure_attention(structure_embeddings)                            # (N, D * K)

class S3GNN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(S3GNN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(S3GNNLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(S3GNNLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)