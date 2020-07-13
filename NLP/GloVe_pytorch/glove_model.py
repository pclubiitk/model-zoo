import torch
import torch.nn as nn

class Glove(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super(Glove, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)

        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

    def forward(self, i_indices, j_indices):
        Wi = self.wi(i_indices)
        Wj = self.wj(j_indices)
        Bi = self.bi(i_indices).squeeze()
        Bj = self.bj(j_indices).squeeze()

        return torch.sum(Wi * Wj, dim=1) + Bi + Bj
