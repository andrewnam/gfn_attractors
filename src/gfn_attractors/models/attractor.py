import torch
from torch import nn

from .discretizer import DiscretizeModule
from .helpers import SafeEmbedding, MLP


class RNNAttractorModel(DiscretizeModule):

    def __init__(self, vocab_size, dim_z, num_layers=1):
        super().__init__(vocab_size, None)
        self.dim_z = dim_z
        self.num_layers = num_layers

        self.embedding = SafeEmbedding(self.num_tokens, self.dim_z)
        self.rnn = nn.RNN(self.dim_z, self.dim_z, num_layers=self.num_layers, nonlinearity='relu', batch_first=True)

    def forward(self, w):
        """
        w: tensor with shape (..., length)
        returns:
            h_w: tensor with shape (..., dim_z)
        """
        shape = w.shape[:-1]
        w = w.view(-1, w.shape[-1])
        h_w = self.embedding(w)
        h_w = h_w * (w > self.eos).unsqueeze(-1)
        h_w, _ = self.rnn(h_w)
        return h_w[:,-1].view(*shape, self.dim_z)


class RecurrentMLPAttractorModel(DiscretizeModule):

    def __init__(self, vocab_size, dim_z, dim_h=128, num_layers=1, residual=True, nonlinearity=nn.ReLU()):
        super().__init__(vocab_size, None)
        self.dim_z = dim_z
        self.dim_h = dim_h
        self.num_layers = num_layers
        self.residual = residual
        
        self.embedding = SafeEmbedding(self.num_tokens, self.dim_h)
        self.z_to_h = nn.Linear(dim_z, dim_h)
        self.mlp = MLP(2*dim_h, dim_h, dim_h, n_layers=num_layers, nonlinearity=nonlinearity)
        self.h_to_z = nn.Linear(dim_h, dim_z)

    def forward(self, w):
        """
        w: [..., length]
        returns: [..., dim_z]
        """
        batch_shape = w.shape[:-1]
        length = w.shape[-1]
        h_w = self.embedding(w)
        h = torch.zeros(*batch_shape, self.dim_h, device=h_w.device)
        for i in range(length):
            h_i = self.mlp(torch.cat([h, h_w[..., i, :]], dim=-1))
            mask = (w[..., i] > self.eos).unsqueeze(-1).float()
            if self.residual:
                h = h + h_i * mask
            else:
                h = h_i * mask + h * (1 - mask)
        z = self.h_to_z(h)
        return z
