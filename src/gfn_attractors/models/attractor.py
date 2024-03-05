import torch
from torch import nn

from .discretizer import DiscretizeModule
from .helpers import SafeEmbedding, MLP


class MLPAttractorModel(DiscretizeModule):

    def __init__(self, vocab_size, dim_z, max_sequence_length, joint_vocabulary, dim_h=128, num_layers=1, nonlinearity=nn.ReLU()):
        super().__init__(vocab_size, None)
        self.dim_z = dim_z
        self.dim_h = dim_h
        self.joint_vocabulary = joint_vocabulary
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        
        if joint_vocabulary:
            num_tokens = 2 + vocab_size
        else:
            num_tokens = 2 + max_sequence_length * vocab_size
        if num_layers == 0:
            self.embedding = SafeEmbedding(num_tokens, dim_z)
        else:
            self.embedding = SafeEmbedding(num_tokens + max_sequence_length * vocab_size, dim_h)
            self.mlp = MLP(dim_h, dim_z, dim_h, n_layers=num_layers, nonlinearity=nonlinearity)

    def forward(self, w):
        """
        w: [..., length]
        returns: [..., dim_z]
        """
        batch_shape = w.shape[:-1]
        length = w.shape[-1]
        w = w.reshape(-1, length)
        if not self.joint_vocabulary:
            position = self.vocab_size * torch.arange(length, device=w.device)
            mask = w > self.eos
            w = ~mask * w + mask * (position + w)

        z = self.embedding(w).sum(1)
        if self.num_layers > 0:
            z = self.mlp(z)
        return z.view(*batch_shape, self.dim_z)


class RNNAttractorModel(DiscretizeModule):

    def __init__(self, vocab_size, dim_z, num_layers=1, dim_h=128):
        super().__init__(vocab_size, None)
        self.dim_z = dim_z
        self.num_layers = num_layers
        self.dim_h = dim_h

        self.embedding = SafeEmbedding(self.num_tokens, self.dim_z)
        self.rnn = nn.RNN(self.dim_z, self.dim_z, num_layers=self.num_layers, nonlinearity='relu', batch_first=True)

    def forward(self, w):
        """
        w: tensor with shape (..., length)
        returns:
            h_w: tensor with shape (..., dim_z)
        """
        shape = w.shape[:-1]
        w = w.reshape(-1, w.shape[-1])
        h_w = self.embedding(w)
        h_w = h_w * (w > self.eos).unsqueeze(-1)
        h_w, _ = self.rnn(h_w)
        return h_w[:,-1].view(*shape, self.dim_z)


class GRUAttractorModel(DiscretizeModule):

    def __init__(self, vocab_size, dim_z, num_layers=1, dim_h=128):
        super().__init__(vocab_size, None)
        self.dim_z = dim_z
        self.num_layers = num_layers
        self.dim_h = dim_h

        self.embedding = SafeEmbedding(self.num_tokens, self.dim_h)
        self.gru = nn.GRU(self.dim_h, self.dim_h, num_layers=self.num_layers, batch_first=True)
        self.h_to_z = nn.Linear(self.dim_h, self.dim_z)

    def forward(self, w):
        """
        w: tensor with shape (..., length)
        returns:
            h_w: tensor with shape (..., dim_z)
        """
        shape = w.shape[:-1]
        w = w.reshape(-1, w.shape[-1])
        h_w = self.embedding(w)
        h_w = h_w * (w > self.eos).unsqueeze(-1)
        h, _ = self.gru(h_w)
        return self.h_to_z(h[:,-1]).view(*shape, self.dim_z)


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
