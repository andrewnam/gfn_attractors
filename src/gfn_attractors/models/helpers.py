import torch
from torch import nn, Tensor
import einops
import math


from ..misc import torch_utils as tu


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 96,
        n_layers: int = 3,
        nonlinearity: nn.Module = nn.ELU(),
        squeeze: bool = True,
    ) -> None:
        """
        squeeze: if True, squeeze the last dimension of the output
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.squeeze = squeeze

        layers = []
        dim_i = input_dim
        dim_ip1 = hidden_dim if n_layers > 1 else output_dim
        for i in range(n_layers):
            layers.append(nn.Linear(dim_i, dim_ip1))
            if i < n_layers - 1:
                layers.append(nonlinearity)
            dim_i = dim_ip1
            dim_ip1 = output_dim if i == n_layers - 2 else hidden_dim
        self.layers = nn.Sequential(*layers)

    @property
    def device(self):
        return self.layers[0].weight.device

    def forward(self, x: Tensor) -> Tensor:
        y = self.layers(x)
        if self.squeeze:
            y = y.squeeze(-1)
        return y


class SafeEmbedding(nn.Embedding):

    def forward(self, x):
        if (x < 0).any():
            raise ValueError(f'x contains negative indices: {x}')
        if (x >= self.num_embeddings).any():
            raise ValueError(f'x contains indices >= num_embeddings: {x}')
        return super().forward(x)


class PositionalEncoding(nn.Module):
    
    def __init__(self, dim: int, max_len: int = 100, concat: bool = True):
        super(PositionalEncoding, self).__init__()

        self.concat = concat
        self.dim = dim

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, t: int | torch.Tensor | None = None):
        """
        x: tensor with shape [..., dim]
        t: if int, applies the same positional encoding to all elements in the batch
           if None, applies the positional encoding to the second dimension of x
           if tensor, assumes that x has shape [..., num_timesteps, dim] and t has shape [..., num_timesteps]
        """
        if t is None:
            batch_shape = x.shape[:-2]
            pe = tu.prepend_shape(self.pe[:x.shape[-2]], batch_shape)
        elif isinstance(t, int):
            batch_shape = x.shape[:-1]
            pe = tu.prepend_shape(self.pe[t], batch_shape)
        elif isinstance(t, torch.Tensor):
            if x.shape[:-2] != t.shape[:-1]:
                raise ValueError(f"x and t must have the same shape, got {x.shape[:-2]} and {t.shape[:-1]}")
            pe = self.pe[t]
        else:
            raise ValueError(f"t must be None, int, or tensor, got {type(t)}")
        
        if self.concat:
            return torch.cat([x, pe], dim=-1)
        else:
            return x + pe


class PositionalEncoding2D(nn.Module):
    
    def __init__(self, dim: int, nrow: int, ncol: int, concat: bool = True):
        super().__init__()
        assert dim % 4 == 0 # 'dim must be multiple of 4'

        self.concat = concat
        self.dim = dim
        self.nrow = nrow
        self.ncol = ncol

        # Compute the positional encodings once in log space.
        max_len = max(nrow, ncol)
        halfdim = dim // 2
        pe = torch.zeros(max_len, halfdim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, halfdim, 2).float() * -(math.log(max_len**2) / halfdim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: tensor with shape [..., nrow, ncol, dim]
        t: if int, applies the same positional encoding to all elements in the batch
           if None, applies the positional encoding to the second dimension of x
           if tensor, assumes that x has shape [..., num_timesteps, dim] and t has shape [..., num_timesteps]
        """
        batch_shape = x.shape[:-3]
        x = x.view(-1, *x.shape[-3:])
        batch_size, nrow, ncol, dim = x.shape
        
        pe_rows = einops.repeat(self.pe[:nrow], 'r d -> b r c d', b=batch_size, c=ncol)
        pe_cols = einops.repeat(self.pe[:ncol], 'c d -> b r c d', b=batch_size, r=nrow)
        pe = torch.cat([pe_rows, pe_cols], dim=-1)
        
        if self.concat:
            x = torch.cat([x, pe], dim=-1)
        else:
            x = x + pe
        return x.view(*batch_shape, nrow, ncol, -1)
