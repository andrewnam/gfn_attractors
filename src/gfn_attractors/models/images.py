import torch
import torch.nn as nn
from torch.distributions import Normal
import einops

from .helpers import PositionalEncoding2D, MLP


class TransformerImageEncoder(nn.Module):
    """
    Note: when using variational mode, standard deviation is between 0 and 1 for stability.
    """

    def __init__(self, 
                 size, 
                 num_channels, 
                 patch_size,
                 dim_z,
                 variational=False, 
                 mu_dependent_sd=False,
                 sd_model_num_layers=2,
                 max_sd=1.,
                 num_encodings=None,
                 independent_encodings=False,
                 dim_h=None,
                 nhead=8, 
                 dim_feedforward=512, 
                 num_layers=3, 
                 dropout=0.):
        assert size % patch_size == 0
        super().__init__()
        dim_h = dim_z if dim_h is None else dim_h

        self.size = size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.dim_z = dim_z
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.num_encodings = num_encodings
        self.variational = variational
        self.independent_encodings = independent_encodings
        self.dim_h = dim_h
        self.mu_dependent_sd = mu_dependent_sd
        self.sd_model_num_layers = sd_model_num_layers
        self.max_sd = max_sd

        if dim_h != dim_z:
            self.h_to_z = nn.Linear(dim_h, (1 + variational - mu_dependent_sd) * dim_z)
            self.z_to_h = nn.Linear(dim_z, dim_h)
        else:
            if variational:
                self.h_to_z = nn.Linear(dim_h, (2 - mu_dependent_sd) * dim_z)
            else:
                self.h_to_z = nn.Identity()
            self.z_to_h = nn.Identity()

        if variational and mu_dependent_sd:
            self.sd_model = nn.Sequential(MLP(dim_z, dim_z, dim_feedforward, n_layers=sd_model_num_layers), nn.Sigmoid())

        self.patch_linear = nn.Linear(num_channels * patch_size**2, dim_h)
        self.positional_encoding = PositionalEncoding2D(dim_h, size // patch_size, size // patch_size, concat=False)
        self.transformer = nn.Transformer(dim_h, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=1,
                                          dim_feedforward=dim_feedforward, dropout=dropout, activation=nn.SiLU(),
                                          norm_first=True, batch_first=True)
        self.query = nn.Parameter(torch.randn(1 if num_encodings is None else num_encodings, dim_h))
        self.condition_flag = nn.Parameter(torch.rand(dim_h))

        if num_encodings is None:
            self.mask = None
        else:
            self.register_buffer('mask', ~torch.eye(num_encodings, dtype=bool))

    @property
    def num_patches(self):
        return (self.size // self.patch_size)**2
    
    def get_sd(self, mu):
        if self.mu_dependent_sd:
            return self.max_sd * self.sd_model(mu)
        else:
            raise ValueError('Standard deviation is not dependent on z0.')
    
    def forward(self, 
                x: torch.Tensor, 
                c: torch.Tensor = None,
                return_params: bool = False):
        """
        x: image tensor with shape [batch_size, num_channels, height, width]
        c: condition tensor with shape [batch_size, dim_h] or [batch_size, k, dim_h]
        c_key_padding_mask: mask for condition tensor with shape [batch_size] or [batch_size, k]
            True wherever the corresponding element should be ignored
        
        returns: tensor with shape [batch_size, num_encodings, dim_z]
        """
        batch_size = len(x)
        n_patches = self.size // self.patch_size
        h_x = x.permute(0, 2, 3, 1).reshape(batch_size, n_patches, n_patches, -1)
        h_x = self.patch_linear(h_x)
        h = self.positional_encoding(h_x).flatten(1, 2)
        if c is not None:
            h_c = c + self.condition_flag
            if h_c.ndim == 2:
                h_c = h_c.unsqueeze(1)
            h = torch.cat([h, h_c], dim=1)
        query = einops.repeat(self.query, 'k h -> b k h', b=batch_size)

        mask = self.mask if self.independent_encodings else None
        h = self.transformer(h, query, tgt_mask=mask)
        if self.num_encodings is None:
            h = h.squeeze(1)

        if self.variational:
            if self.mu_dependent_sd:
                mu = self.h_to_z(h)
                sigma = self.get_sd(mu)
            else:
                mu, logsigma = self.h_to_z(h).chunk(2, dim=-1)
                sigma = self.max_sd * logsigma.sigmoid()
            z = Normal(mu, sigma).rsample()
            if return_params:
                return z, mu, sigma
            else:
                return z
            
        z = self.h_to_z(h)
        return z


class TransformerImageDecoder(nn.Module):

    def __init__(self, size, num_channels, patch_size, dim_z, dim_h=256, nhead=8, dim_feedforward=512,
                 num_layers=3, dropout=0.):
        assert size % patch_size == 0
        super().__init__()
        self.size = size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.dim_z = dim_z
        self.dim_h = dim_h
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers

        if dim_z != dim_h:
            self.z_to_h = nn.Linear(dim_z, dim_h)
        else:
            self.z_to_h = nn.Identity()

        decoder_layers = nn.TransformerDecoderLayer(dim_h, nhead, dim_feedforward, activation=nn.SiLU(), 
                                                    batch_first=True, norm_first=True, dropout=dropout)
        self.transformer = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        self.patch_linear = nn.Linear(dim_h, num_channels * patch_size**2)
        self.positional_encoding = PositionalEncoding2D(dim_h, size // patch_size, size // patch_size, concat=False)

    @property
    def num_patches(self):
        return (self.size // self.patch_size)**2
    
    def forward(self, z: torch.Tensor):
        """
        z: tensor with shape [batch_size, dim_z] or [batch_size, length, dim_z]
        """
        if z.ndim == 2:
            z = z.unsqueeze(1)
        h = self.z_to_h(z)
        batch_size = len(h)
        n_patches = self.size // self.patch_size 
        h_pos = torch.zeros(batch_size, n_patches, n_patches, self.dim_h, device=h.device)
        h_pos = self.positional_encoding(h_pos).flatten(1, 2)
        h = self.transformer(h_pos, memory=h)
        logits = self.patch_linear(h)
        logits = logits.view(batch_size, self.num_patches, self.num_channels, self.patch_size, self.patch_size)
        logits = logits.permute(0, 2, 1, 3, 4).reshape(batch_size, self.num_channels, self.size, self.size)
        return logits


# class TransformerImageEncoder(nn.Module):
#     """
#     Note: when using variational mode, standard deviation is between 0 and 1 for stability.
#     """

#     def __init__(self, 
#                  size, 
#                  num_channels, 
#                  patch_size,
#                  variational=False, 
#                  additive_conditioning=False,
#                  num_encodings=None,
#                  independent_encodings=False,
#                  dim_encoding=256, 
#                  dim_h=None,
#                  dim_conditioning=None,
#                  nhead=8, 
#                  dim_feedforward=512, 
#                  num_layers=3, 
#                  dropout=0.):
#         assert size % patch_size == 0
#         super().__init__()
#         dim_h = dim_encoding if dim_h is None else dim_h
#         dim_conditioning = dim_encoding if dim_conditioning is None else dim_conditioning

#         self.size = size
#         self.patch_size = patch_size
#         self.num_channels = num_channels
#         self.dim_encoding = dim_encoding
#         self.dim_conditioning = dim_conditioning
#         self.nhead = nhead
#         self.dim_feedforward = dim_feedforward
#         self.num_layers = num_layers
#         self.num_encodings = num_encodings
#         self.variational = variational
#         self.independent_encodings = independent_encodings
#         self.additive_conditioning = additive_conditioning
#         self.dim_h = dim_h
#         self.dim_conditioning = dim_conditioning


#         if dim_h != dim_encoding:
#             self.h_to_e = nn.Linear(dim_h, (1 + variational) * dim_encoding)
#             self.e_to_h = nn.Linear(dim_conditioning, dim_h)
#         else:
#             if variational:
#                 self.h_to_e = nn.Linear(dim_h, 2 * dim_encoding)
#             else:
#                 self.h_to_e = nn.Identity()
#             self.e_to_h = nn.Identity()

#         self.c_to_h = nn.Identity() if dim_conditioning == dim_h else nn.Linear(dim_conditioning, dim_h)

#         self.patch_linear = nn.Linear(num_channels * patch_size**2, dim_h)
#         self.positional_encoding = PositionalEncoding2D(dim_h, size // patch_size, size // patch_size, concat=False)
#         self.transformer = nn.Transformer(dim_h, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=1,
#                                           dim_feedforward=dim_feedforward, dropout=dropout, activation=nn.SiLU(),
#                                           norm_first=True, batch_first=True)
#         self.query = nn.Parameter(torch.randn(1 if num_encodings is None else num_encodings, dim_h))
#         self.condition_flag = nn.Parameter(torch.rand(dim_h))

#         if additive_conditioning:
#             if self.num_encodings is not None:
#                 raise ValueError('Additive conditioning is only supported for a single encoding.')
#             self.c_to_e = nn.Identity() if dim_conditioning == dim_encoding else nn.Linear(dim_conditioning, dim_encoding)
#             self.gate = nn.Sequential(MLP(2*dim_encoding, dim_encoding, dim_encoding, n_layers=2), 
#                                       nn.Sigmoid())

#         if num_encodings is None:
#             self.mask = None
#         else:
#             self.register_buffer('mask', ~torch.eye(num_encodings, dtype=bool))

#     @property
#     def num_patches(self):
#         return (self.size // self.patch_size)**2
    
#     def additive_condition(self, encoding, c):
#         if c is None:
#             return encoding
#         gate = self.gate(torch.cat([encoding, c], dim=-1))
#         c = self.c_to_e(c)
#         return c + gate * encoding
    
#     def forward(self, 
#                 x: torch.Tensor, 
#                 c: torch.Tensor = None,
#                 use_reparam: bool = None,
#                 return_params: bool = False):
#         """
#         x: image tensor with shape [batch_size, num_channels, height, width]
#         c: condition tensor with shape [batch_size, dim_encoding] or [batch_size, k, dim_encoding]
#         c_key_padding_mask: mask for condition tensor with shape [batch_size] or [batch_size, k]
#             True wherever the corresponding element should be ignored
        
#         returns: tensor with shape [batch_size, num_encodings, dim_h]
#         """
#         if self.variational and use_reparam is None:
#             raise ValueError('Must specify whether to use reparametrization or not.')
        
#         batch_size = len(x)
#         n_patches = self.size // self.patch_size
#         h_x = x.permute(0, 2, 3, 1).reshape(batch_size, n_patches, n_patches, -1)
#         h_x = self.patch_linear(h_x)
#         h = self.positional_encoding(h_x).flatten(1, 2)
#         if c is not None:
#             h_c = self.c_to_h(c) + self.condition_flag
#             if h_c.ndim == 2:
#                 h_c = h_c.unsqueeze(1)
#             h = torch.cat([h, h_c], dim=1)
#         query = einops.repeat(self.query, 'k h -> b k h', b=batch_size)

#         mask = self.mask if self.independent_encodings else None
#         h = self.transformer(h, query, tgt_mask=mask)
#         if self.num_encodings is None:
#             h = h.squeeze(1)

#         if self.variational:
#             mu, logsigma = self.h_to_e(h).chunk(2, dim=-1)
#             sigma = logsigma.sigmoid()
#             if use_reparam:
#                 encoding = Normal(mu, sigma).rsample()
#             else:
#                 encoding = Normal(mu, sigma).sample()
#             if self.additive_conditioning:
#                 encoding = self.additive_condition(encoding, c)
#             if return_params:
#                 return encoding, mu, sigma
#             else:
#                 return encoding
            
#         encoding = self.h_to_e(h)
#         if self.additive_conditioning:
#             encoding = self.additive_condition(encoding, c)
#         return encoding


# class TransformerImageDecoder(nn.Module):

#     def __init__(self, size, num_channels, patch_size, dim_encoding, dim_h=256, nhead=8, dim_feedforward=512,
#                  num_layers=3, dropout=0.):
#         assert size % patch_size == 0
#         super().__init__()
#         self.size = size
#         self.patch_size = patch_size
#         self.num_channels = num_channels
#         self.dim_encoding = dim_encoding
#         self.dim_h = dim_h
#         self.nhead = nhead
#         self.dim_feedforward = dim_feedforward
#         self.num_layers = num_layers

#         if dim_encoding != dim_h:
#             self.e_to_h = nn.Linear(dim_encoding, dim_h)
#         else:
#             self.e_to_h = nn.Identity()

#         decoder_layers = nn.TransformerDecoderLayer(dim_h, nhead, dim_feedforward, activation=nn.SiLU(), 
#                                                     batch_first=True, norm_first=True, dropout=dropout)
#         self.transformer = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
#         self.patch_linear = nn.Linear(dim_h, num_channels * patch_size**2)
#         self.positional_encoding = PositionalEncoding2D(dim_h, size // patch_size, size // patch_size, concat=False)

#     @property
#     def num_patches(self):
#         return (self.size // self.patch_size)**2
    
#     def forward(self, encoding: torch.Tensor):
#         """
#         encoding: tensor with shape [batch_size, dim_encoding] or [batch_size, length, dim_encoding]
#         """
#         if encoding.ndim == 2:
#             encoding = encoding.unsqueeze(1)
#         h = self.e_to_h(encoding)
#         batch_size = len(h)
#         n_patches = self.size // self.patch_size 
#         h_pos = torch.zeros(batch_size, n_patches, n_patches, self.dim_h, device=h.device)
#         h_pos = self.positional_encoding(h_pos).flatten(1, 2)
#         h = self.transformer(h_pos, memory=h)
#         logits = self.patch_linear(h)
#         logits = logits.view(batch_size, self.num_patches, self.num_channels, self.patch_size, self.patch_size)
#         logits = logits.permute(0, 2, 1, 3, 4).reshape(batch_size, self.num_channels, self.size, self.size)
#         return logits