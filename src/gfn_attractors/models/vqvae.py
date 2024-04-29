from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from dataclasses import dataclass

from .discretizer import BoWDiscretizeModule
from .helpers import MLP
from .codebook import Codebook
from .images import TransformerImageEncoder, TransformerImageDecoder
from ..misc.config import Config


@dataclass
class ImageVQVAEConfig(Config):
    
    dim_z: int
    num_entries: int
    num_codebooks: int
    seed: int = 0
    
    patch_size: int = 8
    dim_h: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    nhead: int = 8
    dim_feedforward: int = 256


class ImageVQVAE(BoWDiscretizeModule):

    def __init__(self, config: ImageVQVAEConfig, data_module):
        super().__init__(vocab_size=config.num_entries * config.num_codebooks, group_size=config.num_codebooks)
        self.config = config
        self.data_module = data_module

        self.encoder = TransformerImageEncoder(data_module.size, 
                                               data_module.num_channels, 
                                               patch_size=config.patch_size, 
                                               dim_z=config.dim_h, 
                                               dim_h=config.dim_h,
                                               num_layers=config.num_encoder_layers,
                                               dim_feedforward=config.dim_feedforward,
                                               nhead=config.nhead)
        self.decoder = TransformerImageDecoder(data_module.size, 
                                               data_module.num_channels, 
                                               patch_size=config.patch_size, 
                                               dim_z=config.dim_z, 
                                               dim_h=config.dim_h,
                                               num_layers=config.num_decoder_layers,
                                               dim_feedforward=config.dim_feedforward,
                                               nhead=config.nhead)
        self.mlp = MLP(config.dim_h, config.num_codebooks * config.dim_h, config.dim_h, n_layers=3, nonlinearity=nn.ReLU())
        self.h_to_z = MLP(config.dim_h, config.dim_z, config.dim_h, n_layers=1, nonlinearity=nn.ReLU())
        self.codebook = Codebook(config.dim_h, num_entries=config.num_entries, num_codebooks=config.num_codebooks, beta=0.25)

    @property
    def max_w_length(self):
        return self.config.num_codebooks
    
    @property
    def device(self):
        return self.codebook.device

    def forward(self, x, return_losses=False):
        h = self.encoder(x)
        h = torch.stack(self.mlp(h).chunk(self.config.num_codebooks, dim=-1), dim=-2)
        h_q, indices, dictionary_loss, commitment_loss = self.codebook(h, return_losses=True)
        h_q = h_q.sum(-2)
        z = self.h_to_z(h_q)

        if return_losses:
            return z, indices, dictionary_loss, commitment_loss
        return z, indices
    
    def decode(self, z):
        return self.decoder(z.unsqueeze(1))
    
    def get_loss(self, x):
        h, w, dictionary_loss, commitment_loss = self(x, return_losses=True)
        logits = self.decode(h)
        recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='none').sum((-3, -2, -1)).mean()
        loss = recon_loss + dictionary_loss + commitment_loss
        loss = loss.mean()
        metrics = {'loss': loss.item(), 
                   'reconstruction_loss': recon_loss.item(), 
                   'dictionary_loss': dictionary_loss.item(), 
                   'commitment_loss': commitment_loss.item()}
        return loss, metrics
    
    def get_w(self, x, **kwargs):
        z, w = self(x)
        return F.one_hot(w, self.config.num_entries).flatten(-2, -1)


# class BoWVQVAE(BoWDiscretizeModule):
#     """
#     A simple module that takes an input tensor and returns a sequence of bits.
#     Useful for sampling graphs, etc.

#     group_size: if not None, the bits are grouped into groups of size group_size. Only one bit in each group can be active.
#     """

#     def __init__(self, num_bits, dim_z, num_steps, dim_input=None, group_size=None, beta=.25, allow_null=False, 
#                  dim_h=256, num_layers=3, nonlinearity=nn.ReLU()):
#         super().__init__(vocab_size=num_bits, group_size=group_size)
#         if group_size is not None and num_bits % group_size != 0:
#             raise ValueError('num_bits must be divisible by group_size.')

#         self.dim_input = dim_input
#         self.dim_z = dim_z
#         self.num_steps = num_steps
#         self.beta = beta
#         self.num_bits = num_bits
#         self.dim_h = dim_h
#         self.num_layers = num_layers
#         self.nonlinearity = nonlinearity
#         self.allow_null = allow_null
        
#         self.encoder = self.init_encoder()
#         self.mlp = MLP(dim_z + dim_h, dim_h, dim_h, n_layers=num_layers, nonlinearity=nonlinearity)
#         self.codebook = nn.Parameter(torch.randn(num_bits, dim_h))
#         self.h_to_z = nn.Linear(dim_h, dim_z)
#         self.decoder = self.init_decoder()

#     @property
#     def device(self):
#         return self.codebook.device

#     def init_encoder(self):
#         """
#         Should take a tensor with shape [batch_size, dim_input]
#         and return a tensor with shape [batch_size, dim_h]
#         """
#         return nn.Linear(self.dim_input, self.dim_h)

#     @abstractmethod
#     def init_decoder(self):
#         raise NotImplementedError

#     @abstractmethod
#     def get_recon_loss(self, x, z):
#         """
#         x: tensor with shape [batch_size, *input_shape]
#         z: tensor with shape [batch_size, dim_z]
#         returns: scalar and metrics dict
#         """
#         raise NotImplementedError
    
#     def encode(self, x, num_steps=None, return_losses=False):
#         """
#         x: (batch_size, dim_input)
#         """
#         if num_steps is None:
#             num_steps = self.num_steps
#         batch_size = len(x)
#         x = self.encoder(x)
#         w = torch.zeros(batch_size, self.num_bits, device=x.device)
#         h_w = torch.zeros(batch_size, self.dim_h, device=x.device)
#         dists = torch.zeros(batch_size, num_steps, device=x.device)

#         dictionary_loss = 0
#         commitment_loss = 0

#         for i in range(num_steps):
#             if self.group_size is not None:
#                 invalid = (w.view(len(w), -1, self.group_size) > 0).any(-1)
#                 invalid = einops.repeat(invalid, 'b k -> b (k g)', g=self.group_size)
#             else:
#                 invalid = w.bool()

#             h = self.mlp(torch.cat([x, h_w], dim=-1))
#             h = F.tanh(h) # the latent space explodes without this
#             dist = (h.unsqueeze(1) - self.codebook.unsqueeze(0)).pow(2).sum(-1)
#             dist_i, w_i = dist.masked_fill(invalid, float('inf')).min(-1)
#             hq = h + (self.codebook[w_i] - h).detach()
#             h_w = h_w + hq

#             w = w.scatter(1, w_i.unsqueeze(1), 1)
#             dists[:, i] = dist_i
#             dictionary_loss += F.mse_loss(h.detach(), hq)#, reduction='none').sum((-1)).mean()
#             commitment_loss += F.mse_loss(hq, h.detach())#, reduction='none').sum((-1)).mean()
#         z = self.h_to_z(h_w)

#         if return_losses:
#             return w, z, dists, dictionary_loss, commitment_loss
#         return w, z, dists

#     def get_loss(self, x, num_steps=None):
#         if num_steps is None:
#             num_steps = self.num_steps
#         w, z, dists, dictionary_loss, commitment_loss = self.encode(x, num_steps, return_losses=True)
#         recon_loss, metrics = self.get_recon_loss(x, z)
        
#         loss = recon_loss + dictionary_loss + self.beta * commitment_loss
        
#         metrics.update({
#             'recon_loss': recon_loss.item(),
#             'dictionary_loss': dictionary_loss.item(),
#             'vq_distance': dists.mean().item(),
#             'codebook': self.codebook.norm(dim=-1).max().item(),
#             'z': z.norm(dim=-1).max().item(),
#             'loss': loss.item()
#         })
#         return loss, metrics
    

# class ImageVQVAE(BoWVQVAE):

#     def __init__(self, image_size, num_channels, patch_size, num_steps,
#                  nhead=8, dim_feedforward=256, num_encoder_layers=3, num_decoder_layers=1, 
#                  *args, **kwargs):
#         self.image_size = image_size
#         self.num_channels = num_channels
#         self.patch_size = patch_size

#         self.nhead = nhead
#         self.dim_feedforward = dim_feedforward
#         self.num_encoder_layers = num_encoder_layers
#         self.num_decoder_layers = num_decoder_layers
#         super().__init__(*args, num_steps=num_steps, dim_input=(num_channels, image_size, image_size), **kwargs)
        
#     def init_encoder(self):
#         return TransformerImageEncoder(size=self.image_size,
#                                        num_channels=self.num_channels,
#                                        patch_size=self.patch_size,
#                                        dim_z=self.dim_z,
#                                        dim_h=self.dim_h,
#                                        nhead=self.nhead,
#                                        dim_feedforward=self.dim_feedforward,
#                                        num_layers=self.num_encoder_layers,
#                                        dropout=0.)

#     def init_decoder(self):
#         return TransformerImageDecoder(size=self.image_size,
#                                        num_channels=self.num_channels,
#                                        patch_size=self.patch_size,
#                                        dim_z=self.dim_z,
#                                        nhead=self.nhead,
#                                        dim_feedforward=self.dim_feedforward,
#                                        num_layers=self.num_decoder_layers,
#                                        dropout=0.)

#     def get_recon_loss(self, x, z):
#         logits = self.decoder(z)
#         loss = F.binary_cross_entropy_with_logits(logits, x, reduction='none').sum((-3, -2, -1)).mean()
#         return loss, {'recon_loss': loss.item()}
