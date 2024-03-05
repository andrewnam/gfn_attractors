import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .helpers import MLP


class Codebook(nn.Module):

    def __init__(self, dim, num_entries, num_codebooks, dim_entries=None, proj_num_layers=None, proj_dim_h=None, beta=.25):
        super().__init__()
        self.dim = dim
        self.num_entries = num_entries
        self.num_codebooks = num_codebooks
        self.beta = beta
        self.dim_entries = dim_entries if dim_entries is not None else dim
        self.proj_dim_h = proj_dim_h if proj_dim_h is not None else max(dim, self.dim_entries)

        if self.dim != self.proj_dim_h or proj_num_layers is not None:
            if proj_num_layers is None:
                self.proj_num_layers = 1
            self.proj_in = MLP(self.dim, self.dim_entries, self.proj_dim_h, self.proj_num_layers, nonlinearity=nn.ReLU())
        else:
            self.proj_in = nn.Identity()

        if self.dim_entries != self.dim:
            self.proj_out = nn.Parameter(torch.randn(self.num_codebooks, self.dim_entries, self.dim))
        else:
            self.proj_out = None

        self.entries = nn.Parameter(torch.randn(num_codebooks, num_entries, self.dim_entries))

    def forward(self, x, sample=False, use_ag_scale_probs=False, return_losses=False, quantize=True):
        """
        x: (batch, num_codebooks, dim)
        returns: (batch, num_codebooks, dim)
        """
        batch_size = x.shape[0]
        x = self.proj_in(x).view(batch_size, self.num_codebooks, self.dim_entries)
        x = x.transpose(0, 1)
        dist = torch.cdist(x, self.entries) # (num_codebooks, batch, num_entries)
        if sample:
            probs = 1 / dist.transpose(0, 1) # (batch, num_codebooks, num_entries)
            probs = probs / probs.sum(-1, keepdim=True)
            if use_ag_scale_probs:
                scale = get_ag_mean_scaling_factor(probs)
                probs = probs * scale.view(-1, 1, 1)
            idx = torch.distributions.Categorical(probs).sample()
            idx = idx.transpose(0, 1)
        else:
            idx = torch.argmin(dist, dim=-1)
        quantized = self.entries.gather(1, einops.repeat(idx, 'k b -> k b d', d=self.dim_entries))
        x = x.transpose(0, 1)
        quantized = quantized.transpose(0, 1)
        idx = idx.transpose(0, 1)

        if quantize:
            x_q = x + (quantized - x).detach()
        else:
            x_q = x
        if self.proj_out is not None:
            x_q = x_q.unsqueeze(-1) * self.proj_out.unsqueeze(0)
            x_q = x_q.sum(-2)

        if return_losses:
            dictionary_loss = F.mse_loss(x.detach(), quantized, reduction='none').sum((-1)).mean()
            commitment_loss = self.beta * F.mse_loss(x, quantized.detach(), reduction='none').sum((-1)).mean()
            return x_q, idx, dictionary_loss, commitment_loss
        return x_q, idx

    def get_entries(self, indices):
        """
        indices: (batch, num_codebooks)
        """

        quantized = self.entries.gather(1, einops.repeat(indices, 'b k -> k b d', d=self.dim))
        return quantized.transpose(0, 1)
    

def get_ag_mean_scaling_factor(probs, use_logsummean=True):
    """
    Given a tensor of k probabilities over n items each, find a scaling factor such that
    the geometric mean of all n^k combinations of the probabilities is  is equal to the
    arithmetic mean of all n^k combinations.
    This is useful when sampling from the Codebook a token sequence to compute the reward for a GFN, since it allows
    the GFN to learn the arithmetic mean of the rewards.
    Since the Codebook entries are sampled independently, this can run in O(nk) time instead of O(n^k), where n is the number of entries
    and k is the number of codebooks.

    Probs: tensor with shape [..., k, n]
    Returns: [...]
    """
    n = probs.shape[-1]
    a = probs.prod(-1).float().mean(-1)
    if use_logsummean:
        g = probs.log().sum(-1).mean(-1).exp()
    else:
        g = probs.pow(1/n).prod((-2, -1))
    return a / g
    


# class Codebook(nn.Module):

#     def __init__(self, dim, num_entries, num_codebooks, beta=.25):
#         super().__init__()
#         self.dim = dim
#         self.num_entries = num_entries
#         self.num_codebooks = num_codebooks
#         self.beta = beta

#         self.entries = nn.Parameter(torch.randn(num_codebooks, num_entries, dim))

#     def forward(self, x, return_losses=False):
#         """
#         x: (batch, num_codebooks, dim)
#         """
#         x = x.transpose(0, 1)
#         dist = torch.cdist(x, self.entries)
#         idx = torch.argmin(dist, dim=-1)
#         quantized = self.entries.gather(1, einops.repeat(idx, 'k b -> k b d', d=self.dim))
#         x = x.transpose(0, 1)
#         quantized = quantized.transpose(0, 1)
#         idx = idx.transpose(0, 1)

#         x_q = x + (quantized - x).detach()

#         if return_losses:
#             dictionary_loss = F.mse_loss(x.detach(), quantized, reduction='none').sum((-1)).mean()
#             commitment_loss = self.beta * F.mse_loss(x, quantized.detach(), reduction='none').sum((-1)).mean()
#             return x_q, idx, dictionary_loss, commitment_loss
#         return x_q, idx

#     def get_entries(self, indices):
#         """
#         indices: (batch, num_codebooks)
#         """

#         quantized = self.entries.gather(1, einops.repeat(indices, 'b k -> k b d', d=self.dim))
#         return quantized.transpose(0, 1)
    