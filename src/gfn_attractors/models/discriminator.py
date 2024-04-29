import torch
from torch import nn
import einops
from .helpers import MLP
from ..misc import torch_utils as tu



class Discriminator(nn.Module):
    
    def __init__(self, dim_q, num_negatives, dim_k=None, dim_h=256, num_layers=3):
        super().__init__()
        self.dim_q = dim_q
        self.num_negatives = num_negatives
        self.dim_k = dim_q if dim_k is None else dim_k
        self.dim_h = dim_h
        self.num_layers = num_layers
        self.mlp = MLP(self.dim_q + self.dim_k, 1, dim_h, num_layers, nn.ReLU())
        
    def forward(self, *args, **kwargs):
        return self.log_prob(*args, **kwargs)
        
    def get_logits(self, q, k):
        """
        q: (batch_size, dim_q)
        k: (batch_size, n, dim_k)
        returns: (batch_size, n)
        """
        q = einops.repeat(q, 'b q -> b n q', n=k.shape[1])
        return self.mlp(torch.cat([q, k], dim=-1))
    
    def get_accuracy(self, q, k):
        """
        q: (batch_size, dim_q)
        k: (batch_size, dim_k)
        """
        log_prob, logits = self.log_prob(q, k, return_logits=True)
        accuracy = (logits.argmax(-1) == 0).float().mean()
        return accuracy
    
    def log_prob(self, q, k, return_logits=False):
        """
        q: (batch_size, dim_q)
        k: (batch_size, dim_k)
        returns: (batch_size)
            If return_logits, also returns (batch_size, 1 + num_negatives)
        """
        assert q.ndim == 2 and k.ndim == 2
        batch_size = len(q)
        
        if batch_size <= self.num_negatives:
            indices = torch.arange(batch_size, device=q.device)
            indices = einops.repeat(indices, 'k -> b k', b=batch_size)
            i = torch.eye(batch_size, dtype=bool, device=q.device)
            indices = indices[~i].view(batch_size, batch_size-1)
            negatives = k[indices] # (batch_size, batch_size-1, dim_k)
        else:
            i = torch.eye(batch_size, device=q.device)
            indices = torch.multinomial(1-i, self.num_negatives, replacement=False)
            negatives = k[indices] # (batch_size, num_negatives, dim_k)
        
        support = torch.cat([k.unsqueeze(1), negatives], dim=1)
        logits = self.get_logits(q, support) # (batch_size, 1+num_negatives)
        targets = torch.zeros(logits.shape[:-1], dtype=torch.long, device=logits.device)
        log_prob = -tu.batch_cross_entropy(logits, targets, reduction='none')
        if return_logits:
            return log_prob, logits
        return log_prob


# class Discriminator(nn.Module):
    
#     def __init__(self, dim_q, dim_k=None, dim_h=256, num_layers=3):
#         super().__init__()
#         self.dim_q = dim_q
#         self.dim_k = dim_q if dim_k is None else dim_k
#         self.dim_h = dim_h
#         self.num_layers = num_layers
#         self.mlp = MLP(self.dim_q + self.dim_k, 1, dim_h, num_layers, nn.ReLU())
        
#     def forward(self, *args, **kwargs):
#         return self.log_prob(*args, **kwargs)
        
#     def get_logits(self, q, k):
#         """
#         q: (batch_size, dim_q) or (batch_size, m, dim_q)
#         k: (batch_size, n, dim_k)
#         returns: (batch_size, n) or (batch_size, m, n)
#         """
#         if q.ndim == 2:
#             q = einops.repeat(q, 'b q -> b n q', n=k.shape[1])
#         else:
#             q = einops.repeat(q, 'b m q -> b m n q', n=k.shape[1])
#             k = einops.repeat(k, 'b n k -> b m n k', m=q.shape[1])
#         return self.mlp(torch.cat([q, k], dim=-1))
    
#     def get_accuracy(self, q, k):
#         """
#         q: (batch_size, dim_q)
#         k: (batch_size, dim_k)
#         """
#         log_prob, logits = self.log_prob(q, k, return_logits=True)
#         accuracy = (logits.argmax(-1) == 0).float().mean()
#         return accuracy
    
#     def log_prob(self, q, k, return_logits=False):
#         """
#         q: (batch_size, dim_q) or (batch_size, m, dim_q)
#         k: (batch_size, dim_k)
#         returns: (batch_size) or (batch_size, m)
#             If return_logits, also returns (batch_size, 1 + num_k) or (batch_size, m, 1 + num_k)
#         """
#         batch_size = len(q)
        
#         indices = torch.arange(batch_size, device=q.device)
#         indices = einops.repeat(indices, 'k -> b k', b=batch_size)
#         i = torch.eye(batch_size, dtype=bool, device=q.device)
#         indices = indices[~i].view(batch_size, batch_size-1)
#         negatives = k[indices] # (batch_size, batch_size-1, dim_k)
        
#         support = torch.cat([k.unsqueeze(1), negatives], dim=1)
#         logits = self.get_logits(q, support) # (batch_size, batch_size) or (batch_size, m, batch_size)
#         targets = torch.zeros(logits.shape[:-1], dtype=torch.long, device=logits.device)
#         log_prob = -tu.batch_cross_entropy(logits, targets, reduction='none')
#         if return_logits:
#             return log_prob, logits
#         return log_prob
