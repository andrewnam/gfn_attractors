import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops
import math
from collections import OrderedDict
from functools import cached_property

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
           if tensor, assumes that x has shape [..., dim] and t has shape [...]
        """
        if t is None:
            batch_shape = x.shape[:-2]
            pe = tu.prepend_shape(self.pe[:x.shape[-2]], batch_shape)
        elif isinstance(t, int):
            batch_shape = x.shape[:-1]
            pe = tu.prepend_shape(self.pe[t], batch_shape)
        elif isinstance(t, torch.Tensor):
            if x.shape[:-1] != t.shape:
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



class FeaturePredictor(nn.Module):

    def __init__(self, dim_z, dim_h, num_layers, joint: bool, **features):
        """
        features: dict of feature names and
            0: continuous target variable; uses MSE loss
            1: binary target variable; uses BCE loss
            2+: multi-class target variable; uses CrossEntropy loss
        """
        super().__init__()
        self.dim_z = dim_z
        self.dim_h = dim_h
        self.num_layers = num_layers
        self.joint = joint
        self.features = OrderedDict(features)
        
        if joint:
            self.mlp = MLP(dim_z, self.dim_output, dim_h, num_layers, nonlinearity=nn.ReLU())
        else:
            self.mlp = nn.ModuleDict({k: MLP(dim_z, max(1, v), dim_h, num_layers, nonlinearity=nn.ReLU(), squeeze=False)
                                      for k, v in self.features.items()})
            
    @cached_property
    def dim_output(self):
        d = 0
        for v in self.features.values():
            d += max(1, v)
        return d
    
    def forward(self, z, mode: str, logits=False, squeeze=True):
        """
        mode: 'dict', 'tuple', 'tensor'
        """
        # unsqueeze = unsqueeze and as_dict
        if self.joint:
            _logits = self.mlp(z)
            if mode == 'tensor' and logits:
                return _logits
            _logits = _logits.split([max(v, 1) for v in self.features.values()], dim=-1)
            outputs = OrderedDict()
            for (k, v), o in zip(self.features.items(), _logits):
                if squeeze:
                    o = o.squeeze(-1)
                if v == 1 and not logits:
                    o = o.sigmoid()
                elif v > 1 and not logits:
                    o = o.softmax(dim=-1)
                outputs[k] = o
                
        else:
            outputs = OrderedDict()
            for k, v in self.features.items():
                o = self.mlp[k](z)
                if squeeze:
                    o = o.squeeze(-1)
                if v == 1 and not logits:
                    o = o.sigmoid()
                elif v > 1 and not logits:
                    o = o.softmax(dim=-1)
                outputs[k] = o
                
        if mode == 'dict':
            return outputs
        elif mode == 'tuple':
            return tuple(outputs.values())
        elif mode == 'tensor':
            return torch.cat(list(outputs.values()), dim=-1)
        else:
            raise Exception(f"Invalid mode: {mode}")
        
    def get_loss(self, z, targets, reduce=True):
        outputs = self(z, mode='dict', logits=True, squeeze=True)
        losses = {}
        metrics = {}
        
        for i, (k, v) in enumerate(self.features.items()):
            t = targets[..., i]
            o = outputs[k]
            if v == 0:
                loss = F.mse_loss(o, t.float(), reduction='none')
            elif v == 1:
                loss = F.binary_cross_entropy_with_logits(o, t.float(), reduction='none')
                metrics[f'acc_{k}'] = ((o > 0) == t).float().mean().item()
            else:
                loss = tu.batch_cross_entropy(o, t.long(), reduction='none')
                metrics[f'acc_{k}'] = (o.argmax(-1) == t).float().mean().item()
            losses[k] = loss
            metrics[f'loss_{k}'] = loss.mean().item()
        
        if reduce:
            losses = sum([v.mean() for v in losses.values()])
        return losses, metrics


# class FeaturePredictor(nn.Module):

#     def __init__(self, dim_z, dim_h, num_layers, **features):
#         """
#         features: dict of feature names and
#             0: continuous target variable; uses MSE loss
#             1: binary target variable; uses BCE loss
#             2+: multi-class target variable; uses CrossEntropy loss
#         """
#         super().__init__()
#         self.dim_z = dim_z
#         self.dim_h = dim_h
#         self.num_layers = num_layers
#         self.features = OrderedDict(features)
#         self.mlp = MLP(dim_z, self.dim_output, dim_h, num_layers, nonlinearity=nn.ReLU())

#         continuous_indices = []
#         binary_indices = []
#         categorical_indices = []
#         i = 0
#         for k, v in self.features.items():
#             if v == 0:
#                 continuous_indices.append(i)
#                 i += 1
#             elif v == 1:
#                 binary_indices.append(i)
#                 i += 1
#             else:
#                 categorical_indices.extend(range(i, i+v))
#         self.continuous_indices = tuple(continuous_indices)
#         self.binary_indices = tuple(binary_indices)
#         self.categorical_indices = tuple(categorical_indices)

#     @cached_property
#     def dim_output(self):
#         d = 0
#         for v in self.features.values():
#             if v == 0:
#                 d += 1
#             elif v > 0:
#                 d += v
#         return d
    
#     def forward(self, z, logits=False):
#         h = self.mlp(z)
#         logit = h.split([max(v, 1) for v in self.features.values()], dim=-1)
        
#         outputs = []
#         for k, v in zip(self.features, logit):
#             v = v.squeeze(-1)
#             d = self.features[k]
#             if d == 0 or logits:
#                 outputs.append(v)
#             elif d == 1:
#                 outputs.append(v.sigmoid())
#             else:
#                 outputs.append(v.softmax(dim=-1))
#         return tuple(outputs)
        

#     def get_loss(self, z, targets, reduce=True, as_dict=True):
#         outputs = self(z, logits=True)
#         if reduce:
#             loss = 0
#         else:
#             loss = {}
#         metrics = {}
        
#         continuous_features = [k for k, v in self.features.items() if v == 0]
#         binary_features = [k for k, v in self.features.items() if v == 1]
#         multiclass_features = [k for k, v in self.features.items() if v > 1]
#         if len([k for k, v in self.features.items() if v == 0]) > 0:
#             # features = {k}
#             continuous_outputs = torch.stack([outputs[i] for i, k in enumerate(self.features) if self.features[k] == 0], dim=0)
#             continuous_targets = torch.stack([targets[..., i] for i, k in enumerate(self.features) if self.features[k] == 0], dim=0)
#             mse_loss = F.mse_loss(continuous_outputs, continuous_targets, reduction='none').view(len(continuous_outputs), -1)#.mean(-1)
#             if reduce:
#                 loss += mse_loss.mean()
#             else:
#                 for i, k in enumerate(continuous_features):
#                     loss[k] = mse_loss[i]
#                     print(mse_loss[i])
#             mse_loss = mse_loss.mean(-1)
#             metrics.update({f'loss_{k}': mse_loss[i].item() for i, k in enumerate([f for f, v in self.features.items() if v == 0])})
#             metrics['mse_loss'] = mse_loss.mean().item()

#         if len([k for k, v in self.features.items() if v == 1]) > 0:
#             binary_outputs = torch.stack([outputs[i] for i, k in enumerate(self.features) if self.features[k] == 1], dim=0)
#             binary_targets = torch.stack([targets[..., i] for i, k in enumerate(self.features) if self.features[k] == 1], dim=0)
#             bce_loss = F.binary_cross_entropy_with_logits(binary_outputs, binary_targets, reduction='none').view(len(binary_outputs), -1)
#             if reduce:
#                 loss += bce_loss.mean()
#             else:
#                 for i, k in enumerate(binary_features):
#                     loss[k] = bce_loss[i]
#             bce_loss = bce_loss.mean(-1)
#             metrics.update({f'loss_{k}': bce_loss[i].item() for i, k in enumerate([f for f, v in self.features.items() if v == 1])})
#             accuracies = ((binary_outputs > 0) == binary_targets).float().mean((-1))
#             metrics.update({f'accuracy_{f}': a.item() for f, a in zip([f for f, v in self.features.items() if v == 1], accuracies)})
#             metrics['bce_loss'] = bce_loss.mean().item()

#         ce_loss = 0
#         for i, k in enumerate(self.features):
#             if self.features[k] > 1:
#                 f_loss = tu.batch_cross_entropy(outputs[i], targets[..., i].long(), reduction='none')
#                 if not reduce:
#                     loss[k] = f_loss#.unsqueeze(0)
#                 f_loss = f_loss.mean()
#                 ce_loss += f_loss
#                 accuracy = (outputs[i].argmax(-1) == targets[..., i].long()).float().mean()
#                 metrics[f'loss_{k}'] = f_loss.item()
#                 metrics[f'accuracy_{k}'] = accuracy.item()
#         metrics['ce_loss'] = ce_loss.item()

#         if not reduce:
#             if as_dict:
#                 loss = OrderedDict({k: loss[k] for k in self.features})
#             else:
#                 loss = torch.stack([loss[k] for k in self.features], dim=-1)
#         return loss, metrics
