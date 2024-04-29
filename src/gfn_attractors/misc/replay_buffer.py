import torch
from torch import nn
import einops
import numpy as np
from torch.distributions import Categorical
from torch.utils.data.dataset import IterableDataset

from . import torch_utils as tu


class ReplayBuffer(nn.Module, IterableDataset):
    
    def __init__(self, batch_size, inv_freq_sequence=False, inv_freq_token=False, size=100000):
        super().__init__()
        self.size = size
        self.batch_size = batch_size
        self.inv_freq_sequence = inv_freq_sequence
        self.inv_freq_token = inv_freq_token
        self._index_dist = None

    @property
    def index_dist(self):
        if self._index_dist is None:
            logp = torch.zeros(len(self), device=self.w.device)
            if self.inv_freq_sequence:
                logp += self.get_inv_freq_sequence()
            if self.inv_freq_token:
                logp += self.get_inv_freq_token()
            self._index_dist = Categorical(logits=logp)
        return self._index_dist
    
    def reset(self):
        raise NotImplementedError
    
    def add(self, *args, **kwargs):
        raise NotImplementedError
    
    def sample_indices(self, indices):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size > len(self):
            indices = np.arange(len(self))
        else:
            indices = self.index_dist.sample((batch_size, ))
        return self.sample_indices(indices)

    def __iter__(self):
        yield self.sample()
    
    def get_inv_freq_sequence(self):
        unique, inverse, counts, index = tu.unique(self.w)
        logp = -counts.log()
        logp = logp - torch.logsumexp(logp, dim=0)
        logp = logp[inverse]
        return logp
    
    def get_inv_freq_token(self):
        logp = -self.w.sum(0).log()
        logp = logp - torch.logsumexp(logp, dim=0)
        logp = (logp * self.w).sum(-1)
        return logp
    

class EStepReplayBuffer(ReplayBuffer):

    def __init__(self, dim_z, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_z = dim_z
        self.vocab_size = vocab_size
        self.register_buffer('z0', torch.zeros(0, dim_z))
        self.register_buffer('z1', torch.zeros(0, dim_z))
        self.register_buffer('z2', torch.zeros(0, dim_z))
        self.register_buffer('t', torch.zeros(0, dtype=torch.long))
        self.register_buffer('w', torch.zeros(0, vocab_size, dtype=torch.long))

    def __len__(self):
        return len(self.z0)
        
    def reset(self):
        self.z0 = self.z0[:0]
        self.z1 = self.z1[:0]
        self.z2 = self.z2[:0]
        self.t = self.t[:0]
        self.w = self.w[:0]

    def sample_indices(self, indices):
        return self.z0[indices], self.z1[indices], self.z2[indices], self.w[indices], self.t[indices]
    
    def add(self, z0, z_traj, w):
        """
        z0: tensor with shape [batch_size, dim_z]
        z_traj: tensor with shape [batch_size, num_steps, dim_z]
        w: tensor with shape [batch_size, w_length]
        """
        t = torch.arange(z_traj.shape[1] - 1, device=z_traj.device)
        t = einops.repeat(t, 'k -> (b k)', b=z_traj.shape[0])
        z0 = einops.repeat(z0, 'b d -> (b k) d', k=z_traj.shape[1] - 1)
        w = einops.repeat(w, 'b l -> (b k) l', k=z_traj.shape[1] - 1)
        z1 = z_traj[:,:-1].reshape(-1, self.dim_z)
        z2 = z_traj[:,1:].reshape(-1, self.dim_z)
        
        self.z0 = torch.cat([self.z0, z0], dim=0)[-self.size:]
        self.z1 = torch.cat([self.z1, z1], dim=0)[-self.size:]
        self.z2 = torch.cat([self.z2, z2], dim=0)[-self.size:]
        self.t = torch.cat([self.t, t], dim=0)[-self.size:]
        self.w = torch.cat([self.w, w], dim=0)[-self.size:]


class MStepReplayBuffer(ReplayBuffer):

    def __init__(self, input_shape, vocab_size, input_dtype=torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.vocab_size = vocab_size
        self.input_dtype = input_dtype
        self.register_buffer('x', torch.zeros(0, *input_shape, dtype=input_dtype))
        self.register_buffer('w', torch.zeros(0, vocab_size, dtype=torch.long))

    def __len__(self):
        return len(self.x)
        
    def reset(self):
        self._index_dist = None
        self.x = self.x[:0]
        self.w = self.w[:0]

    def sample_indices(self, indices):
        return self.x[indices], self.w[indices]
        
    def add(self, x, w):
        """
        x: tensor with shape [batch_size, *input_shape]
        w: tensor with shape [batch_size, w_length]
        """
        self._index_dist = None
        self.x = torch.cat([self.x, x], dim=0)[-self.size:]
        self.w = torch.cat([self.w, w], dim=0)[-self.size:]
        

# class TrajectoryReplayBuffer(nn.Module):

#     def __init__(self, dim_z, vocab_size, inv_freq_sequence=False, inv_freq_token=False, size=100000):
#         super().__init__()
#         self.dim_z = dim_z
#         self.size = size
#         self.vocab_size = vocab_size
#         self.inv_freq_sequence = inv_freq_sequence
#         self.inv_freq_token = inv_freq_token
#         self.register_buffer('z0', torch.zeros(0, dim_z))
#         self.register_buffer('z1', torch.zeros(0, dim_z))
#         self.register_buffer('z2', torch.zeros(0, dim_z))
#         self.register_buffer('t', torch.zeros(0, dtype=torch.long))
#         self.register_buffer('w', torch.zeros(0, vocab_size, dtype=torch.long))
        
#     def reset(self):
#         self.z0 = self.z0[:0]
#         self.z1 = self.z1[:0]
#         self.z2 = self.z2[:0]
#         self.t = self.t[:0]
#         self.w = self.w[:0]

#     def add(self, z0, z_traj, w):
#         """
#         z0: tensor with shape [batch_size, dim_z]
#         z_traj: tensor with shape [batch_size, num_steps, dim_z]
#         w: tensor with shape [batch_size, w_length]
#         """
#         t = torch.arange(z_traj.shape[1] - 1, device=z_traj.device)
#         t = einops.repeat(t, 'k -> (b k)', b=z_traj.shape[0])
#         z0 = einops.repeat(z0, 'b d -> (b k) d', k=z_traj.shape[1] - 1)
#         w = einops.repeat(w, 'b l -> (b k) l', k=z_traj.shape[1] - 1)
#         z1 = z_traj[:,:-1].reshape(-1, self.dim_z)
#         z2 = z_traj[:,1:].reshape(-1, self.dim_z)
        
#         self.z0 = torch.cat([self.z0, z0], dim=0)[-self.size:]
#         self.z1 = torch.cat([self.z1, z1], dim=0)[-self.size:]
#         self.z2 = torch.cat([self.z2, z2], dim=0)[-self.size:]
#         self.t = torch.cat([self.t, t], dim=0)[-self.size:]
#         self.w = torch.cat([self.w, w], dim=0)[-self.size:]
        
#     def get_inv_freq_sequence(self):
#         unique, inverse, counts, index = tu.unique(self.w)
#         logp = -counts.log()
#         logp = logp - torch.logsumexp(logp, dim=0)
#         logp = logp[inverse]
#         return logp
    
#     def get_inv_freq_token(self):
#         logp = -self.w.sum(0).log()
#         logp = logp - torch.logsumexp(logp, dim=0)
#         logp = (logp * self.w).sum(-1)
#         return logp

#     def sample(self, batch_size):
#         if batch_size > len(self):
#             indices = np.arange(len(self))
#         else:
#             logp = torch.zeros(len(self), device=self.w.device)
#             if self.inv_freq_sequence:
#                 logp += self.get_inv_freq_sequence()
#             if self.inv_freq_token:
#                 logp += self.get_inv_freq_token()
#             indices = Categorical(logits=logp).sample((batch_size, ))
#         return self.z0[indices], self.z1[indices], self.z2[indices], self.w[indices], self.t[indices]
    
#     def __len__(self):
#         return len(self.z0)


# class TrajectoryReplayBuffer(nn.Module):

#     def __init__(self, dim_z, vocab_size, size=100000):
#         super().__init__()
#         self.dim_z = dim_z
#         self.size = size
#         self.vocab_size = vocab_size
#         self.register_buffer('z0', torch.zeros(0, dim_z))
#         self.register_buffer('z1', torch.zeros(0, dim_z))
#         self.register_buffer('z2', torch.zeros(0, dim_z))
#         self.register_buffer('t', torch.zeros(0, dtype=torch.long))
#         self.register_buffer('w', torch.zeros(0, vocab_size, dtype=torch.long))

#     def reset(self):
#         self.z0 = self.z0[:0]
#         self.z1 = self.z1[:0]
#         self.z2 = self.z2[:0]
#         self.t = self.t[:0]
#         self.w = self.w[:0]

#     def add(self, z0, z_traj, w):
#         """
#         z0: tensor with shape [batch_size, dim_z]
#         z_traj: tensor with shape [batch_size, num_steps, dim_z]
#         w: tensor with shape [batch_size, w_length]
#         """
#         t = torch.arange(z_traj.shape[1] - 1, device=z_traj.device)
#         t = einops.repeat(t, 'k -> (b k)', b=z_traj.shape[0])
#         z0 = einops.repeat(z0, 'b d -> (b k) d', k=z_traj.shape[1] - 1)
#         w = einops.repeat(w, 'b l -> (b k) l', k=z_traj.shape[1] - 1)
#         z1 = z_traj[:,:-1].reshape(-1, self.dim_z)
#         z2 = z_traj[:,1:].reshape(-1, self.dim_z)
        
#         self.z0 = torch.cat([self.z0, z0], dim=0)[-self.size:]
#         self.z1 = torch.cat([self.z1, z1], dim=0)[-self.size:]
#         self.z2 = torch.cat([self.z2, z2], dim=0)[-self.size:]
#         self.t = torch.cat([self.t, t], dim=0)[-self.size:]
#         self.w = torch.cat([self.w, w], dim=0)[-self.size:]

#     def sample(self, batch_size):
#         if batch_size > len(self):
#             indices = np.arange(len(self))
#         else:
#             indices = np.random.choice(np.arange(len(self)), size=batch_size, replace=False)
#         return self.z0[indices], self.z1[indices], self.z2[indices], self.w[indices], self.t[indices]
    
#     def __len__(self):
#         return len(self.z0)
    

# class TrajectoryReplayBuffer(nn.Module):

#     def __init__(self, dim_z, size=100000):
#         super().__init__()
#         self.dim_z = dim_z
#         self.size = size
#         self.register_buffer('z0', torch.zeros(0, dim_z))
#         # self.register_buffer('z', torch.zeros(0, dim_z))
#         self.register_buffer('z1', torch.zeros(0, dim_z))
#         self.register_buffer('z2', torch.zeros(0, dim_z))
#         self.register_buffer('t', torch.zeros(0, dtype=torch.long))

#     def add(self, z0, z_traj):
#         """
#         z0: tensor with shape [batch_size, dim_z]
#         z_traj: tensor with shape [batch_size, num_steps, dim_z]
#         """
#         t = torch.arange(z_traj.shape[1] - 1, device=z_traj.device)
#         t = einops.repeat(t, 'k -> (b k)', b=z_traj.shape[0])
#         z0 = einops.repeat(z0, 'b d -> (b k) d', k=z_traj.shape[1] - 1)
#         z1 = z_traj[:,:-1].reshape(-1, self.dim_z)
#         z2 = z_traj[:,1:].reshape(-1, self.dim_z)
#         self.z0 = torch.cat([self.z0, z0], dim=0)[-self.size:]
#         self.z1 = torch.cat([self.z1, z1], dim=0)[-self.size:]
#         self.z2 = torch.cat([self.z2, z2], dim=0)[-self.size:]
#         self.t = torch.cat([self.t, t], dim=0)[-self.size:]
#     # def add(self, z0, z, t=None):
#     #     """
#     #     z0: tensor with shape [batch_size, dim_z]
#     #     z: tensor with shape [batch_size, dim_z] or [batch_size, num_steps, dim_z]
#     #     """
#     #     if len(z0) != len(z):
#     #         raise ValueError(f'z0 and z must have the same batch dimension. Got {len(z0)} and {len(z)}.')
#     #     if t is None:
#     #         if z.ndim == 3:
#     #             t = torch.arange(z.shape[-2], device=z.device)
#     #             t = einops.repeat(t, 'k -> (b k)', b=z.shape[0])
#     #         else:
#     #             raise Exception('t must be provided if z has 2 dimensions.')
#     #     z0 = einops.repeat(z0, 'b d -> (b k) d', k=z.shape[1])
#     #     z = z.view(-1, self.dim_z)
#     #     self.z0 = torch.cat([self.z0, z0], dim=0)[-self.size:]
#     #     self.z = torch.cat([self.z, z], dim=0)[-self.size:]
#     #     self.t = torch.cat([self.t, t], dim=0)[-self.size:]

        
#     def sample(self, batch_size):
#         if batch_size > len(self):
#             indices = np.arange(len(self))
#         else:
#             indices = np.random.choice(np.arange(len(self)), size=batch_size, replace=False)
#         return self.z0[indices], self.z1[indices], self.z2[indices], self.t[indices]
    
#     def __len__(self):
#         return len(self.z0)


# # class ReplayBuffer(nn.Module):

# #     def __init__(self, num_inputs, size_per_input, state_shape, state_dtype, log_reward_decay=-.0513):
# #         super().__init__()
# #         self.num_inputs = num_inputs
# #         self.size_per_input = size_per_input
# #         self.state_shape = state_shape
# #         self.state_dtype = state_dtype
# #         self.log_reward_decay = log_reward_decay
        
# #         length = torch.prod(torch.tensor(state_shape))
# #         self.register_buffer('buffer', torch.zeros(num_inputs, size_per_input, length, dtype=state_dtype))
# #         self.register_buffer('log_rewards', torch.full((num_inputs, size_per_input), -torch.inf))

# #     def update(self, input_ids, states, log_rewards):
# #         """
# #         input_ids: (batch_size, )
# #         states: (batch_size, *state_shape) or (batch_size, k, *state_shape)
# #         log_rewards: (batch_size, ) or (batch_size, k)
# #         """
# #         states = states.detach().clone()
# #         log_rewards = log_rewards.detach().clone()

# #         if states.ndim == len(self.state_shape) + 1:
# #             states = states.unsqueeze(1)
# #             log_rewards = log_rewards.unsqueeze(1)
# #         states = states.view(states.shape[0], states.shape[1], -1)
# #         buffer = torch.cat([self.buffer[input_ids], states], dim=1)
# #         log_rewards = torch.cat([self.log_rewards[input_ids] - self.log_reward_decay, log_rewards], dim=1)
# #         log_rewards, top_indices = torch.sort(log_rewards, dim=1, descending=True)
# #         top_indices = top_indices[:, :self.size_per_input]
# #         log_rewards = log_rewards[:, :self.size_per_input]

# #         self.buffer[input_ids] = buffer.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, buffer.shape[-1]))
# #         self.log_rewards[input_ids] = log_rewards

# #     def sample(self, input_ids):
# #         """
# #         input_ids: (batch_size, )
# #         returns: 
# #             batch_mask: (batch_size, ) indicating whether there was a sample. There are sample_batch_size True values in batch_mask.
# #             states: (sample_batch_size, *state_shape)

# #         Returns input_ids because it may have been truncated.
# #         """
# #         has_samples = self.log_rewards[input_ids] > -torch.inf
# #         keep = has_samples.any(-1)
# #         input_ids = input_ids[keep]
# #         indices = torch.multinomial(has_samples[keep].float(), num_samples=1).squeeze(1)
# #         states = self.buffer[input_ids, indices].view(len(input_ids), *self.state_shape)
# #         return keep, states.clone()
