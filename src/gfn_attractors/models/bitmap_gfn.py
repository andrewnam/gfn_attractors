import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import Counter
from tqdm.auto import trange
import numpy as np
import einops

from .helpers import MLP
from ..misc.torch_utils import nCk
from .discretizer import BoWDiscretizeModule


class BitmapGFN(BoWDiscretizeModule):
    """
    A simple module that takes an input tensor and returns a sequence of bits.
    Useful for sampling graphs, etc.

    group_size: if not None, the bits are grouped into groups of size group_size. Only one bit in each group can be active.
    """

    def __init__(self, dim_input, num_bits, group_size=None, dim_h=256, num_layers=3, nonlinearity=nn.ReLU(), epsilon=1e-10, 
                 no_flow_model=False, dim_flow_input=None, *args, **kwargs):
        super().__init__(vocab_size=num_bits, group_size=group_size)
        if group_size is not None and num_bits % group_size != 0:
            raise ValueError('num_bits must be divisible by group_size.')

        self.dim_input = dim_input
        self.num_bits = num_bits
        self.dim_h = dim_h
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.no_flow_model = no_flow_model
        self.dim_flow_input = dim_input if dim_flow_input is None else dim_flow_input
        self.epsilon = epsilon

        self.mlp = MLP(dim_input + num_bits, 1 + 2*num_bits, dim_h, n_layers=num_layers, nonlinearity=nonlinearity)
        if not no_flow_model:
            self.flow_model = MLP(self.dim_flow_input, 1, dim_h, n_layers=num_layers, nonlinearity=nonlinearity)

    def get_logpw(self, x, log_reward):
        """
        x: tensor with shape [batch_size, dim_flow_input]
        log_reward: tensor with shape [batch_size, 1+max_steps]
        """
        if self.no_flow_model:
            raise ValueError('Cannot get marginal P(w|x) without a flow model.')
        logZ = self.flow_model(x)
        return log_reward - logZ


    def sample(self, x, min_steps=0, max_steps=None, target=None, temperature=1, argmax=False, p_explore=0, allow_terminate=True):
        """
        x: image tensor with shape [batch_size, dim_input]
        returns:
            w: If allow terminate, tensor with shape [batch_size, num_bits] 
               Otherwise, tensor with shape [batch_size, 1+max_steps, num_bits]
            logpf: tensor with shape [batch_size, max_steps]
            logpb: tensor with shape [batch_size, max_steps]
            logpt: If allow terminate, tensor with shape [batch_size] 
               Otherwise, tensor with shape [batch_size, max_steps+1]
        """
        if max_steps is None:
            max_steps = self.num_bits
            if self.group_size is not None:
                max_steps = max_steps // self.group_size
        if not allow_terminate:
            min_steps = max_steps 

        max_steps += 1
        batch_size = x.shape[0]
        if allow_terminate:
            states = torch.zeros(batch_size, self.num_bits, dtype=torch.float32, device=x.device)
            logpt = torch.zeros(batch_size, device=x.device)
        else:
            states = torch.zeros(batch_size, max_steps, self.num_bits, dtype=torch.float32, device=x.device)
            logpt = torch.zeros(batch_size, max_steps, device=x.device)
        logpf = torch.zeros(batch_size, max_steps-1, device=x.device)
        logpb = torch.zeros(batch_size, max_steps-1, device=x.device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for i in range(max_steps):
            if allow_terminate:
                state = states[~done]
            else:
                state = states[:, i]
            h = torch.cat([x[~done], state], dim=-1)
            forward_logits, backward_logits = self.mlp(h).split([1+self.num_bits, self.num_bits], dim=-1)

            if i > 0:
                backward_mask = state == 0
                backward_logits = backward_logits.masked_fill(backward_mask, -1e8)
                backward_logits = backward_logits / temperature
                backward_log_probs = backward_logits.log_softmax(-1)
                logpb[~done, i-1] = backward_log_probs.gather(-1, w_i.unsqueeze(-1)).squeeze(-1)
                
                if done.all():
                    break
            if i < max_steps-1:
                if self.group_size is None:
                    invalid = state > 0
                else:
                    invalid = (state.view(len(state), -1, self.group_size) > 0).any(-1)
                    invalid = einops.repeat(invalid, 'b k -> b (k g)', g=self.group_size)
                forward_mask = F.pad(invalid, (0, 1), value=0)#i < min_steps-1)
                forward_logits = forward_logits.masked_fill(forward_mask, -1e8)
                forward_logits = forward_logits / temperature

                forward_log_probs = forward_logits.log_softmax(-1)

                if allow_terminate:
                    logpt[~done] = forward_log_probs[:,-1]
                else:
                    logpt[:,i] = forward_log_probs[:,-1]

                if target is not None:
                    forward_probs = forward_log_probs.exp()
                    forward_mask = invalid | ~(target[~done].bool())
                    forward_mask = torch.cat([forward_mask, ~(forward_mask.all(-1)).unsqueeze(-1)], dim=-1)
                    forward_probs = forward_probs.masked_fill(forward_mask, 0)
                    w_i = Categorical(forward_probs.clamp(self.epsilon)).sample()
                elif argmax:
                    w_i = forward_logits.argmax(-1)
                else:
                    forward_probs = forward_log_probs.exp()
                    forward_probs = (1-p_explore) * forward_probs + p_explore / (1 + self.num_bits)
                    forward_probs = forward_probs.masked_fill(forward_mask, 0)
                    if i < min_steps:
                        forward_probs[:, -1] = 0
                    w_i = Categorical(forward_probs.clamp(self.epsilon)).sample()

                if allow_terminate:
                    terminate = (w_i == self.num_bits)
                    done[~done] = done[~done] | terminate
                    w_i = w_i[~terminate]
                    logpf[~done, i] = forward_log_probs[~terminate].gather(-1, w_i.unsqueeze(-1)).squeeze(-1)
                    states[~done] += F.one_hot(w_i, self.num_bits).float()
                else:
                    logpf[:, i] = forward_log_probs.gather(-1, w_i.unsqueeze(-1)).squeeze(-1)
                    states[:, i+1] = states[:,i] + F.one_hot(w_i, self.num_bits).float()

        return states, logpf, logpb, logpt
    
    def get_db_loss(self, logpf, logpb, logpt, log_reward):
        """
        x: tensor with shape [batch_size, dim_input]
        logpf: tensor with shape [batch_size, max_steps]
        logpb: tensor with shape [batch_size, max_steps]
        logpt: tensor with shape [batch_size, 1+max_steps]
        log_reward: tensor with shape [batch_size, 1+max_steps]
        """
        log_reward = log_reward.detach()
        lhs = log_reward[:,:-1] - logpt[:,:-1] + logpf
        rhs = log_reward[:,1:] - logpt[:,1:] + logpb
        loss = (lhs - rhs).pow(2).mean()
        metrics = {
            'log_reward': log_reward.mean().item(),
            'logpf': logpf.mean().item(),
            'logpb': logpb.mean().item(),
            'logpt': logpt.mean().item(),
            'loss': loss.item()
        }
        return loss, metrics
    
    def get_tb_loss(self, x, logpf, logpb, logpt, log_reward, logZ=None):
        """
        x: tensor with shape [batch_size, dim_flow_input]
        logpf: tensor with shape [batch_size, max_steps]
        logpb: tensor with shape [batch_size, max_steps]
        logpt: tensor with shape [batch_size] or [batch_size, 1+max_steps]
        log_reward: tensor with shape [batch_size] or [batch_size, 1+max_steps]
        logZ: tensor with shape [batch_size]
            If not None, uses this value instead of computing it from the flow model.
            Use this if there's some other model that computes the flow.
        """
        if x.ndim > 2:
            raise ValueError('x must have shape [batch_size, dim_input]')
        if self.no_flow_model:
            raise ValueError('Cannot use trajectory balance objective without a flow model.')
        log_reward = log_reward.detach()
        if logZ is None:
            logZ = self.flow_model(x)
        if logpt.ndim == 1:
            loss = logZ + logpf.sum(-1) + logpt - logpb.sum(-1) - log_reward
        if logpt.ndim == 2:
            logpf = F.pad(logpf, (1, 0))
            logpb = F.pad(logpb, (1, 0))
            logpf = logpf.cumsum(-1)
            logpb = logpb.cumsum(-1)
            loss = logZ.unsqueeze(1) + logpf + logpt - logpb - log_reward
        loss = loss.pow(2).mean()
        metrics = {
            'logZ': logZ.mean(),
            'logpf': logpf.mean(),
            'logpb': logpb.mean(),
            'logpt': logpt.mean(),
            'log_reward': log_reward.mean(),
            'loss': loss.item()
        }
        return loss, metrics


from tqdm.auto import trange
from collections import Counter

class ExampleBitmapGFN(nn.Module):
    """
    An example of a model that uses the BitmapGFN module.
    Reward is the number of bits that the model got correct, scaled by the number of possible sequences with k 1-bits.
        For example, if num_bits=4, P(num_bits_correct) should be [0, 1/10, 2/10, 3/10, 4/10].
    """

    def __init__(self, num_bits=8, dim_h=256, num_layers=3):
        super().__init__()
        self.num_bits = num_bits
        self.dim_h = dim_h
        self.num_layers = num_layers
        self.discretizer = BitmapGFN(num_bits=num_bits, dim_input=num_bits, dim_h=dim_h, num_layers=num_layers)

    def get_loss(self, x, objective: str, p_explore=0.):
        assert objective in ('tb', 'subtb', 'db')
        
        w, logpf, logpb, logpt = self.discretizer.sample(x, p_explore=p_explore, allow_terminate=objective=='tb')
        if w.ndim == 2:
            match = (x == w).sum(-1)
        else:
            match = (x.unsqueeze(1) == w).sum(-1)
        with torch.no_grad():
            log_reward = (match + .001).log() - tu.nCk(torch.full_like(match, fill_value=self.num_bits), match, log=True)
        if objective == 'db':
            loss, metrics = self.discretizer.get_db_loss(logpf, logpb, logpt, log_reward)
        else:
            loss, metrics = self.discretizer.get_tb_loss(x, logpf, logpb, logpt, log_reward)
        return loss, metrics
    
    def train(self, objective: str, num_iters=2000, batch_size=64, p_explore=.05, device='cpu'):
        n = np.arange(1 + self.num_bits)
        expected = (n / sum(n)).round(3)
        print(f"Expected P(correct): {list(enumerate(expected))}")

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        for i in trange(num_iters):
            x = (torch.rand(batch_size, self.num_bits, device=device) > 0.5).float()
            loss, metrics = self.get_loss(x, objective, p_explore=p_explore)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 200 == 0 or i == num_iters-1:
                x = torch.rand(500, self.num_bits, device=device) > .5
                with torch.no_grad():
                    w, logpf, logpb, logpt = self.discretizer.sample(x)
                match = (w == x).sum(-1).tolist()
                counts = Counter(match)
                counts = {k: counts[k]/500 for k in sorted(counts.keys())}
                print(i, loss.item(), counts)
            



# class BitmapGFN(BoWDiscretizeModule):
#     """
#     A simple module that takes an input tensor and returns a sequence of bits.
#     Useful for sampling graphs, etc.

#     group_size: if not None, the bits are grouped into groups of size group_size. Only one bit in each group can be active.
#     """

#     def __init__(self, dim_input, num_bits, group_size=None, dim_h=256, num_layers=3, nonlinearity=nn.ReLU(), *args, **kwargs):
#         super().__init__(vocab_size=num_bits)
#         if group_size is not None and num_bits % group_size != 0:
#             raise ValueError('num_bits must be divisible by group_size.')

#         self.dim_input = dim_input
#         self.num_bits = num_bits
#         self.group_size = group_size
#         self.dim_h = dim_h
#         self.num_layers = num_layers
#         self.nonlinearity = nonlinearity
#         self.mlp = MLP(dim_input + num_bits, 1 + 2*num_bits, dim_h, n_layers=num_layers, nonlinearity=nonlinearity)

#     def sample(self, x, min_steps=0, max_steps=None, target=None, temperature=1, argmax=False, p_explore=0):
#         """
#         x: image tensor with shape [batch_size, dim_input]
#         returns:
#             w: tensor with shape [batch_size, num_bits]
#             logpf: tensor with shape [batch_size, max_steps]
#             logpb: tensor with shape [batch_size, max_steps]
#         """
#         if max_steps is None:
#             max_steps = self.num_bits
#             if self.group_size is not None:
#                 max_steps = max_steps // self.group_size
#         batch_size = x.shape[0]
#         state = torch.zeros(batch_size, self.num_bits, dtype=torch.float32, device=x.device)
#         logpf = torch.zeros(batch_size, max_steps, device=x.device)
#         logpb = torch.zeros(batch_size, max_steps, device=x.device)
#         done = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

#         for i in range(max_steps+1):
#             s = state[~done]
#             h = torch.cat([x[~done], s], dim=-1)
#             forward_logits, backward_logits = self.mlp(h).split([1+self.num_bits, self.num_bits], dim=-1)

#             if i > 0:
#                 backward_mask = s == 0
#                 backward_logits = backward_logits.masked_fill(backward_mask, -1e8)
#                 backward_logits = backward_logits / temperature
#                 backward_log_probs = backward_logits.log_softmax(-1)
#                 logpb[~done, i-1] = backward_log_probs.gather(-1, w_i.unsqueeze(-1)).squeeze(-1)
                
#                 if done.all():
#                     break
#             if i < max_steps:
#                 if self.group_size is None:
#                     invalid = s > 0
#                 else:
#                     invalid = (s.view(len(s), -1, self.group_size) > 0).any(-1)
#                     invalid = einops.repeat(invalid, 'b k -> b (k g)', g=self.group_size)
#                 forward_mask = F.pad(invalid, (0, 1), value=i < min_steps-1)
#                 forward_logits = forward_logits.masked_fill(forward_mask, -1e8)
#                 forward_logits = forward_logits / temperature

#                 forward_log_probs = forward_logits.log_softmax(-1)
#                 forward_dist = Categorical(logits=forward_logits)

#                 if target is not None:
#                     forward_probs = forward_log_probs.exp()
#                     forward_mask = invalid | ~(target[~done].bool())
#                     forward_mask = torch.cat([forward_mask, ~(forward_mask.all(-1)).unsqueeze(-1)], dim=-1)
#                     forward_probs = forward_probs.masked_fill(forward_mask, 0)
#                     w_i = Categorical(forward_probs).sample()
#                 elif argmax:
#                     w_i = forward_logits.argmax(-1)
#                 elif p_explore > 0:
#                     forward_probs = forward_log_probs.exp()
#                     forward_probs = (1-p_explore) * forward_probs + p_explore / (1 + self.num_bits)
#                     forward_probs = forward_probs.masked_fill(forward_mask, 0)
#                     w_i = Categorical(forward_probs).sample()
#                 else:
#                     w_i = forward_dist.sample()
#                 logpf[~done, i] = forward_log_probs.gather(-1, w_i.unsqueeze(-1)).squeeze(-1)
#                 terminate = (w_i == self.num_bits)
#                 done[~done] = done[~done] | terminate
#                 state[~done] += F.one_hot(w_i[~terminate], self.num_bits).float()
#                 w_i = w_i[~terminate]

#         return state, logpf, logpb


# class ExampleBitmapGFN(nn.Module):
#     """
#     An example of a model that uses the BitmapGFN module.
#     Reward is the number of bits that the model got correct, scaled by the number of possible sequences with k 1-bits.
#         For example, if num_bits=4, P(num_bits_correct) should be [0, 1/10, 2/10, 3/10, 4/10].
#     Uses TB objective for training.
#     """

#     def __init__(self, num_bits=8, dim_h=256, num_layers=3):
#         super().__init__()
#         self.num_bits = num_bits
#         self.dim_h = dim_h
#         self.num_layers = num_layers
#         self.discretizer = BitmapGFN(num_bits=num_bits, dim_input=num_bits, dim_h=dim_h, num_layers=num_layers)
#         self.flow_model = MLP(num_bits, 1, dim_h, n_layers=2)

#     def get_loss(self, x, p_explore=0.):
#         log_Z = self.flow_model(x)
#         w, logpf, logpb = self.discretizer.sample(x, p_explore=p_explore)
#         logpf = logpf.sum(-1)
#         logpb = logpb.sum(-1)
#         match = (x == w).sum(-1)
#         log_reward = (match + .001).log() - nCk(torch.full_like(match, fill_value=self.num_bits), match, log=True)
#         loss = log_Z + logpf - log_reward - logpb
#         loss = loss.pow(2).mean()
#         return loss
    
#     def train(self, num_iters=2000, batch_size=64, p_explore=.05, device='cpu'):
#         n = np.arange(1 + self.num_bits)
#         expected = (n / sum(n)).round(3)
#         print(f"Expected P(correct): {list(enumerate(expected))}")

#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         for i in trange(num_iters):
#             x = (torch.rand(batch_size, self.num_bits, device=device) > 0.5).float()
#             loss = self.get_loss(x, p_explore=p_explore)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             if i % 100 == 0:
#                 x = torch.rand(500, self.num_bits, device=device) > .5
#                 with torch.no_grad():
#                     w, logpf, logpb = self.discretizer.sample(x)
#                 match = (w == x).sum(-1).tolist()
#                 counts = Counter(match)
#                 counts = {k: counts[k]/500 for k in sorted(counts.keys())}
#                 print(loss.item(), counts)
