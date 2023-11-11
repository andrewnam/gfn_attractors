import string
import einops
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..misc import torch_utils as tu
from .helpers import SafeEmbedding, MLP


class DiscretizeModule(nn.Module):
    """
    Abstract class for modules that use discrete tokens sequences.
    """

    def __init__(self, vocab_size: int, max_length: int, characters=string.ascii_letters):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.characters = characters[:vocab_size]
        
    @property
    def num_tokens(self):
        return 2 + self.vocab_size
    
    @property
    def pad(self):
        return 0
    
    @property
    def eos(self):
        return 1
    
    def get_all_w(self, length=None, device='cpu'):
        if length is None:
            length = self.max_length
        w = [''.join(s) for s in itertools.product(self.characters[:self.vocab_size], repeat=length)]
        w = tu.from_strings(w, eos=self.eos, min_value=1+self.eos, device=device)
        return w
    
    def sample_w(self, p_length, n, device='cpu'):
        return sample_w(p_length, self.vocab_size, n, device=device)
    
    def tokenize(self, w, device='cpu'):
        if hasattr(self, 'device'):
            device = self.device
        return tu.from_strings(w, chars=self.characters, device=device)
    
    def stringify(self, w):
        """
        w: [batch_size, sequence_length]
        """
        return tu.to_strings(w, chars=self.characters, min_value=2)
    
    def get_length(self, w):
        eos = (w == self.eos)
        if not eos.any(-1).all():
            raise ValueError('Not all sequences have an EOS token.')
        return eos.byte().argmax(-1)
    
    def get_mask(self, w):
        """
        w: [..., sequence_length]
        returns a mask of shape [..., sequence_length], where 1 indicates that the token comes after EOS and should be masked
        """
        length = self.get_length(w)
        mask = torch.arange(w.shape[-1], device=w.device)
        mask = einops.repeat(mask, 'l -> b l', b=w.shape[0])
        mask = mask > length.unsqueeze(-1)
        return mask
    

class Discretizer(DiscretizeModule):
    """
    Abstract class for vector-to-seq models.
    """

    def __init__(self,
                 vocab_size: int,
                 max_length: int,
                 dim_h=256,
                 characters=string.ascii_letters,):
        super().__init__(vocab_size, max_length, characters=characters)
        self.dim_h = dim_h

        self.sos = nn.Parameter(torch.randn(dim_h))
        self.embedding = SafeEmbedding(self.num_tokens, self.dim_h)
        self.sigma_logit = nn.Parameter(torch.tensor(0.))

    @property
    def device(self):
        return self.sos.device
    
    @property
    def sigma(self):
        return self.sigma_logit.sigmoid()

    def get_wi_logits(self, x, w_embedding, i: int, _cache=None):
        """
        z: [batch_size, ...]
        w_embedding: [batch_size, 1+length, dim_h]
            First token is SOS

        Returns logits of shape [batch_size, num_tokens]
        cache: a dict of tensors
        """
        raise NotImplementedError
    
    def get_logits(self, x, w_embedding):
        """
        Return autoregressive next-token prediction logits using teacher-forcing.

        x: [batch_size, ...]
        w_embedding: [batch_size, 1+max_length, dim_h]
            First token is SOS, last token is missing (because it should only be predicted)
        returns: [batch_size, 1+max_length, num_tokens]
        """
        raise NotImplementedError
    
    def get_vae_loss(self, x, w, beta=1):
        """
        Assuming that x is a latent representation of w, returns the VAE loss.
        Uses x as the predicted mean of the latent.

        x: [batch_size, ...]
        w: [batch_size, 1+max_length]
        """
        z = x + torch.randn_like(x) * self.sigma
        w_embedding = self.embedding(w)
        logits = self.get_logits(z, w_embedding)
        recon_loss = tu.batch_cross_entropy(logits, w, ignore_index=self.pad, reduction='none').sum(-1).mean()
        kl_loss = tu.get_kl_div(x, self.sigma.log()).mean()
        loss = recon_loss + beta * kl_loss
        accuracy = ((logits.argmax(-1) == w) | self.get_mask(w)).all(-1).float().mean()
        metrics = {'dvae/recon_loss': recon_loss, 'dvae/kl_loss': kl_loss, 'dvae/accuracy': accuracy, 'dvae/sigma': self.sigma}
        return loss, metrics
    
    def get_random_substrings(self, w):
        """
        For each w, returns a random contiguous substring of w.
        If w is a string of length 0 or 1, returns w.
        w: [batch_size, sequence_length]
        returns: [batch_size, sequence_length]
        """
        w_strings = self.stringify(w)
        substrings = []
        for s in w_strings:
            if len(s) <= 1:
                substrings.append(s)
            else:
                sublength = np.random.randint(1, len(s)+1)
                start = np.random.randint(0, len(s)-sublength+1)
                substrings.append(s[start:start+sublength])
        subw = self.tokenize(substrings).to(w.device)
        return subw
    
    def sample(self, x, temperature=1, p_explore=0, argmax=False, target=None):
        """
        x: tensor with shape [batch_size, ...]
        returns:
            w_seq: tensor with shape [batch_size, max_length+1]
            logp_w: tensor with shape [batch_size]
        """
        batch_size = x.shape[0]
        w_seq = torch.zeros(batch_size, 1+self.max_length, dtype=torch.long, device=self.device)
        logp_w = torch.zeros(batch_size, device=self.device)
        cache = {}
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        w_embeddings = einops.repeat(self.sos, 'h -> b 1 h', b=batch_size)
        for i in range(self.max_length):
            logits, cache = self.get_wi_logits(x[~done], w_embeddings, i, _cache=cache)
            logits = logits / temperature
            logits[:,0] = -1e8 # pad
            log_probs = logits.log_softmax(-1)
            if target is not None:
                w = target[:,i]
            elif argmax:
                w = log_probs.argmax(-1)
            else:
                sample_probs = (1-p_explore) * log_probs.exp() + p_explore / (1 + self.vocab_size)
                sample_probs[:,0] = 0 # pad
                w = Categorical(sample_probs).sample()
        
            logp = log_probs.gather(-1, w.unsqueeze(-1)).squeeze(-1)
            logp_w[~done] += logp
            w_seq[~done, i] = w
            terminate = (w == self.eos)
            done[~done] = done[~done] | terminate
            w_embeddings = torch.cat([w_embeddings[~terminate], self.embedding(w[~terminate]).unsqueeze(1)], dim=1)
            cache = {k: v[~terminate] if isinstance(v, torch.Tensor) else v for k, v in cache.items()}
            if done.all():
                break
        w_seq[~done, self.max_length] = self.eos
        return w_seq, logp_w
    
    def sample_terminate_every_step(self, x, temperature=1, p_explore=0):
        """
        Used for training with terminate at every step.

        x: image tensor with shape [batch_size, ...]
        returns:
            w: tensor with shape [batch_size, max_length+1, max_length+1]
            logpf: tensor with shape [batch_size, max_length+1]
            logpt: tensor with shape [batch_size, max_length+1]
        """
        batch_size = x.shape[0]

        w_seq = torch.zeros(batch_size, 1+self.max_length, dtype=torch.long, device=self.device)
        logpf = torch.zeros(batch_size, 1+self.max_length, device=self.device)
        logpt = torch.zeros(batch_size, 1+self.max_length, device=self.device)
        cache = {}
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        w_embeddings = einops.repeat(self.sos, 'h -> b 1 h', b=batch_size)
        for i in range(self.max_length):
            logits, cache = self.get_wi_logits(x[~done], w_embeddings, i, _cache=cache)
            logits = logits / temperature
            logits[:,0] = -1e8 # pad
            log_probs = logits.log_softmax(-1)

            sample_probs = (1-p_explore) * log_probs.exp() + p_explore / (1 + self.vocab_size)
            sample_probs[:,:2] = 0 # pad and eos
            w = Categorical(sample_probs).sample()
    
            logpw = log_probs.gather(-1, w.unsqueeze(-1)).squeeze(-1)
            logpf[~done, i] += logpw
            logpt[~done, i] += log_probs[:,self.eos]
            w_seq[~done, i] = w
            terminate = (w == self.eos)
            done[~done] = done[~done] | terminate
            w_embeddings = torch.cat([w_embeddings[~terminate], self.embedding(w[~terminate]).unsqueeze(1)], dim=1)
            cache = {k: v[~terminate] if isinstance(v, torch.Tensor) else v for k, v in cache.items()}
            if done.all():
                break
        w_seq[~done, self.max_length] = self.eos

        w = torch.zeros(w_seq.shape[0], w_seq.shape[1], w_seq.shape[1], dtype=int, device=w_seq.device)
        w[:, range(w_seq.shape[1]), range(w_seq.shape[1])] = 1
        for i in range(1, w_seq.shape[1]):
            w[:, i, :i] = w_seq[:, :i]
        return w, logpf, logpt
    

class MLPDiscretizer(Discretizer):

    def __init__(self, vocab_size: int, length: int, dim_input: int,
                 dim_h=256, characters=string.ascii_letters, num_layers=2, nonlinearity=nn.ReLU(), **kwargs):
        super().__init__(vocab_size=vocab_size, max_length=length, dim_h=dim_h, characters=characters)
        self.dim_input = dim_input
        self.length = length
        self.mlp = MLP(dim_input, length * self.vocab_size, hidden_dim=dim_h, n_layers=num_layers, nonlinearity=nonlinearity)

    def sample(self, z, temperature=1, p_explore=0, argmax=False, target=None):
        """
        z: [batch_size, dim_z]
        returns
            w: [batch_size, 1+length]
            logp: [batch_size]
        """
        batch_size = z.shape[0]
        logits = self.mlp(z).view(batch_size, self.length, self.vocab_size) / temperature
        if target is not None:
            w = target[:,:-1] - 2 # drop EOS and drop indices by 2
        elif argmax:
            w = logits.argmax(-1)
        else:
            probs = (1-p_explore) * logits.softmax(-1) + p_explore / self.vocab_size
            w = Categorical(probs).sample()
        
        logp = Categorical(logits=logits).log_prob(w).sum(-1)
        w = F.pad(2 + w, (0, 1), value=self.eos)
        return w, logp
    
    def get_logits(self, z, w_embedding):
        """
        z: [batch_size, dim_z]
        w_embedding: [batch_size, 1+length, dim_h]
        returns: [batch_size, 1+length, num_tokens]
        """
        logits = self.mlp(z).view(z.shape[0], self.length, self.vocab_size)
        return logits
    

def sample_w(p_length, vocab_size, n, device='cpu'):
    """
    p_length: tensor with shape [max_length]
    Assumes that pad=0 and eos=1
    """
    length = Categorical(p_length).sample((n, ))
    max_length = length.max()
    w = 2 + torch.randint(0, vocab_size, (n, 1 + max_length), device=device)
    length = length.unsqueeze(1)
    w = w.scatter(1, length, torch.ones_like(length, device=device))
    mask = torch.arange(w.shape[-1], device=w.device) > length
    w = w.masked_fill(mask, 0)
    return w
