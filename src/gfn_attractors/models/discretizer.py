import string
import einops
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from functools import cache

from ..misc import torch_utils as tu
from .helpers import SafeEmbedding, MLP, PositionalEncoding


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
    
    @cache
    def get_all_w(self, length=None):
        if length is None:
            length = self.max_length
        w = [''.join(s) for s in itertools.product(self.characters[:self.vocab_size], repeat=length)]
        w = tu.from_strings(w, eos=self.eos, min_value=1+self.eos)
        return w
    
    @cache
    def get_all_w_strings(self, length=None):
        w = self.get_all_w(length=None)
        return self.stringify(w)
    
    def sample_w(self, p_length, n, enforce_sort=False):
        """
        p_length: tensor with shape [1 + max_length]
        """
        if not enforce_sort:
            return super().sample_w(p_length, self.vocab_size, n)
        else:
            # Definitely not the fastest way, but oh well
            length = Categorical(p_length).sample((n, ))
            max_length = length.max()
            tokens = 2 + np.arange(20)
            w = np.zeros((n, 1+max_length), dtype=int)
            for i, l in enumerate(length.tolist()):
                row = np.random.choice(tokens, size=l, replace=False)
                row = np.sort(row)
                w[i,:l] = row
                w[i, l] = self.eos
            return torch.tensor(w)
    
    def tokenize(self, w, device='cpu'):
        if hasattr(self, 'device'):
            device = self.device
        return tu.from_strings(w, chars=self.characters, device=device)
    
    def stringify(self, w, sep='', pad='', offset=False):
        """
        w: [batch_size, sequence_length]
        """
        if offset:
            w = self.get_w_offset(w)
        return tu.to_strings(w, chars=self.characters, min_value=2, sep=sep, pad=pad)
    
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
    
    def mask_random_tokens(self, w, p_mask):
        """
        Zeroes out random tokens in w, with probability p_mask.
        w: [..., sequence_length]
        returns: [..., sequence_length]
        """
        mask = ((w == self.eos) | (torch.rand(w.shape, device=w.device) > p_mask))
        return w.masked_fill(~mask, self.pad)
    
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

    def get_w_offset(self, w):
        length = w.shape[-1]
        w = w.view(-1, length)
        position = self.vocab_size * torch.arange(length, device=w.device)
        mask = w > self.eos
        w = ~mask * w + mask * (position + w)
        return w
    

class BoWDiscretizeModule(nn.Module):
    """
    Abstract class for modules that use discrete tokens sequences.
    """

    def __init__(self, vocab_size: int, group_size=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.group_size = group_size
        
    @property
    def num_tokens(self):
        return self.vocab_size
    
    @property
    def characters(self):
        return string.ascii_letters
    
    def to_token_sequence(self, w):
        """
        Turns a multihot representation into a token sequence, e.g. [0, 1, 0, 1, 0] -> [2, 4, 0]
        Note: Uses 0 as the padding token.

        w: [batch_size, vocab_size]
        returns: [batch_size, sequence_length]
        """
        w = w.bool()
        a = 1 + torch.arange(self.vocab_size, device=w.device)
        a = a.unsqueeze(0) * w
        a = a.masked_fill(~w, self.vocab_size+1).sort(dim=-1)[0]
        a = a.masked_fill(a == self.vocab_size+1, 0)
        
        all_pad = (a == 0).all(0)
        if not all_pad.any():
            return a
        first_pad = (a == 0).all(0).nonzero()[0,0]
        return a[:,:first_pad]
    
    def stringify(self, w, sep=''):
        """
        w: [batch_size, vocab_size]
        """
        w = w.tolist()
        return [sep.join([self.characters[i] for i, x in enumerate(row) if x]) for row in w]
    
    # def sample_from_prior(self, n, length, dtype=bool, seed=None, device='cpu'):
    #     rng = np.random.default_rng(seed)
    #     tokens = np.arange(self.vocab_size)
    #     indices = np.array([rng.choice(tokens, size=length, replace=False) for _ in range(n)])
    #     x = np.zeros((n, self.vocab_size), dtype=bool)
    #     x[np.arange(n)[:, None], indices] = True
    #     return torch.tensor(x, dtype=dtype, device=device)

    def sample_from_prior(self, n, length, dtype=bool, seed=None, device='cpu'):
        rng = np.random.default_rng(seed)
        if self.group_size is not None:
            tokens = rng.integers(0, self.group_size, (n, length))
            offsets = np.arange(self.vocab_size // self.group_size)
            offsets = self.group_size * np.array([rng.choice(offsets, size=length, replace=False) for _ in range(n)])
            tokens += offsets
            tokens = torch.tensor(tokens, device=device)
            tokens = F.one_hot(tokens, self.vocab_size).to(dtype=dtype).sum(1)
            return tokens
        else:
            indices = np.array([rng.choice(tokens, size=length, replace=False) for _ in range(n)])
            tokens = np.arange(self.vocab_size)
            x = np.zeros((n, self.vocab_size), dtype=bool)
            x[np.arange(n)[:, None], indices] = True
            return torch.tensor(x, dtype=dtype, device=device)
    
    def sample_substrings(self, w, min_length=1):
        """
        For each w, repeatly samples a random subset of w until the length is at least min_length.
        Also returns the indices of the original w that correspond to the sampled substring.

        w: [batch_size, vocab_size]
        returns: 
            sub_w: [n, vocab_size]
            src_indices: [n]
        """
        src_indices = [torch.arange(len(w), device=w.device)]
        all_w = [w.clone()]
        done = w.sum(-1) <= min_length
        while not done.all() > 0:
            w = all_w[-1][~done].clone()
            src_idx = src_indices[-1][~done]
            indices = Categorical(w).sample()
            w[torch.arange(len(w)), indices] = 0
            done = w.sum(-1) <= min_length
            all_w.append(w)
            src_indices.append(src_idx)
        all_w = torch.cat(all_w, dim=0)
        src_indices = torch.cat(src_indices, dim=0)
        return all_w, src_indices


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
    
    # def sample(self, x, temperature=1, p_explore=0, argmax=False, target=None):
    #     """
    #     x: tensor with shape [batch_size, ...]
    #     returns:
    #         w_seq: tensor with shape [batch_size, max_length+1]
    #         logp_w: tensor with shape [batch_size]
    #     """
    #     batch_size = x.shape[0]
    #     w_seq = torch.zeros(batch_size, 1+self.max_length, dtype=torch.long, device=self.device)
    #     logp_w = torch.zeros(batch_size, device=self.device)
    #     cache = {}
    #     done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
    #     w_embeddings = einops.repeat(self.sos, 'h -> b 1 h', b=batch_size)
    #     for i in range(self.max_length):
    #         logits, cache = self.get_wi_logits(x[~done], w_embeddings, i, _cache=cache)
    #         logits = logits / temperature
    #         logits[:,0] = -1e8 # pad
    #         log_probs = logits.log_softmax(-1)
    #         if target is not None:
    #             w = target[:,i]
    #         elif argmax:
    #             w = log_probs.argmax(-1)
    #         else:
    #             sample_probs = (1-p_explore) * log_probs.exp() + p_explore / (1 + self.vocab_size)
    #             sample_probs[:,0] = 0 # pad
    #             w = Categorical(sample_probs).sample()
        
    #         logp = log_probs.gather(-1, w.unsqueeze(-1)).squeeze(-1)
    #         logp_w[~done] += logp
    #         w_seq[~done, i] = w
    #         terminate = (w == self.eos)
    #         done[~done] = done[~done] | terminate
    #         w_embeddings = torch.cat([w_embeddings[~terminate], self.embedding(w[~terminate]).unsqueeze(1)], dim=1)
    #         cache = {k: v[~terminate] if isinstance(v, torch.Tensor) else v for k, v in cache.items()}
    #         if done.all():
    #             break
    #     w_seq[~done, self.max_length] = self.eos
    #     return w_seq, logp_w

    def sample(self, x, min_length=1, max_length=None, temperature=1, p_explore=0, argmax=False, target=None):
        """
        x: tensor with shape [batch_size, ...]
        returns:
            w_seq: tensor with shape [batch_size, max_length+1]
            logp_w: tensor with shape [batch_size]
        """
        if max_length is None:
            max_length = self.max_length
        batch_size = x.shape[0]
        w_seq = torch.zeros(batch_size, 1+max_length, dtype=torch.long, device=self.device)
        logp_w = torch.zeros(batch_size, device=self.device)
        cache = {}
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        w_embeddings = einops.repeat(self.sos, 'h -> b 1 h', b=batch_size)
        for i in range(max_length):
            logits, cache = self.get_wi_logits(x[~done], w_embeddings, i, _cache=cache)
            logits = logits / temperature
            logits[:,0] = -1e8 # pad
            if i < min_length - 1:
                logits[:,self.eos] = -1e8
            log_probs = logits.log_softmax(-1)
            if target is not None:
                w = target[~done,i]
            elif argmax:
                w = log_probs.argmax(-1)
            else:
                sample_probs = (1-p_explore) * log_probs.exp() + p_explore / (1 + self.vocab_size)
                sample_probs[:,0] = 0 # pad
                if i < min_length - 1:
                    sample_probs[:,self.eos] = 0
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
        w_seq[~done, max_length] = self.eos
        return w_seq, logp_w
    
    # def sample_terminate_every_step(self, x, temperature=1, p_explore=0):
    #     """
    #     Used for training with terminate at every step.

    #     x: image tensor with shape [batch_size, ...]
    #     returns:
    #         w: tensor with shape [batch_size, max_length+1, max_length+1]
    #         logpf: tensor with shape [batch_size, max_length+1]
    #         logpt: tensor with shape [batch_size, max_length+1]
    #     """
    #     batch_size = x.shape[0]

    #     w_seq = torch.zeros(batch_size, 1+self.max_length, dtype=torch.long, device=self.device)
    #     logpf = torch.zeros(batch_size, 1+self.max_length, device=self.device)
    #     logpt = torch.zeros(batch_size, 1+self.max_length, device=self.device)
    #     cache = {}
    #     done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
    #     w_embeddings = einops.repeat(self.sos, 'h -> b 1 h', b=batch_size)
    #     for i in range(self.max_length):
    #         logits, cache = self.get_wi_logits(x[~done], w_embeddings, i, _cache=cache)
    #         logits = logits / temperature
    #         logits[:,0] = -1e8 # pad
    #         log_probs = logits.log_softmax(-1)

    #         sample_probs = (1-p_explore) * log_probs.exp() + p_explore / (1 + self.vocab_size)
    #         sample_probs[:,:2] = 0 # pad and eos
    #         w = Categorical(sample_probs).sample()
    
    #         logpw = log_probs.gather(-1, w.unsqueeze(-1)).squeeze(-1)
    #         logpf[~done, i] += logpw
    #         logpt[~done, i] += log_probs[:,self.eos]
    #         w_seq[~done, i] = w
    #         terminate = (w == self.eos)
    #         done[~done] = done[~done] | terminate
    #         w_embeddings = torch.cat([w_embeddings[~terminate], self.embedding(w[~terminate]).unsqueeze(1)], dim=1)
    #         cache = {k: v[~terminate] if isinstance(v, torch.Tensor) else v for k, v in cache.items()}
    #         if done.all():
    #             break
    #     w_seq[~done, self.max_length] = self.eos

    #     w = torch.zeros(w_seq.shape[0], w_seq.shape[1], w_seq.shape[1], dtype=int, device=w_seq.device)
    #     w[:, range(w_seq.shape[1]), range(w_seq.shape[1])] = 1
    #     for i in range(1, w_seq.shape[1]):
    #         w[:, i, :i] = w_seq[:, :i]
    #     return w, logpf, logpt
    
    def sample_terminate_every_step(self, x, min_length=1, max_length=None, temperature=1, p_explore=0):
        """
        Used for training with terminate at every step.

        x: image tensor with shape [batch_size, ...]
        returns:
            w: tensor with shape [batch_size, max_length+1, max_length+1]
            logpf: tensor with shape [batch_size, max_length+1]
            logpt: tensor with shape [batch_size, max_length+1]
        """
        if max_length is None:
            max_length = self.max_length
        batch_size = x.shape[0]

        w_seq = torch.zeros(batch_size, 1+max_length, dtype=torch.long, device=self.device)
        logpf = torch.zeros(batch_size, 1+max_length, device=self.device)
        logpt = torch.zeros(batch_size, 1+max_length, device=self.device)
        cache = {}
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        w_embeddings = einops.repeat(self.sos, 'h -> b 1 h', b=batch_size)
        for i in range(max_length):
            logits, cache = self.get_wi_logits(x[~done], w_embeddings, i, _cache=cache)
            logits = logits / temperature
            logits[:,0] = -1e8 # pad
            if i < min_length - 1:
                logits[:,self.eos] = -1e8
            log_probs = logits.log_softmax(-1)

            sample_probs = (1-p_explore) * log_probs.exp() + p_explore / (1 + self.vocab_size)
            sample_probs[:,0] = 0 # pad
            if i < min_length - 1:
                sample_probs[:,self.eos] = 0
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
        w_seq[~done, max_length] = self.eos

        w = torch.zeros(w_seq.shape[0], w_seq.shape[1], w_seq.shape[1], dtype=int, device=w_seq.device)
        w[:, range(w_seq.shape[1]), range(w_seq.shape[1])] = 1
        for i in range(1, w_seq.shape[1]):
            w[:, i, :i] = w_seq[:, :i]
        return w, logpf, logpt


class MLPDiscretizer(Discretizer):

    def __init__(self, vocab_size: int, length: int, dim_input: int,
                 dim_h=256, characters=string.ascii_letters, num_layers=2, nonlinearity=nn.ReLU(), 
                 allow_pad_outputs=False, **kwargs):
        super().__init__(vocab_size=vocab_size, max_length=length, dim_h=dim_h, characters=characters)
        self.allow_pad_outputs = allow_pad_outputs
        self.dim_input = dim_input
        self.length = length
        self.mlp = MLP(dim_input, length * (allow_pad_outputs + self.vocab_size), hidden_dim=dim_h, n_layers=num_layers, nonlinearity=nonlinearity)

    def sample(self, z, temperature=1, p_explore=0, argmax=False, target=None, disallow_pad=False):
        """
        z: [batch_size, dim_z]
        returns
            w: [batch_size, 1+length]
            logp: [batch_size]
        """
        batch_size = z.shape[0]
        logits = self.mlp(z).view(batch_size, self.length, -1) / temperature
        if self.allow_pad_outputs and disallow_pad:
            sample_logits = logits.clone()
            sample_logits[:,:,0] = -1e8
        else:
            sample_logits = logits
        if target is not None:
            if self.allow_pad_outputs:
                w = (target[:,:-1] - 1).clamp(0) # drop EOS and drop indices by 1
            else:
                w = target[:,:-1] - 2 # drop EOS and drop indices by 2
        elif argmax:
            w = sample_logits.argmax(-1)
        else:
            probs = (1-p_explore) * sample_logits.softmax(-1) + p_explore / self.vocab_size
            if self.allow_pad_outputs and disallow_pad:
                probs[:,:,0] = 0
            w = Categorical(probs).sample()
        
        logp = Categorical(logits=logits).log_prob(w).sum(-1)
        if self.allow_pad_outputs:
            w = (w > 0) * (w + 1)
        else:
            w = 2 + w
        w = F.pad(w, (0, 1), value=self.eos)
        return w, logp
    
    def get_logits(self, z, w_embedding):
        """
        z: [batch_size, dim_z]
        w_embedding: [batch_size, 1+length, dim_h]
        returns: [batch_size, 1+length, num_tokens]
        """
        logits = self.mlp(z).view(z.shape[0], self.length, -1)
        if self.allow_pad_outputs:
            eos_logits = torch.full_like(logits[:,:,:1], -1e8)
            logits = torch.cat([logits[:,:,:1], eos_logits, logits[:,:,1:]], dim=-1)
        else:
            logits = F.pad(logits, (2, 0), value=-1e8)
        eos_logits = torch.zeros_like(logits[:,:1])
        eos_logits[:, :, self.eos] = 1e8
        logits = torch.cat([logits, eos_logits], dim=1)
        return logits
    

class TransformerDiscretizer(Discretizer):
    """
    Transformer-based discretizer that takes in a sequence of vectors and outputs a sequence of tokens.
    """

    def __init__(self, vocab_size: int, max_length: int, dim_input: int,
                 num_layers=2, dim_feedforward=512, num_heads=8, dropout=0.,
                 characters=string.ascii_letters, **kwargs):
        super().__init__(vocab_size=vocab_size, max_length=max_length, dim_h=dim_input, characters=characters)
        self.dim_input = dim_input

        # self.positional_embedding = nn.Parameter(torch.randn(max_length, dim_input))
        self.positional_encoding = PositionalEncoding(dim_input, concat=False)
        layers = nn.TransformerDecoderLayer(dim_input, nhead=num_heads, dim_feedforward=dim_feedforward,
                                            norm_first=True, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(layers, num_layers=num_layers)
        self.token_decoder = nn.Linear(dim_input, self.vocab_size)
        self.register_buffer('causal_mask', nn.Transformer.generate_square_subsequent_mask(1+max_length))

    def get_wi_logits(self, x, w_embedding, i: int, _cache=None):
        """
        x: [batch_size, input_length, dim_h]
        w_embedding: [batch_size, i+1, dim_h]
            First token is SOS

        Returns logits of shape [batch_size, num_tokens]
        cache: a dict of tensors
        """
        length = w_embedding.shape[1]
        w_embedding = self.positional_encoding(w_embedding)
        h = self.transformer(w_embedding, x, tgt_mask=self.causal_mask[:length,:length])
        logits = self.token_decoder(h[:,-1])
        return logits, {}


class SimpleTransformerDiscretizer(Discretizer):
    """
    Transformer-based discretizer that takes in a sequence of vectors and outputs a sequence of tokens.
    The output length is fixed and the tokens are sampled simultaneously (no dependence on previous tokens).
    """

    def __init__(self, vocab_size: int, length: int, dim_input: int,
                 num_layers=2, dim_feedforward=512, num_heads=8, dropout=0.,
                 allow_pad=False,
                 characters=string.ascii_letters, **kwargs):
        super().__init__(vocab_size=vocab_size, max_length=length, dim_h=dim_input, characters=characters)
        self.dim_input = dim_input
        self.length = length
        self.allow_pad = allow_pad

        self.positional_embedding = nn.Parameter(torch.randn(length, dim_input))
        layers = nn.TransformerDecoderLayer(dim_input, nhead=num_heads, dim_feedforward=dim_feedforward,
                                            norm_first=True, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(layers, num_layers=num_layers)
        self.token_decoder = nn.Linear(dim_input, allow_pad+self.vocab_size)
        self.register_buffer('mask', ~torch.eye(length, dtype=torch.bool))

    def sample(self, h, n=1, temperature=1, p_explore=0, argmax=False, target=None):
        """
        h: [batch_size, dim_input] or [batch_size, k, dim_input]
        target: [batch_size, length]
        returns
            w: [batch_size, 1+length]
            logp: [batch_size]
        """
        if h.ndim == 2:
            h = h.unsqueeze(1)
        batch_size = h.shape[0]

        queries = einops.repeat(self.positional_embedding, 'l h -> b l h', b=batch_size)
        h = self.transformer(queries, h, tgt_mask=self.mask)
        logits = self.token_decoder(h) / temperature
        if target is not None:
            w = target[:,:-1] - 2 # drop EOS and drop indices by 2
            w = w.unsqueeze(0)
        elif argmax:
            assert n == 1
            w = logits.argmax(-1)
            w = w.unsqueeze(0)
        else:
            probs = (1-p_explore) * logits.softmax(-1) + p_explore / self.vocab_size
            w = Categorical(probs).sample((n, ))
        
        logp = Categorical(logits=logits).log_prob(w).sum(-1)
        if self.allow_pad:
            w = w + (w > 0)
        else:
            w = w + 2
        w = F.pad(w, (0, 1), value=self.eos)

        w = w.transpose(0, 1).contiguous().squeeze(1)
        logp = logp.transpose(0, 1).contiguous().squeeze(1)
        return w, logp
    

# class SimpleTransformerDiscretizer(Discretizer):
#     """
#     Transformer-based discretizer that takes in a sequence of vectors and outputs a sequence of tokens.
#     The output length is fixed and the tokens are sampled simultaneously (no dependence on previous tokens).
#     """

#     def __init__(self, vocab_size: int, length: int, dim_input: int,
#                  num_layers=2, dim_feedforward=512, num_heads=8, dropout=0.,
#                  characters=string.ascii_letters, **kwargs):
#         super().__init__(vocab_size=vocab_size, max_length=length, dim_h=dim_input, characters=characters)
#         self.dim_input = dim_input
#         self.length = length

#         self.positional_embedding = nn.Parameter(torch.randn(length, dim_input))
#         layers = nn.TransformerDecoderLayer(dim_input, nhead=num_heads, dim_feedforward=dim_feedforward,
#                                             norm_first=True, dropout=dropout)
#         self.transformer = nn.TransformerDecoder(layers, num_layers=num_layers)
#         self.token_decoder = nn.Linear(dim_input, self.vocab_size)

#     def sample(self, h, n=1, temperature=1, p_explore=0, argmax=False, target=None):
#         """
#         h: [batch_size, dim_input] or [batch_size, k, dim_input]
#         returns
#             w: [batch_size, 1+length]
#             logp: [batch_size]
#         """
#         if h.ndim == 2:
#             h = h.unsqueeze(1)
#         batch_size = h.shape[0]

#         queries = einops.repeat(self.positional_embedding, 'l h -> b l h', b=batch_size)
#         h = self.transformer(queries, h)
#         logits = self.token_decoder(h) / temperature
#         if target is not None:
#             w = target[:,:-1] - 2 # drop EOS and drop indices by 2
#         elif argmax:
#             w = logits.argmax(-1)
#         else:
#             probs = (1-p_explore) * logits.softmax(-1) + p_explore / self.vocab_size
#             w = Categorical(probs).sample(n).squeeze(1)
        
#         logp = Categorical(logits=logits).log_prob(w).sum(-1)
#         w = F.pad(2 + w, (0, 1), value=self.eos)
#         return w, logp
    
    

def sample_w(p_length, vocab_size, n):
    """
    p_length: tensor with shape [max_length]
    Assumes that pad=0 and eos=1
    """
    length = Categorical(p_length).sample((n, ))
    max_length = length.max()
    w = 2 + torch.randint(0, vocab_size, (n, 1 + max_length), device=p_length.device)
    length = length.unsqueeze(1)
    w = w.scatter(1, length, torch.ones_like(length, device=p_length.device))
    mask = torch.arange(w.shape[-1], device=w.device) > length
    w = w.masked_fill(mask, 0)
    return w
