import string
import torch
from torch.distributions import Normal, Bernoulli
from torch.utils.data import RandomSampler, WeightedRandomSampler
import torch.nn.functional as F
import einops
import pandas as pd
import numpy as np
from torch.utils.data.dataset import IterableDataset

from .utils import extract_args


class DummyDataset(IterableDataset):
    
    def __init__(self, size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __iter__(self):
        for i in range(self.size):
            yield 0


class RandomSampler2(RandomSampler):
    """
    The standard RandomSampler returns indices in range(len(data_source)),
        e.g. RandomSampler([2, 4, 6, 8]) will still sample from [0, 1, 2, 3].
    This class returns the actual data instead.
    """

    def __iter__(self):
        for i in super().__iter__():
            yield self.data_source[i]

def load_partial_state_dict(model, state_dict, verbose=False):
    """
    Loads a state dict into a model, ignoring any keys that don't match.
    Returns a list of keys that were not loaded.
    """
    model_sd = model.state_dict()
    mismatch = []
    for k in sorted(model_sd.keys() | state_dict.keys()):
        if k not in model_sd:
            if verbose:
                print(f"Missing in model: {k}")
            del state_dict[k]
            mismatch.append(k)
        elif k not in state_dict:
            if verbose:
                print(f"Missing in loaded: {k}")
            state_dict[k] = model_sd[k]
            mismatch.append(k)
        elif model_sd[k].shape != state_dict[k].shape:
            if verbose:
                print(f"Shape mismatch: {k}")
            state_dict[k] = model_sd[k]
            mismatch.append(k)
    model.load_state_dict(state_dict)
    return mismatch


def prepend_shape(tensor, *shape):
    shape = extract_args(shape)
    dims = {a: i for a, i in zip(string.ascii_lowercase, shape)}
    letters = ' '.join(string.ascii_lowercase[:len(shape)])
    return einops.repeat(tensor, f'... -> {letters} ...', **dims)


def to_strings(array, chars=None, sep='', min_value=0, pad=''):
    """
    array: a vector or 2 dimensional array of integers
    """
    strings = []
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    if array.ndim == 1:
        return sep.join([(chars[i-min_value] if chars is not None else str(i-min_value)) if i >= min_value else pad for i in array])
    for s in array:
        s = sep.join([(chars[i-min_value] if chars is not None else str(i-min_value)) if i >= min_value else pad for i in s])
        # s = sep.join([chars[i-min_value] if chars is not None else str(i-min_value) for i in s if i >= min_value])
        strings.append(s)
    return strings


def from_strings(strings, chars=string.ascii_letters, pad_token='$', pad=0, eos: int|None = 1,  min_value=2, device='cpu'):
    """
    array: a 2 dimensional array of integers
    """
    tokens = []
    max_length = max(len(s) for s in strings)
    indices = {c: i + min_value for i, c in enumerate(chars)} if chars is not None else range(min_value, max(strings))
    indices[pad_token] = pad
    for s in strings:
        s = [indices[c] for c in s]
        if eos is None:
            s = s + [pad] * (max_length - len(s))
        else:
            s = s + [eos] + [pad] * (max_length - len(s))
        tokens.append(s)
    return torch.tensor(tokens, device=device)


def batch_cross_entropy(input, target, *args, **kwargs):
    """
    Wrapper for cross_entropy loss so that the dimensions are more intuitive.
    Permutes the dimensions so that the last dim of input corresponds to number of classes.

    input: [batch size, ..., num classes]
    target: [batch size, ...]
    """
    dims = [0, -1] + list(range(1, len(input.shape) - 1))
    input = input.permute(dims)
    return F.cross_entropy(input, target, *args, **kwargs)


def get_kl_div(mu, logsigma):
    sigma = logsigma.exp()
    dist1 = Normal(mu, sigma)
    dist2 = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    return torch.distributions.kl_divergence(dist1, dist2).sum(-1)

def correct_trajectory(z_traj, z_hat, beta=1.0):
    """
    Apply correction to a trajectory to ensure it ends at z_hat.
    
    :param z_traj: Tensor of shape (batch_size, time, dim_z) representing the original trajectories.
    :param z_hat: Tensor of shape (batch_size, dim_z) representing the desired end points.
    :param beta: A parameter controlling the rate of correction distribution. 
        When beta = 0, the correction is linear. When beta > 0, the correction is exponential such that later steps are adjusted more aggressively.
    :return: Corrected trajectory tensor with shape (batch_size, time, dim_z).
    """
    num_steps = z_traj.shape[1]

    # Calculate correction vectors for each index
    correction_vectors = z_hat - z_traj[:, -1, :]

    # Create a scaling factor tensor that varies along the time dimension
    # Linear or exponential scaling can be used depending on the requirement
    if isinstance(z_traj, torch.Tensor):
        time_scale = torch.arange(num_steps, dtype=torch.float32, device=z_traj.device) / (num_steps - 1)
        scaling_factors = time_scale * np.exp(-beta) + torch.exp(beta * time_scale) - 1
    else: # numpy
        time_scale = np.arange(num_steps, dtype=np.float32) / (num_steps - 1)
        scaling_factors = time_scale * np.exp(-beta) + np.exp(beta * time_scale) - 1
    scaling_factors = scaling_factors / scaling_factors[-1]
    scaling_factors = scaling_factors.view(1, -1, 1)

    # Apply the correction
    corrected_traj = z_traj + scaling_factors * correction_vectors.reshape(-1, 1, z_traj.shape[-1])
    return corrected_traj

def sample_brownian_bridge(num_samples, num_steps, a=0, b=0, sd=1):
    """
    Sample a Brownian bridge from a to b with num_steps time steps.
    Returns a numpy array of shape (num_samples, num_steps).
    """
    B = np.empty((num_samples, num_steps), dtype=np.float32)
    B[:, 0] = 0
    B[:, -1] = 0

    for t in range(num_steps-2):
        noise = np.random.randn(num_samples) * sd
        B[:, t+1] = B[:, t] * (num_steps - t - 2) / (num_steps - t - 1) + noise
    
    B += np.linspace(a, b, num_steps)
    return B

def sample_brownian_bridge_nd(a, b, num_steps, sd):
    """
    Samples a brownian bridge connecting a and b with num_steps time steps.
    a: (num_samples, num_dims)
    b: (num_samples, num_dims)
    sd: scalar or (num_samples, num_dims)
    returns: (num_samples, num_steps, num_dims)
    """
    if torch.is_tensor(sd):
        sd = sd.flatten().cpu().numpy()
    B = sample_brownian_bridge(a.numel(), num_steps, sd=sd)
    B = torch.tensor(B, device=a.device).view(a.shape[0], num_steps, a.shape[1])
    
    dx = torch.linspace(0, 1, num_steps, device=a.device)
    dx = a.unsqueeze(1) + (b - a).unsqueeze(1) * dx.view(1, -1, 1)
    return B + dx


def unravel_index(indices: torch.Tensor, shape: torch.Size):
    r"""Converts a tensor of flat indices into a tensor of coordinate vectors.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.

    Returns:
        The unraveled coordinates, (*, D).
    """

    shape = indices.new_tensor((*shape, 1))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]

def sample_substrings(w, min_value):
    """
    For each w, randomly 0s out tokens so that the first sub_w is all 0s, the second has an average of 1 non-zero token,
        the third sub_w has an average of 2 non-zero tokens, etc.
        The last sub_w is w itself.
    w: (batch_size, length)
    returns: (batch_size, length, length)
    """
    length = w.shape[-1]
    mask = Bernoulli(probs=(torch.arange(length, device=w.device)) / length).sample(w.shape).bool()
    mask = mask.transpose(1, 2)
    mask = mask | (w.unsqueeze(1) < min_value)
    masked_w = mask * w.unsqueeze(1)
    return masked_w

def nCk(n, k, log=False):
    """
    Compute the binomial coefficient n choose k.
    """
    log_nck = (n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()
    if log:
        return log_nck
    return log_nck.exp()


def to_long_df(array, dim_names, value_name: str|list = 'value', **kwargs):
    """
    If value_name is a list, then the length of the list must be the same as the last dimension in array.
    """
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    shape = array.shape
    if isinstance(value_name, str):
        array = array.flatten()
        index = pd.MultiIndex.from_product([range(i) for i in shape], names=dim_names)
        df = pd.DataFrame(array, columns=[value_name], index=index).reset_index()
    else:
        array = array.reshape(-1, len(value_name))
        index = pd.MultiIndex.from_product([range(i) for i in shape[:-1]], names=dim_names)
        df = pd.DataFrame(array, columns=value_name, index=index).reset_index()
    for k, v in kwargs.items():
        i = len(v.shape)
        v = v.flatten()
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        while len(v) < len(df):
            v = np.repeat(v, shape[i])
            i += 1
        df[k] = v
    return df


def unique(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim, 
        sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, counts, index