import string
import torch
from torch.distributions import Normal
import torch.nn.functional as F
import einops

from .utils import extract_args


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


def to_strings(array, chars=None, sep='', min_value=0):
    """
    array: a 2 dimensional array of integers
    """
    strings = []
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    for s in array:
        s = sep.join([chars[i-min_value] if chars is not None else str(i-min_value) for i in s if i >= min_value])
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
