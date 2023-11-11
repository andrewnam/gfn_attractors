import sys
import inspect
from collections.abc import Iterable
import numpy as np


_use_torch = 'torch' in sys.modules
if _use_torch:
    import torch


def kv_str(_delim=" | ", _digits=3, **kwargs):
    s = []
    for k, v in kwargs.items():
        if _use_torch and isinstance(v, torch.Tensor):
            if len(v.shape) == 0:
                v = v.item()
            else:
                v = v.detach().cpu()
        if isinstance(v, float):
            v = round(v, _digits)
        s.append("{}: {}".format(k, v))
    s = _delim.join(s)
    return s


def kv_print(_delim=" | ", _digits=3, **kwargs):
    """
    Pretty-prints kwargs

    :param _delim: Delimiter to separate kwargs
    :param _digits: number of decimal digits to round to
    :param kwargs: stuff to print
    :return:
    """
    print(kv_str(_delim, _digits=_digits, **kwargs))


def is_iterable(obj, allow_str=False):
    if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
        return len(obj.shape) > 0
    if allow_str:
        return isinstance(obj, Iterable)
    else:
        return isinstance(obj, Iterable) and not isinstance(obj, str)


def extract_args(args):
    """
    Use when *args is used as a function parameter.
    Allows both an iterable and a sequence of parameters to be passed in.
    For example, if f([1, 2, 3]) and f(1, 2, 3) will be valid inputs to the following function
        def f(*args):
            args = extract_args(args)
            ...
    @param args:
    @return:
    """
    if len(args) == 1 and is_iterable(args[0]):
        return args[0]
    return args


def filter_kwargs(fn, kwargs):
    """
    Filters out kwargs that are not in the signature of fn
    """
    return {k: v for k, v in kwargs.items() if k in inspect.signature(fn).parameters}


def cycle(iterator, n):
    """
    draws from an iterator n times, reinstantiating the iterator each time, rather than
    caching the results like itertools.cycle
    this is useful for DataLoader when using shuffle=True

    :param iterator:
    :param n:
    :return:
    """
    i = 0
    while i < n:
        for x in iterator:
            yield x
            i += 1
            if i >= n:
                return
