import sys

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
