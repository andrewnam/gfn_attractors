from typing import Any
from dataclasses import dataclass

from .utils import filter_kwargs


@dataclass
class Config:
    """
    Base class for configuration classes.
    Allows initializing from a dict that is a superset of the class attributes.
    """

    @classmethod
    def from_dict(cls, d: dict[str, Any], **kwargs):
        d.update(kwargs)
        return cls(**filter_kwargs(cls, d))
