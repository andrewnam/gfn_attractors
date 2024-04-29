from typing import Any
from dataclasses import dataclass, fields, asdict
import yaml
from pathlib import Path


@dataclass
class Config:
    """
    Base class for configuration classes.
    Allows initializing from a dict that is a superset of the class attributes.
    """

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        d = dict(d)
        f = [f.name for f in fields(cls)]
        d = {k: v for k, v in d.items() if k in f}
        return cls(**d)

    @classmethod
    def load(cls, filepath: str):
        config = yaml.load(open(filepath, 'r'), Loader=yaml.FullLoader)
        return cls.from_dict(config)
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            f.write(yaml.dump(asdict(self)))
