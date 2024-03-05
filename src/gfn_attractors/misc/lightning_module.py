
from dataclasses import asdict
import pytorch_lightning as pl
import torch
import yaml
from datetime import datetime
from pathlib import Path
import os
import wandb
import numpy as np

from . import torch_utils as tu


class LightningModule(pl.LightningModule):

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.metrics = {}

    @classmethod
    def load(cls, data_module, config_class, filepath, config_path=None):
        directory = Path(filepath).parent
        config_path = config_path or directory / 'config.yaml'

        config = yaml.load(open(f'{directory}/config.yaml', 'r'), Loader=yaml.FullLoader)
        config = config_class(**config)
        model = cls(config, data_module)
        missing = tu.load_partial_state_dict(model, torch.load(filepath))    
        print(f"Loaded {filepath}")
        if len(missing) == 0:
            print("All keys found in the model.")
        else:
            print(f"{len(missing)} mismatched keys were found.")
        return model, missing

    @classmethod
    def load_latest(cls, config_class, save_dir):
        paths = list(Path(save_dir).glob('*.pt'))
        paths.sort(key=os.path.getmtime, reverse=True)
        latest_save = paths[0]
        return cls.load(config_class, latest_save)

    def save(self, save_path=None, save_dir=None, save_name=None):
        if save_path is not None:
            save_dir = Path(save_path).parent
            save_name = Path(save_path).name
        elif save_dir is None:
            raise ValueError("save_dir must be specified if save_path is not.")

        if save_name is None:
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f'{self.global_step}_{now}.pt'

        with open(f'{save_dir}/config.yaml', 'w') as f:
            f.write(yaml.dump(asdict(self.config)))
        torch.save(self.state_dict(), f'{save_dir}/{save_name}')

    def log_metrics(self, metrics: dict):
        self.metrics.update(metrics)
        self.log_dict(self.metrics, prog_bar=True)

    def log_gif(self, key, images):
        imarrays = np.array([np.transpose(np.array(im), (2, 0, 1)) for im in images])
        wandb.log({key: wandb.Video(imarrays, fps=1, format='gif', caption=f"{key} ({self.global_step})")}, commit=True)
