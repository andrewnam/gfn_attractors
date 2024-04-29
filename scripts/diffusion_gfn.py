import torch
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import torch
from dataclasses import asdict
import yaml
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from gfn_attractors.data.branching_diffusion import *
from gfn_attractors.misc import torch_utils as tu
from gfn_attractors.models.gfn_em import GFNEM, GFNEMConfig
from gfn_attractors.models.attractors_gfn_em import AttractorsGFNEM, AttractorsGFNEMConfig
from gfn_attractors.binary_vectors import *


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if args.dynamics:
        config = AttractorsGFNEMConfig.from_dict(config_dict)
    else:
        config = GFNEMConfig.from_dict(config_dict)
    print(f"Successfully loaded config from {args.config}")
    print(config)

    save_dir = f'/data2/pdp/ajhnam/gfn_attractors/diffusion/{args.run_name}'
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    config.save_dir = save_dir
    config.save(f'{save_dir}/config.yaml')
    with open(save_dir + '/args.yaml', 'w') as f:
        f.write(yaml.dump(args))
    print(f"Saved config to {save_dir}/config.yaml")

    data_module = BinarySplitDataModule(depth=config_dict['data_depth'], 
                                        repeat=config_dict['data_repeat'], 
                                        sample_ancestors=config_dict['data_sample_ancestors'],
                                        min_test_depth=config_dict['data_min_test_depth'],
                                        batch_size=config_dict['data_batch_size'],
                                        seed=args.seed)
    data_module.prepare_data()
    print(f"Created data module with {len(data_module)} samples")

    if args.dynamics:
        model = BinaryVectorAttractorsGFNEM(config, data_module)
    else:
        model = BinaryVectorGFNEM(config, data_module)
    
    if args.load:
        print(f"Loading model from {args.load}. Mismatching parameters:")
        print(tu.load_partial_state_dict(model, torch.load(args.load)))

    model.init_optimizers()
    model.to(device)

    print("Performing sanity check")
    model.sanity_test(plot=True)

    if args.test:
        return
    
    print("Training VAE")
    model.train_vae(config_dict['num_vae_updates'], config_dict['vae_batch_size'], 'vae.pt')

    print("Training GFN")
    name = f'{args.run_name}_' + ('dynamics' if args.dynamics else 'discretizer')
    if args.wandb:
        logger = WandbLogger(project='diffusion',
                                name=name,
                                entity='andrewnam',
                                #  tags=[experiment_name],
                                config={**config_dict, **vars(args)})
    else:
        logger = None

    trainer = pl.Trainer(max_epochs=args.epochs, 
                            devices=[device],  
                            logger=logger,
                            num_sanity_val_steps=0,
                            enable_progress_bar=True)
    trainer.fit(model, tu.DummyDataset(1000))

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f'{save_dir}/{args.run_name}_{now}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=bool, default=False, help="If True, runs test")
    parser.add_argument("--config", type=str, help="Filepath to config file")
    parser.add_argument("--run_name", type=str, help="Name of run")
    parser.add_argument("--load", type=str, default=None, help="Filepath to load model from")
    parser.add_argument("--dynamics", type=bool, help="If True, trains attractor dynamics. Else, discretizer only.")
    parser.add_argument("--device", type=int, help="Device index")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--wandb", type=bool, default=False, help="Whether to log to wandb")

    args = parser.parse_args()
    main(args)