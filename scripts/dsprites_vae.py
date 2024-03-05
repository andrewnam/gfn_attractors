import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import git
from pathlib import Path
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from dataclasses import asdict
from datetime import datetime

from gfn_attractors.data.dsprites import ContinuousDSpritesDataModule
from gfn_attractors.models.vae import VAE, VAEConfig


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    save_dir = f'/data2/pdp/ajhnam/gfn_attractors/dsprites_vae/'
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    data_module = ContinuousDSpritesDataModule(batch_size=args.batch_size, size=32, constant_orientation=True, min_scale=0,
                                               holdout_xy_mode=args.xy_mode, holdout_xy_nonmode=args.xy_nonmode,
                                               holdout_xy_shape=args.xy_shape, holdout_xy_mode_color=args.xy_mode_color,
                                               holdout_shape_color=args.shape_color, seed=args.seed)
    data_module.prepare_data()

    batch = next(iter(data_module.train_dataloader()))
    print(batch['image'].shape)

    config = VAEConfig(dim_z=args.dim_z, 
                       dim_h=args.dim_h, 
                       num_encoder_layers=args.num_encoder_layers, 
                       num_decoder_layers=args.num_decoder_layers, 
                       dim_feedforward=args.dim_feedforward, 
                       mse_weight=args.mse_weight, 
                       bce_weight=args.bce_weight, 
                       lr=args.lr)
    model = VAE(config, data_module)
    
    if args.wandb:
        logger = WandbLogger(project='dsprites_vae',
                            name=f'{run_name}',
                            entity='andrewnam',
                            #  tags=[experiment_name],
                            config=asdict(config))
    else:
        logger = None

    trainer = pl.Trainer(max_epochs=args.epochs, 
                         devices=[device],  
                         logger=logger,
                         num_sanity_val_steps=0,
                        #  check_val_every_n_epoch=53,
                        #  detect_anomaly=True, 
                         enable_progress_bar=True)
    trainer.fit(model, data_module.train_dataloader())

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f'{save_dir}/{run_name}_{now}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default='cpu', help="Device index")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--xy_mode", type=bool, default=False, help="Whether to hold out xy mode")
    parser.add_argument("--xy_nonmode", type=bool, default=False, help="Whether to hold out xy nonmode")
    parser.add_argument("--xy_shape", type=bool, default=False, help="Whether to hold out xy shape")
    parser.add_argument("--xy_mode_color", type=bool, default=False, help="Whether to hold out xy mode color")
    parser.add_argument("--shape_color", type=bool, default=False, help="Whether to hold out shape color")

    parser.add_argument("--dim_z", type=int, default=8, help="Dimension of latent space")
    parser.add_argument("--dim_h", type=int, default=128, help="Dimension of hidden layer")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Number of layers in encoder")
    parser.add_argument("--num_decoder_layers", type=int, default=2, help="Number of layers in decoder")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="Dimension of feedforward layer")
    parser.add_argument("--mse_weight", type=float, default=0, help="Weight of mse loss")
    parser.add_argument("--bce_weight", type=float, default=1, help="Weight of bce loss")

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wandb", type=bool, default=False, help="Whether to log to wandb")

    args = parser.parse_args()

    if args.xy_mode + args.xy_nonmode + args.xy_shape + args.xy_mode_color + args.shape_color > 1:
        raise ValueError("Only one holdout option can be True")
    if args.xy_mode:
        run_name = 'xy_mode'
    elif args.xy_nonmode:
        run_name = 'xy_nonmode'
    elif args.xy_shape:
        run_name = 'xy_shape'
    elif args.xy_mode_color:
        run_name = 'xy_mode_color'
    elif args.shape_color:
        run_name = 'shape_color'
    else:
        run_name = 'full'

    print(args)
    main(args)
