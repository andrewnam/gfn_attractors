from dataclasses import dataclass
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from gfn_attractors.misc.lightning_module import LightningModule
from gfn_attractors.models.images import TransformerImageEncoder, TransformerImageDecoder
from gfn_attractors.misc import Config


@dataclass
class VAEConfig(Config):

    dim_z: int
    dim_h: int = 256
    patch_size: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    nhead: int = 8
    dim_feedforward: int = 512

    lr: float = 1e-4
    mse_weight: float = 0
    bce_weight: float = 1


class VAE(LightningModule):

    def __init__(self, config, data_module):
        super().__init__(config)
        self.data_module = data_module

        self.encoder = TransformerImageEncoder(size=data_module.size,
                                               num_channels=data_module.num_channels,
                                               patch_size=config.patch_size,
                                               dim_encoding=config.dim_z,
                                               variational=True,
                                               dim_h=config.dim_h,
                                               nhead=config.nhead,
                                               dim_feedforward=config.dim_feedforward,
                                               num_layers=config.num_encoder_layers,
                                               dropout=0.)
        self.decoder = TransformerImageDecoder(size=data_module.size,
                                               num_channels=data_module.num_channels,
                                               patch_size=config.patch_size,
                                               dim_encoding=config.dim_z,
                                               dim_h=config.dim_h,
                                               nhead=config.nhead,
                                               dim_feedforward=config.dim_feedforward,
                                               num_layers=config.num_decoder_layers,
                                               dropout=0.)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

    def forward(self, x):
        z = self.encoder(x)
        logits = self.decoder(z)
        return logits

    def get_loss(self, x):
        z, mu, sigma = self.encoder(x, use_reparam=True, return_params=True)
        logits = self.decoder(z)
        
        mse_loss = F.mse_loss(logits.sigmoid(), x, reduction='none').sum((-3, -2, -1)).mean()
        bce_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='none').sum((-3, -2, -1)).mean()

        recon_loss = mse_loss * self.config.mse_weight + bce_loss * self.config.bce_weight
        dist = Normal(mu, sigma)
        prior_dist = Normal(0, 1)
        kl_div = torch.distributions.kl_divergence(dist, prior_dist).sum(-1).mean()
        loss = recon_loss + kl_div
        return loss, {'loss': loss.item(), 'kl_div': kl_div.item(), 'mse_loss': mse_loss.item(), 'bce_loss': bce_loss.item()}

    def training_step(self, batch, batch_idx):
        # print(batch['image'].shape)
        x = batch['image']
        loss, metrics = self.get_loss(x)
        self.log_metrics(metrics)
        return loss
