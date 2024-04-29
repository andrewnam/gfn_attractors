from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import einops

from .helpers import MLP
from .discretizer import BoWDiscretizeModule
from .discriminator import Discriminator
from .images import TransformerImageEncoder, TransformerImageDecoder


@dataclass
class ScoreResult:

    score: torch.Tensor # [batch_size, ...]
    logpz_zhat: torch.Tensor # [batch_size, ...]    

    z0: torch.Tensor|None = None# [batch_size, dim_z]
    z0_sd: torch.Tensor|None = None# [batch_size, dim_z]
    z_hat: torch.Tensor|None = None # [batch_size, ..., dim_z]
    z_hat_sd: torch.Tensor|None = None # [batch_size, ..., dim_z]

class MModel(BoWDiscretizeModule, nn.Module, ABC):

    def __init__(self, 
                 dim_z, 
                 vocab_size, 
                 vocab_group_size=None,
                 dim_h=256,
                 num_layers=3,
                 num_w_embedding_layers=1,
                 vae_beta=1.,
                 cvae_beta=1.):
        BoWDiscretizeModule.__init__(self, vocab_size)
        nn.Module.__init__(self)
        self.dim_z = dim_z
        self.vocab_group_size = vocab_group_size
        self.dim_h = dim_h
        self.num_layers = num_layers
        self.num_w_embedding_layers = num_w_embedding_layers
        self.vae_beta = vae_beta
        self.cvae_beta = cvae_beta

        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()
        self.z_sd_model = nn.Sequential(MLP(dim_z, dim_z, dim_h, n_layers=num_layers, nonlinearity=nn.ReLU()), nn.Sigmoid())
        self.zhat_model = MLP(vocab_size, dim_z, dim_h, n_layers=num_w_embedding_layers, nonlinearity=nn.ReLU())
        self.zhat_sd_model = nn.Sequential(MLP(vocab_size, dim_z, dim_h, n_layers=num_layers, nonlinearity=nn.ReLU()), nn.Sigmoid())
        self.freeze_encoder()

    @property
    def device(self):
        return self.zhat_model.device

    @abstractmethod
    def init_encoder(self):
        raise NotImplementedError

    @abstractmethod
    def init_decoder(self):
        raise NotImplementedError

    @abstractmethod
    def get_recon_loss(self, x, z):
        """
        x: tensor with shape [batch_size, *input_shape]
        z: tensor with shape [batch_size, dim_z]
        returns: scalar and metrics dict
        """
        raise NotImplementedError
    
    def freeze_encoder(self):
        self.frozen_encoder = self.init_encoder().to(self.device)
        self.frozen_encoder.load_state_dict(self.encoder.state_dict())
        self.frozen_encoder.eval()
        self.frozen_encoder.requires_grad_(False)

    def encode(self, x, encoder):
        return encoder(x)
    
    def get_z0(self, x, frozen=True):
        if frozen and self.frozen_encoder is None:
            self.freeze_encoder()
        encoder = self.frozen_encoder if frozen else self.encoder
        return self.encode(x, encoder)
    
    def get_z_hat(self, w, return_sd=False):
        z_hat = self.zhat_model(w)
        if return_sd:
            sd = self.zhat_sd_model(w)
            return z_hat, sd
        return z_hat
    
    def get_logpz_zhat(self, z, z_hat, z_hat_sd):
        """
        1.       z: [batch_size, dim_z]
             z_hat: [batch_size, dim_z]
          z_hat_sd: [batch_size, dim_z]
           returns: [batch_size]
        2.       z: [batch_size, dim_z]
             z_hat: [batch_size, k, dim_z]
          z_hat_sd: [batch_size, k, dim_z]
           returns: [batch_size, k]
        3.       z: [batch_size, num_steps, dim_z]
             z_hat: [batch_size, num_steps, dim_z]
          z_hat_sd: [batch_size, num_steps, dim_z]
           returns: [batch_size, num_steps]
        4.       z: [batch_size, num_steps, dim_z]
             z_hat: [batch_size, num_steps, k, dim_z]
          z_hat_sd: [batch_size, num_steps, k, dim_z]
           returns: [batch_size, num_steps, k]
        """
        if z.ndim == 2 and z_hat.ndim == 3:
            z = einops.repeat(z, 'b z -> b k z', k=z_hat.shape[1])
        elif z.ndim == 3 and z_hat.ndim == 4:
            z = einops.repeat(z, 'b t z -> b t k z', k=z_hat.shape[2])

        logpz_zhat = Normal(z_hat, z_hat_sd).log_prob(z).sum(-1)
        return logpz_zhat

    def forward(self, x, w, z=None, z0=None, frozen=True) -> ScoreResult:
        """
        x: tensor with shape [batch_size, *input_shape]
        z0: tensor with shape [batch_size, dim_z]
            If z0 is provided, x is not used.
        
        If z is not provided:
            w: tensor with shape [batch_size, ..., vocab_size]
            returns: [batch_size, ...]
        If z is provided, one of the following combinations
            1.        z: [batch_size, dim_z]
                      w: [batch_size, vocab_size]
                returns: [batch_size]
            2.        z: [batch_size, dim_z]
                      w: [batch_size, k, vocab_size]
                returns: [batch_size, k]
            3.        z: [batch_size, num_steps, dim_z]
                      w: [batch_size, num_steps, vocab_size]
                returns: [batch_size, num_steps]
            4.        z: [batch_size, num_steps, dim_z]
                      w: [batch_size, num_steps, k, vocab_size]
                returns: [batch_size, num_steps, k]
        """
        batch_size = len(w)
        batch_shape = w.shape[:-1]
        if z0 is None:
            z0 = self.get_z0(x, frozen=frozen)
        z0_sd = self.z_sd_model(z0)
        z_hat, z_hat_sd = self.get_z_hat(w, return_sd=True)

        z_hat = z_hat.view(batch_size, -1, self.dim_z)
        z_hat_sd = z_hat_sd.view(batch_size, -1, self.dim_z)
        z0 = einops.repeat(z0, 'b z -> b k z', k=z_hat.shape[1])
        z0_sd = einops.repeat(z0_sd, 'b z -> b k z', k=z_hat.shape[1])

        z0_dist = Normal(z0, z0_sd)
        z_hat_dist = Normal(z_hat, z_hat_sd)
        score = -torch.distributions.kl_divergence(z0_dist, z_hat_dist).sum(-1)
        
        z0 = z0[:,0]
        z0_sd = z0_sd[:,0]
        score = score.view(*batch_shape)
        z_hat = z_hat.view(*batch_shape, self.dim_z)
        z_hat_sd = z_hat_sd.view(*batch_shape, self.dim_z)
        if z is None:
            logpz_zhat = torch.zeros_like(score)
        else:
            logpz_zhat = self.get_logpz_zhat(z, z_hat, z_hat_sd)

        return ScoreResult(score=score, logpz_zhat=logpz_zhat, z0=z0, z0_sd=z0_sd, z_hat=z_hat, z_hat_sd=z_hat_sd)

    def get_vae_loss(self, x):
        z0 = self.get_z0(x, frozen=False)
        z0_sd = self.z_sd_model(z0)
        z0_dist = Normal(z0, z0_sd)
        recon_error, metrics = self.get_recon_loss(x, z0_dist.rsample())
        prior_dist = Normal(torch.zeros_like(z0), torch.ones_like(z0))
        kl_z0 = torch.distributions.kl_divergence(z0_dist, prior_dist).sum(-1).mean()
        metrics['kl'] = kl_z0.item()
        loss = recon_error + self.vae_beta * kl_z0
        return loss, metrics
    
    def get_loss(self, x, w, **kwargs):
        """
        batch: dict containing
            x: tensor with shape [batch_size, num_features]
        w: tensor with shape [batch_size, ..., vocab_size]
        returns:
            loss: scalar
            metrics: dict
        """
        batch_size = len(w)
        score_result: ScoreResult = self.forward(x, w, frozen=False)
        z0_dist = Normal(score_result.z0, score_result.z0_sd)
        prior_dist = Normal(torch.zeros_like(score_result.z0), torch.ones_like(score_result.z0))
        z0 = Normal(score_result.z0, score_result.z0_sd).rsample()
        recon_error, metrics = self.get_recon_loss(x, z0)
        kl_z0 = torch.distributions.kl_divergence(z0_dist, prior_dist).sum(-1).mean()
        
        z_hat = score_result.z_hat.view(batch_size, -1, self.dim_z)
        z_hat_sd = score_result.z_hat_sd.view(batch_size, -1, self.dim_z)
        mask = (w.sum(-1) > 0).view(batch_size, -1)
        z0_ = einops.repeat(score_result.z0, 'b z -> b k z', k=z_hat.shape[1])
        z0_sd_ = einops.repeat(score_result.z0_sd, 'b z -> b k z', k=z_hat.shape[1])
        z0_dist_ = Normal(z0_, z0_sd_)
        z0_dist_detached = Normal(z0_.detach(), z0_sd_.detach())
        zhat_dist = Normal(z_hat, z_hat_sd)
        zhat_dist_detached = Normal(z_hat.detach(), z_hat_sd.detach())
        kl_zhat1 = torch.distributions.kl_divergence(z0_dist_detached, zhat_dist).sum(-1)
        kl_zhat2 = torch.distributions.kl_divergence(z0_dist_, zhat_dist_detached).sum(-1)
        kl_zhat1 = (mask * kl_zhat1).sum() / mask.sum()
        kl_zhat2 = (mask * kl_zhat2).sum() / mask.sum()

        loss = recon_error + self.vae_beta * kl_z0 + self.cvae_beta * kl_zhat1 + .25 * self.cvae_beta * kl_zhat2
        

        metrics.update({'loss': loss.item(),
                        'recon_error': recon_error.item(),
                        'z0_norm': score_result.z0.norm(dim=-1).mean().item(),
                        'z0_sd_norm': score_result.z0_sd.norm(dim=-1).mean().item(),
                        'z_hat_norm': score_result.z_hat.norm(dim=-1).mean().item(),
                        'z_hat_sd_norm': score_result.z_hat_sd.norm(dim=-1).mean().item(),
                        'kl_z0': kl_z0.item(),
                        'kl_z_hat': kl_zhat1.item(),
                        'kl_diff': (kl_z0 - kl_zhat1).mean()})
        return loss, metrics

    def sample(self, w):
        """
        w: (n, vocab_size)
        returns: 
            z0: (n, dim_z)
            z: (n, dim_z)
        """
        mu, sd = self.get_z_hat(w, return_sd=True)
        z = Normal(mu, sd).sample()
        return z, mu
