from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from .data.image_datamodule import ImageDataModule
from .models.bitmap_gfn import BitmapGFN
from .models.images import TransformerImageEncoder, TransformerImageDecoder
from .models.m_model import MModel
from .models.gfn_em import GFNEM, GFNEMConfig
from .models.attractors_gfn_em import AttractorsGFNEM, AttractorsGFNEMConfig
from .misc import image_utils as iu, torch_utils as tu



@dataclass
class ImagesGFNEMConfig(GFNEMConfig):
                 
    discretizer_encoder_dim_h: int = 128
    m_model_patch_size: int = 8
    m_model_use_mu_as_z0: bool = True,
    m_model_sd_model_num_layers: int =3
    m_model_nhead: int = 8
    m_model_dim_feedforward: int = 256
    m_model_num_encoder_layers: int =2
    m_model_num_decoder_layers: int =2
    
    
@dataclass
class ImagesAttractorsGFNEMConfig(ImagesGFNEMConfig, AttractorsGFNEMConfig):
    pass
        

class ImagesMModel(MModel):

    def __init__(self, 
                 image_size, 
                 num_channels, 
                 patch_size,
                 mu_dependent_sd=True,
                 sd_model_num_layers=3,
                 nhead=8, 
                 dim_feedforward=512, 
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 use_mu_as_z0=True,
                 *args, **kwargs):
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.mu_dependent_sd = mu_dependent_sd
        self.sd_model_num_layers = sd_model_num_layers
        self.use_mu_as_z0 = use_mu_as_z0
        super().__init__(*args, **kwargs)

    def init_encoder(self):
        return TransformerImageEncoder(size=self.image_size,
                                               num_channels=self.num_channels,
                                               patch_size=self.patch_size,
                                               dim_z=self.dim_z,
                                               variational=True,
                                               mu_dependent_sd=self.mu_dependent_sd,
                                               sd_model_num_layers=self.sd_model_num_layers,
                                               dim_h=self.dim_h,
                                               nhead=self.nhead,
                                               dim_feedforward=self.dim_feedforward,
                                               num_layers=self.num_encoder_layers,
                                               dropout=0.)
                                    
    def init_decoder(self):
        return TransformerImageDecoder(size=self.image_size,
                                               num_channels=self.num_channels,
                                               patch_size=self.patch_size,
                                               dim_z=self.dim_z,
                                               nhead=self.nhead,
                                               dim_feedforward=self.dim_feedforward,
                                               num_layers=self.num_decoder_layers,
                                               dropout=0.)
    
    def encode(self, x, encoder):
        z, mu, sigma = self.encoder(x, return_params=True)
        if self.use_mu_as_z0:
            return mu
        return z
    
    def get_recon_loss(self, x, z):
        logits = self.decoder(z)
        loss = F.binary_cross_entropy_with_logits(logits, x, reduction='none').sum((-3, -2, -1)).mean()
        return loss, {'recon_loss': loss.item()}


class ImagesBitmapGFN(BitmapGFN):

    def __init__(self, 
                 image_size, 
                 num_channels, 
                 patch_size,
                 dim_h,
                 nhead=8, 
                 dim_feedforward=512, 
                 num_encoder_layers=3,
                 num_mlp_layers=3,
                 *args, **kwargs):
        super().__init__(dim_input=dim_h, 
                         dim_h=dim_feedforward, 
                         num_layers=num_mlp_layers, *args, **kwargs)
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers

        self.encoder = TransformerImageEncoder(size=image_size,
                                               num_channels=num_channels,
                                               patch_size=patch_size,
                                               dim_z=dim_h,
                                               dim_h=self.dim_h,
                                               nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               num_layers=num_encoder_layers,
                                               dropout=0)
        
    def get_logpw(self, x, log_reward):
        z = self.encoder(x)
        return super().get_logpw(z, log_reward)
    
    def sample(self, x, *args, **kwargs):
        z = self.encoder(x)
        return super().sample(z, *args, **kwargs)
    
    def get_tb_loss(self, x, *args, **kwargs):
        z = self.encoder(x)
        logZ = self.flow_model(z)
        return super().get_tb_loss(z, logZ=logZ, *args, **kwargs)


class ImagesGFNEM(GFNEM):

    def __init__(self, config: ImagesGFNEMConfig, data_module: ImageDataModule):
        super().__init__(config, data_module)
        self.config = config

    @property
    def image_size(self):
        return self.data_module.size

    @property
    def num_channels(self):
        return self.data_module.num_channels
    
    @property
    def patch_size(self):
        return self.config.m_model_patch_size
    
    @property
    def input_shape(self):
        return (self.num_channels, self.image_size, self.image_size)

    def init_m_model(self, **kwargs):
        return ImagesMModel(self.data_module.size, 
                            self.data_module.num_channels, 
                            self.config.m_model_patch_size,
                            use_mu_as_z0=self.config.m_model_use_mu_as_z0,
                            mu_dependent_sd=True,
                            sd_model_num_layers=self.config.m_model_sd_model_num_layers,
                            nhead=self.config.m_model_nhead,
                            dim_feedforward=self.config.m_model_dim_feedforward, 
                            num_encoder_layers=self.config.m_model_num_encoder_layers,
                            num_decoder_layers=self.config.m_model_num_decoder_layers,
                            dim_z=self.config.dim_z, 
                            vocab_size=self.config.vocab_size, 
                            vocab_group_size=self.config.vocab_group_size,
                            dim_h=self.config.m_model_dim_h,
                            num_layers=self.config.m_model_num_layers,
                            num_w_embedding_layers=self.config.m_model_num_w_embedding_layers,
                            vae_beta=self.config.m_model_vae_beta,
                            cvae_beta=self.config.m_model_cvae_beta,
                            **kwargs)

    def init_x_discretizer(self):
        return ImagesBitmapGFN(self.image_size, 
                               self.num_channels, 
                               self.patch_size,
                               self.config.discretizer_encoder_dim_h,
                               nhead=self.config.m_model_nhead, 
                               dim_feedforward=self.config.m_model_dim_feedforward,
                               num_encoder_layers=self.config.m_model_num_encoder_layers,
                               num_mlp_layers=self.config.discretizer_num_layers,
                               num_bits=self.config.vocab_size, 
                               group_size=self.config.vocab_group_size,
                               fixed_backward_policy=self.config.discretizer_fixed_backward_policy)
    
    @torch.no_grad()
    def create_plots(self, batch):
        super().create_plots(batch)

        x = batch['x']
        z0 = self.get_z0(x)
        w, logpf, logpb, logpt = self.sample_w(x, argmax=False)
        z_hat_x, sd_x = self.m_model.get_z_hat(w, return_sd=True)
        zeta_x = Normal(z_hat_x, 1*sd_x).sample()
        x_hat_z0 = self.m_model.decoder(z0).sigmoid()
        x_hat_x = self.m_model.decoder(z_hat_x).sigmoid()
        x_hat_zeta_x = self.m_model.decoder(zeta_x).sigmoid()

        k = 12
        w_strings = self.x_discretizer_model.stringify(w)
        images = [self.data_module.render(x[i], size=96) for i in range(k)]
        images += [self.data_module.render(x_hat_z0[i], size=96) for i in range(k)]
        images += [self.data_module.render(x_hat_x[i], size=96, text=w_strings[i]) for i in range(k)]
        images += [self.data_module.render(x_hat_zeta_x[i], size=96) for i in range(k)]
        images = iu.compose_grid(images, 4, k)
        self.log_image('sample_grid', images)


class ImagesAttractorsGFNEM(ImagesGFNEM, AttractorsGFNEM):
    
    @torch.no_grad()
    def create_plots(self, batch):
        image = self.create_sample_grid(batch['x'][:12])
        self.log_image('sample_grid', image)

        images = self.create_pca_gif(batch['x'])
        self.log_gif('pca', images)
        
    @torch.no_grad()
    def create_sample_grid(self, x):
        k = len(x)
        z0 = self.get_z0(x)
        w_x, logpf, logpb, logpt = self.sample_w_x(x, argmax=False)
        z_hat_x, sd_x = self.m_model.get_z_hat(w_x, return_sd=True)
        zeta_x = Normal(z_hat_x, .25*sd_x).sample()
        z_traj = self.sample_forward_trajectory(z0, deterministic=False)
        w, logpf, logpb, logpt = self.sample_w(z_traj[:,-1], z0)
        z_hat, sd = self.m_model.get_z_hat(w, return_sd=True)
        zeta = Normal(z_hat, .25*sd).sample()

        x_hat_z0 = self.m_model.decoder(z0).sigmoid()
        x_hat_x = self.m_model.decoder(z_hat_x).sigmoid()
        x_hat = self.m_model.decoder(z_hat).sigmoid()
        x_hat_zeta_x = self.m_model.decoder(zeta_x).sigmoid()
        x_hat_zeta = self.m_model.decoder(zeta).sigmoid()

        w_strings = self.discretizer_model.stringify(w)
        images = []
        images += [self.data_module.render(x[i], size=96) for i in range(k)]
        images += [self.data_module.render(x_hat_z0[i], size=96) for i in range(k)]
        images += [self.data_module.render(x_hat_x[i], size=96) for i in range(k)]
        images += [self.data_module.render(x_hat_zeta_x[i], size=96) for i in range(k)]
        images += [self.data_module.render(x_hat[i], size=96, text=w_strings[i]) for i in range(k)]
        images += [self.data_module.render(x_hat_zeta[i], size=96) for i in range(k)]
        images = iu.compose_grid(images, 6, k)
        return images

    @torch.no_grad()
    def create_pca_gif(self, x, v=None, pca_mode='z0'):
        if v is None:
            u, s, v = self.get_svd(x, pca_mode=pca_mode)
        z0 = self.get_z0(x)
        z_traj_f = self.sample_forward_trajectory(z0)
        w, _, _, _ = self.sample_w(z_traj_f[:,-1], z0)
        z_hat_pca = (self.get_z_hat(w) @ v.T)[...,:2]
        z_traj_pca = (z_traj_f @ v.T)[:,:,:2]
        df_traj = tu.to_long_df(z_traj_pca, ['idx', 'step'], ['pc1', 'pc2'])
        df_zhat = tu.to_long_df(z_hat_pca[:,0], ['idx'], 'pc1', pc2=z_hat_pca[:,1])
        images = [iu.plot_to_image(self.plot_2d_step(df_traj, step=t, df_zhat=df_zhat, scale=1)) for t in range(z_traj_f.shape[1])]
        return images