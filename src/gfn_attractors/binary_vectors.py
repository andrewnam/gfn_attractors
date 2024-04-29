import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from plotnine import *
from dataclasses import dataclass


from .models.m_model import MModel
from .models.gfn_em import GFNEM
from .models.discretizer import BoWDiscretizeModule
from .models.bitmap_gfn import BitmapGFN
from .models.helpers import MLP
from .models.attractors_gfn_em import AttractorsGFNEM
from .models.codebook import Codebook
from .misc import torch_utils as tu, image_utils as iu, Config


class BinaryVectorMModel(MModel):

    def __init__(self, num_features, *args, **kwargs):
        self.num_features = num_features
        super().__init__(*args, **kwargs)

    def init_encoder(self):
        return MLP(self.num_features, self.dim_z, self.dim_h, n_layers=self.num_layers, nonlinearity=nn.ReLU())
                 
    def init_decoder(self):
        return MLP(self.dim_z, self.num_features, self.dim_h, n_layers=self.num_layers, nonlinearity=nn.ReLU())
    
    def get_recon_loss(self, x, z):
        logits = self.decoder(z)
        loss = F.binary_cross_entropy_with_logits(logits, x, reduction='none').sum(-1).mean()
        accuracy = ((logits > 0) == x).float().mean()
        return loss, {'recon_loss': loss.item(), 'recon_accuracy': accuracy.item()}


class BinaryVectorGFNEM(GFNEM):

    def init_m_model(self, **kwargs):
        return BinaryVectorMModel(num_features=self.data_module.num_features, 
                       dim_z=self.config.dim_z, 
                 vocab_size=self.config.vocab_size, 
                 vocab_group_size=self.config.vocab_group_size,
                 dim_h=self.config.m_model_dim_h,
                 num_layers=self.config.m_model_num_layers,
                 num_w_embedding_layers=self.config.m_model_num_w_embedding_layers,
                 vae_beta=self.config.m_model_vae_beta,
                 cvae_beta=self.config.m_model_cvae_beta, **kwargs)

    def init_x_discretizer(self):
        return BitmapGFN(dim_input=self.data_module.num_features, 
                         num_bits=self.config.vocab_size, 
                         group_size=self.config.vocab_group_size, 
                         dim_h=self.config.discretizer_dim_h, 
                         num_layers=self.config.discretizer_num_layers)
    
    
class BinaryVectorAttractorsGFNEM(BinaryVectorGFNEM, AttractorsGFNEM):

    @property
    def input_shape(self):
        return (self.data_module.num_features, )
    
    def create_plots(self):
        super().create_plots()
        images = self.create_pca_gif(500)
        self.log_gif('pca', images)

    def plot_pca_step(self, df_traj, step, df_zhat=None):
        xlim =  df_traj.pc1.min(), df_traj.pc1.max()
        ylim =  df_traj.pc2.min(), df_traj.pc2.max()
        df_traj = df_traj[df_traj.step == step]
        p = ggplot(df_traj[df_traj.step == step], aes(x='pc1', y='pc2', color='gen1'))
        if df_zhat is not None:
            p = p + geom_point(aes(x='pc1', y='pc2'), data=df_zhat, size=1, color='black')
        return (p
        + geom_point(size=1, alpha=.5)
        + coord_cartesian(xlim=xlim, ylim=ylim)
        + labs(title=f'Step {step}')
        + theme_light()
        )

    @torch.no_grad()
    def create_pca_gif(self, n, pca_mode='z0'):
        batch = self.data_module.create_batch(np.random.randint(0, len(self.data_module), n))
        x = batch['x'].to(self.device)
        labels = batch['labels'].to(self.device)
        u, s, v = self.get_svd(x, pca_mode)

        z0 = self.get_z0(x)
        z_traj = self.sample_forward_trajectory(z0, deterministic=False)
        w, _, _, _ = self.sample_w(z_traj, z0)
        z_hat = self.get_z_hat(w[:,-1])

        z_traj_pca = (z_traj @ v.T)[:,:,:2]
        z_hat_pca = (z_hat @ v.T)[:,:2]
        df_traj = tu.to_long_df(z_traj_pca[:,:,0], ['batch', 'step'], value_name='pc1', pc2=z_traj_pca[:,:,1], gen1=labels[:,0])
        df_traj.gen1 = df_traj.gen1.astype(str)
        df_zhat = tu.to_long_df(z_hat_pca[:,0], ['batch'], value_name='pc1', pc2=z_hat_pca[:,1]).drop_duplicates(['pc1', 'pc2'])

        return [iu.plot_to_image(self.plot_pca_step(df_traj, t, df_zhat)) for t in range(self.config.num_steps)]


@dataclass
class BinaryVectorsVQVAEConfig(Config):
    
    dim_z: int
    
    dm_depth: int
    dm_repeat: int
    dm_sample_ancestors: int
    dm_batch_size: int
    
    seed: int = 0
    dim_h: int = 128


class BinaryVectorsVQVAE(BoWDiscretizeModule):

    def __init__(self, config: BinaryVectorsVQVAEConfig, data_module):
        super().__init__(vocab_size=2*data_module.depth, group_size=2)
        self.config = config
        self.data_module = data_module
        self.mlp = MLP(data_module.num_features, self.num_codebooks * config.dim_h, config.dim_h, n_layers=3, nonlinearity=nn.ReLU())
        self.h_to_z = MLP(config.dim_h, config.dim_z, config.dim_h, n_layers=1, nonlinearity=nn.ReLU())
        self.codebook = Codebook(config.dim_h, num_entries=2, num_codebooks=self.num_codebooks, beta=0.25)
        self.decoder = MLP(config.dim_z, data_module.num_features, config.dim_h, n_layers=3, nonlinearity=nn.ReLU())
        
    @property
    def num_codebooks(self):
        return self.data_module.depth

    @property
    def max_w_length(self):
        return self.codebook.num_codebooks
    
    @property
    def device(self):
        return self.codebook.device

    def forward(self, x, return_losses=False):
        h = torch.stack(self.mlp(x).chunk(self.num_codebooks, dim=-1), dim=-2)
        h_q, indices, dictionary_loss, commitment_loss = self.codebook(h, return_losses=True)
        h_q = h_q.sum(-2)
        z = self.h_to_z(h_q)

        if return_losses:
            return z, indices, dictionary_loss, commitment_loss
        return z, indices
    
    def get_loss(self, x):
        h, w, dictionary_loss, commitment_loss = self(x, return_losses=True)
        logits = self.decoder(h)
        recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='none').sum(-1).mean()
        loss = recon_loss + dictionary_loss + commitment_loss
        loss = loss.mean()
        metrics = {'loss': loss.item(), 
                   'reconstruction_loss': recon_loss.item(), 
                   'dictionary_loss': dictionary_loss.item(), 
                   'commitment_loss': commitment_loss.item()}
        return loss, metrics
    
    def get_w(self, x, **kwargs):
        z, w = self(x)
        return F.one_hot(w, self.codebook.num_entries).flatten(-2, -1)
