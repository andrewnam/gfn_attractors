import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from matplotlib.colors import rgb2hex
from plotnine import *
import math
import itertools

from .models.m_model import MModel
from .models.discretizer import BoWDiscretizeModule
from .models.gfn_em import GFNEM
from .models.bitmap_gfn import BitmapGFN
from .models.helpers import MLP, FeaturePredictor
from .images import ImagesGFNEM, ImagesAttractorsGFNEM
from .misc import torch_utils as tu, image_utils as iu


class DSpritesGFNEM(ImagesGFNEM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = FeaturePredictor(self.config.dim_z, 256, 3, joint=True, r=1, g=1, b=1, shape=3, scale=0, x=0, y=0)

    def add_optimizers(self):
        e_step, m_step, others = super().add_optimizers()
        others['classifier'] = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
        return e_step, m_step, others

    @torch.enable_grad()
    def run_classifier(self, num_updates):
        optimizer = self.optimizers_dict['classifier']
        for i in range(num_updates):
            indices = np.random.choice(len(self.data_module), self.data_module.batch_size)
            batch = self.data_module.create_batch(indices, color_set='both', device=self.device)
            x = batch['x']
            latents = batch['latent']
            labels = batch['label']
            targets = torch.cat([labels[:,:4], latents[:,-3:]], dim=-1)
            with torch.no_grad():
                w = self.sample_w_from_x(x)
                z0 = self.get_z0(x)
                z_hat = self.get_z_hat(w)
            if i < num_updates - 1:
                z = torch.cat([z0, z_hat], dim=0)
                loss, metrics = self.classifier.get_loss(z, torch.cat([targets, targets], dim=0))
            else:
                loss0, metrics0 = self.classifier.get_loss(z0, targets)
                loss_zhat, metrics_zhat = self.classifier.get_loss(z_hat, targets)
                loss = loss0 + loss_zhat
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics0 = {f'classifier/z0/{k}': v for k, v in metrics0.items()}
        metrics_zhat = {f'classifier/zhat/{k}': v for k, v in metrics_zhat.items()}
        return {**metrics0, **metrics_zhat}
    
    def get_performance_metrics(self, n=256):
        metrics = super().get_performance_metrics(n=n)
        metrics.update(self.run_classifier(10))
        return metrics

class DSpritesAttractorsGFNEM(DSpritesGFNEM, ImagesAttractorsGFNEM):

    @torch.no_grad()
    def create_plots(self):
        indices = np.random.choice(self.data_module.train_indices, 500)
        batch = self.data_module.create_batch(indices, device=self.device)

        image = self.create_sample_grid(batch['x'][:12])
        self.log_image('sample_grid', image)

        images = self.create_pca_gif(batch)
        self.log_gif('pca', images)

    @torch.no_grad()
    def create_pca_gif(self, batch):
        x = batch['x']
        labels = batch['label'].detach().cpu().numpy()
        latents = batch['latent'].detach().cpu().numpy()
        rgb = np.array([rgb2hex(c) for c in latents[:,:3].clip(0, 1)])
        shapes = ['square', 'oval', 'heart']
        shapes = np.array([shapes[i] for i in labels[:,3].astype(int)])

        z0 = self.get_z0(x)
        u, s, v = self.get_svd(x, pca_mode='z0')
        z_traj_f = self.sample_forward_trajectory(z0)
        w, _, _, _ = self.sample_w(z_traj_f[:,-1], z0)
        z_hat_pca = (self.get_z_hat(w) @ v.T)[...,:2]
        z_traj_pca = (z_traj_f @ v.T)[:,:,:2]
        df_traj = tu.to_long_df(z_traj_pca, ['idx', 'step'], ['pc1', 'pc2'], color=rgb, shape=shapes)
        df_zhat = tu.to_long_df(z_hat_pca, ['idx'], ['pc1', 'pc2'])
        images = []
        for t in range(z_traj_f.shape[1]):
            p = self.plot_2d_step(df_traj, step=t, df_zhat=df_zhat, scale=1, color='color', shape='shape')
            p += scale_color_identity()
            images.append(iu.plot_to_image(p))
        return images
    

class DSpritesBaseline(BoWDiscretizeModule):
    """
    If compositionality is 'none', all the labels map to a random sequence.
    If compositionality is 'half', the labels are split into categories and continuous values, and the category labels map to a random sequence.
    If compositionality is 'full', all the labels are used as the sequence.
    """
    
    def __init__(self, data_module, compositionality: str, scale_granularity=1, pos_granularity=1, seed=0, device='cpu'):
        vocab_size = 9 + math.ceil(6 / scale_granularity) + 2*math.ceil(31 / pos_granularity)
        super().__init__(vocab_size)
        assert compositionality in ('full', 'half', 'none')
        
        self.compositionality = compositionality
        self.rng = np.random.default_rng(seed)
        self.data_module = data_module
        self.scale_granularity = scale_granularity
        self.pos_granularity = pos_granularity
        self.seed = seed
        self.device = device
        
        if compositionality == 'half':
            keys = tuple(itertools.product(range(2), range(2), range(2), range(3)))
            values = []
            for k in keys:
                s = [0, 0, 0]
                s[k[3]] = 1
                values.append(list(k[:3]) + s)
            self.w = torch.tensor(values, dtype=torch.float, device=device)
            self.indices = {k: i for k, i in zip(keys, self.rng.permutation(len(self.w)))}
        elif compositionality == 'none':
            keys = tuple(itertools.product(range(2), range(2), range(2), range(3), range(6), range(31), range(31)))
            values = torch.tensor(keys)

            rgb = F.one_hot(values[:,:3], 2).flatten(-2, -1)
            shape = F.one_hot(values[:,3])
            scale = F.one_hot((values[:,4] / self.scale_granularity).ceil().long())
            x = F.one_hot((values[:,5] / self.pos_granularity).ceil().long())
            y = F.one_hot((values[:,6] / self.pos_granularity).ceil().long())
            self.w = torch.cat([rgb, shape, scale, x, y], dim=-1).float().to(device)
            self.indices = {k: i for k, i in zip(keys, self.rng.permutation(len(self.w)))}
            
    @property
    def max_w_length(self):
        return 7

    def to(self, device):
        self.device = device
        if self.compositionality in ('half', 'none'):
            self.w = self.w.to(device)
        return self
        
    def encode_continuous(self, scale, x, y):
        scale = scale.long() // self.scale_granularity
        x = x.long() // self.pos_granularity
        y = y.long() // self.pos_granularity
        w = torch.cat([F.one_hot(scale, math.ceil(6 / self.scale_granularity)),
                       F.one_hot(x, math.ceil(31 / self.pos_granularity)),
                       F.one_hot(y, math.ceil(31 / self.pos_granularity))], dim=-1)
        return w
    
    def get_w(self, labels, **kwargs):
        if self.compositionality == 'none':
            indices = [self.indices[tuple(c)] for c in labels.int().detach().cpu().numpy()]
            return self.w[indices].float() 
        if self.compositionality == 'half':
            categories = labels[:,:4].int().detach().cpu().numpy()
            indices = [self.indices[tuple(c)] for c in categories]
            w_cat = self.w[indices]
            w_cont = self.encode_continuous(labels[:,4], labels[:,5], labels[:,6])
            return torch.cat([w_cat, w_cont], dim=-1).float()
        
        rgb = F.one_hot(labels[:,:3].long(), 2).flatten(-2, -1)
        # rgb = labels[:,:3].float()
        shape = F.one_hot(labels[:,3].long(), 3)
        w_cont = self.encode_continuous(labels[:,4], labels[:,5], labels[:,6])
        return torch.cat([rgb, shape, w_cont], dim=-1).float() 
