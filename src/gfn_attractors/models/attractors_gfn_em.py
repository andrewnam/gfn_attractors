import numpy as np
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
import einops
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lightning.pytorch.utilities import CombinedLoader
from plotnine import *
from pytorch_lightning.loggers import WandbLogger

from .gfn_em import GFNEM, GFNEMConfig
from .bitmap_gfn import BitmapGFN
from ..misc import torch_utils as tu, image_utils as iu
from ..misc.replay_buffer import EStepReplayBuffer, MStepReplayBuffer
from .helpers import MLP, PositionalEncoding
from .dynamics import MLPMeanBoundedDynamics, MLPLangevinMeanBoundedDynamics


@dataclass
class AttractorsGFNEMConfig(GFNEMConfig):

    attractor_sd: float = .05

    dim_t: int = 10
    dynamics_dim_h: int = 256
    dynamics_num_layers: int = 3
    num_steps: int = 20
    max_mean: float = 0.15 # If None, uses max_travel to calculate max_mean. Otherwise, uses this value.
    zT_max_mean: float|None = None # If not None, learns P(z_T | z_hat) with a fixed sd = attractor_sd.
    max_sd: float = .25
    min_sd: float = 0.
    z0_dependent_forward: bool = False
    z0_dependent_backward: bool = True
    t_dependent_forward: bool = False
    t_dependent_backward: bool = True
    directional_dynamics: bool = True

    # Discretizer
    z0_dependent_discretizer: bool = True

    # Replay buffer
    replay_buffer_inv_freq_sequence: bool = True
    replay_buffer_inv_freq_token: bool = True
    e_step_batch_size: int = 2048
    m_step_batch_size: int = 256

    e_step_buffer_size: int = 100000
    e_step_start_rollouts: int = 1024
    e_step_buffer_update_interval: int = 50
    e_step_update_rollouts: int = 128

    m_step_buffer_size: int = 10000
    m_step_start_rollouts: int = 1024
    m_step_buffer_update_interval: int = 50
    m_step_update_rollouts: int = 128

    # Training
    lr_dynamics: float = 1e-4
    p_explore_dynamics = 0.05
    p_add_exploration_trajectory: float = 0.


class AttractorsGFNEM(GFNEM):

    def __init__(self, config: AttractorsGFNEMConfig, data_module):
        super().__init__(config, data_module)
        self.config = config
        self.temporal_encoding = PositionalEncoding(config.dim_t) if config.dim_t is not None else None
        self.g_model = MLP(2 * config.dim_z + config.dim_t, 1, hidden_dim=config.flow_dim_h, n_layers=config.flow_num_layers, nonlinearity=nn.ReLU())
        self.dynamics_model = self.init_dynamics()
        self.discretizer_model = self.init_discretizer()
        
        if config.zT_max_mean is None:
            self.zT_dynamics_model = None
        else:
            self.zT_dynamics_model = MLPMeanBoundedDynamics(dim_x=self.config.dim_z, dim_z=self.config.dim_z, 
                                                            dim_t=self.config.dim_t, dim_h=self.config.dynamics_dim_h,                                                   
                                                            directional=self.config.directional_dynamics,
                                                            flag_z0=False,
                                                            allow_terminate=False,
                                                            num_layers=self.config.dynamics_num_layers, 
                                                            nonlinearity=nn.ReLU(),
                                                            max_mean=self.config.zT_max_mean,
                                                            fixed_sd=self.config.attractor_sd,
                                                            t_dependent_forward=False,
                                                            t_dependent_backward=False,
                                                            x_dependent_forward=self.config.z0_dependent_forward,
                                                            x_dependent_backward=self.config.z0_dependent_backward)
        
        self.e_step_replay_buffer = EStepReplayBuffer(config.dim_z, size=config.e_step_buffer_size, 
                                                      batch_size=config.e_step_batch_size, vocab_size=config.vocab_size,
                                                      inv_freq_sequence=self.config.replay_buffer_inv_freq_sequence, 
                                                      inv_freq_token=self.config.replay_buffer_inv_freq_token)
        self.m_step_replay_buffer = MStepReplayBuffer(self.input_shape, vocab_size=config.vocab_size,
                                                      batch_size=config.m_step_batch_size,
                                                      inv_freq_sequence=False, 
                                                      inv_freq_token=False,
                                                      size=config.m_step_buffer_size)
        
    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def vocab_size(self):
        return self.config.vocab_size
    
    @property
    def max_w_length(self):
        return self.config.max_w_length
    
    def train_dataset(self):
        return DataLoader(tu.DummyDataset(size=1000))

    def init_dynamics(self):
        return MLPMeanBoundedDynamics(dim_x=self.config.dim_z, dim_z=self.config.dim_z, 
                                      dim_t=self.config.dim_t, dim_h=self.config.dynamics_dim_h,                                                   
                                      directional=self.config.directional_dynamics,
                                      flag_z0=not self.config.t_dependent_forward,
                                      allow_terminate=False,
                                      num_layers=self.config.dynamics_num_layers, 
                                      nonlinearity=nn.ReLU(),
                                      max_mean=self.config.max_mean, 
                                      min_sd=self.config.min_sd, 
                                      max_sd=self.config.max_sd,
                                      fixed_sd=None,
                                      t_dependent_forward=self.config.t_dependent_forward,
                                      t_dependent_backward=self.config.t_dependent_backward,
                                      x_dependent_forward=self.config.z0_dependent_forward,
                                      x_dependent_backward=self.config.z0_dependent_backward)
    
    def init_discretizer(self):
        dim_input = (1 + self.config.z0_dependent_discretizer) * self.config.dim_z
        return BitmapGFN(dim_input=dim_input, 
                         num_bits=self.config.vocab_size, 
                         group_size=self.config.vocab_group_size, 
                         fixed_backward_policy=self.config.discretizer_fixed_backward_policy,
                         dim_h=self.config.discretizer_dim_h, 
                         num_layers=self.config.discretizer_num_layers,
                         dim_flow_input=2*self.config.dim_z)
    
    def add_optimizers(self):
        e_step = [{'params': self.dynamics_model.parameters(), 'lr': self.config.lr_dynamics},
                  {'params': self.discretizer_model.parameters(), 'lr': self.config.lr_discretizer},
                  {'params': [*self.g_model.parameters()], 'lr': self.config.lr_flows}]
        if self.config.zT_max_mean is not None:
            e_step.append({'params': self.zT_dynamics_model.parameters(), 'lr': self.config.lr_dynamics})
        m_step = []
        others = {}
        return e_step, m_step, others
    
    def sample_substrings(self, w):
        return self.discretizer_model.sample_substrings(w)
    
    def get_z0(self, x):
        return self.m_model.get_z0(x)

    def get_z_hat(self, w):
        return self.m_model.get_z_hat(w)
    
    def get_w(self, x, **kwargs):
        z0 = self.get_z0(x)
        z_traj = self.sample_forward_trajectory(z0)
        w, _, _, _ = self.sample_w(z_traj[:,self.config.num_steps], z0)
        return w
    
    @torch.no_grad()
    def sample_w_from_x(self, x, *args, **kwargs):
        z0 = self.get_z0(x)
        z_traj = self.sample_forward_trajectory(z0)
        w, _, _, _ = self.sample_w(z_traj[:,self.config.num_steps], z0, *args, **kwargs)
        return w
    
    def get_g(self, z, z0, t):
        """
        z: (..., dim_z)
        z0: (..., dim_z)
        t: (...)
        """
        h = torch.cat([z0, z], dim=-1)
        g = self.g_model(self.temporal_encoding(h, t))
        g = g * (self.config.num_steps - t)
        return g
    
    def get_logpb_z_z_hat(self, z0, z, z_hat):
        if self.zT_dynamics_model is None:
            sd = torch.full_like(z_hat, self.config.attractor_sd)
            logpz = Normal(z_hat, sd).log_prob(value=z).sum(-1)
            return logpz, torch.zeros_like(z_hat), sd
        else:
            _, _, logpz, _, _, mu, sd = self.zT_dynamics_model.step(z_hat, z0, target=z, return_params=True)
            return logpz, mu, sd
    
    @torch.no_grad()
    def get_log_reward(self, z0, z, w, return_score_result=False):
        """
        z0: (batch_size, dim_z)
        z: (batch_size, dim_z)
        w: (batch_size, max_w_length)
        """
        metrics = {}
        score_result = self.m_model(x=None, w=w, z=z, z0=z0)
        decodability = score_result.logpz_zhat
        log_reward = self.config.score_weight * score_result.score + self.config.decodability_weight * decodability
        log_reward = log_reward.clamp(self.config.min_log_reward, self.config.max_log_reward)
        metrics.update({'reward/score': score_result.score.mean().item(),
                        'reward/decodability': decodability.mean().item(),
                        'reward/total': log_reward.mean().item()})
        if return_score_result:
            return log_reward, metrics, score_result
        return log_reward, metrics
    
    def correct_backward_trajectory(self, z_traj, z0, delay=0, beta=.5):
        return tu.correct_trajectory(z_traj.flip(dims=[1]), z0, beta=beta).flip(dims=[1])

    def sample_forward_trajectory(self, z, z0=None, num_steps=None, deterministic=False, scale=1., p_explore=0.):
        """
        z0: (batch_size, dim_z)
        returns: (batch_size, num_steps, dim_z)
        """
        if num_steps is None:
            num_steps = self.config.num_steps

        if z0 is None:
            z0 = z

        z = z.detach()
        z0 = z0.detach()
        if num_steps == 0:
            return z0.unsqueeze(1)
        return self.dynamics_model.sample_trajectory(z, z0, num_steps=num_steps, forward=True, 
                                                     deterministic=deterministic, scale=scale, p_explore=p_explore, explore_mean=True)
    
    def sample_backward_trajectory(self, z, z0, num_steps=None, deterministic=False, scale=1., p_explore=0.):
        """
        z: (batch_size, dim_z)
        z0: (batch_size, dim_z)
        returns: (batch_size, num_steps, dim_z)
            The first step is z0, and the last step is z.
        """
        if num_steps is None:
            num_steps = self.config.num_steps
        
        z = z.detach()
        if num_steps == 0:
            return z.unsqueeze(1)
        z0 = z0.detach()
        return self.dynamics_model.sample_trajectory(z, z0, num_steps=num_steps, forward=False, 
                                                     deterministic=deterministic, scale=scale, p_explore=p_explore, explore_mean=True)

    def sample_w_x(self, *args, **kwargs):
        return super().sample_w(*args, **kwargs)

    def sample_w(self, z, z0=None, min_steps=None, max_steps=None, target=None, temperature=1., argmax=False, allow_terminate=True, p_explore=0.):
        """
        z: tensor with shape (batch_size, ..., dim_z)
        z0: tensor with shape (batch_size, dim_z)
            If z0_dependent_discretizer is False, this is ignored.
        target: tensor with shape (batch_size, ..., w_length)
        returns:
            w: tensor with shape (batch_size, ..., w_length) or (batch_size, ..., 1+w_length, w_length)
            logpf: tensor with shape (batch_size, ..., w_length)
            logpb: tensor with shape (batch_size, ..., w_length)
            logpt: tensor with shape (batch_size, ...)
        """
        if min_steps is None:
            min_steps = self.config.min_w_length
        if max_steps is None:
            max_steps = self.config.max_w_length
        z = z.detach()
        if z0 is not None:
            z0 = z0.detach()
        if z.ndim == 2:
            if self.config.z0_dependent_discretizer:
                z = torch.cat([z0, z], dim=-1)
            return self.discretizer_model.sample(z, min_steps=min_steps, max_steps=max_steps, target=target, temperature=temperature, argmax=argmax, 
                                                 allow_terminate=allow_terminate, p_explore=p_explore)

        batch_size = z.shape[0]
        batch_shape = z.shape[:-1]
        z = z.view(-1, self.config.dim_z)
        if target is not None:
            target = target.view(-1, target.shape[-1])
        if self.config.z0_dependent_discretizer:
            z0 = einops.repeat(z0, 'b z -> (b k) z', k=z.shape[0] // batch_size)
            z = torch.cat([z0, z], dim=-1)

        w, logpf, logpb, logpt = self.discretizer_model.sample(z, min_steps=min_steps, max_steps=max_steps, target=target, temperature=temperature, argmax=argmax, 
                                                               allow_terminate=allow_terminate, p_explore=p_explore)
        w = w.reshape(*batch_shape, *w.shape[1:])
        logpf = logpf.reshape(*batch_shape, *logpf.shape[1:])
        logpb = logpb.reshape(*batch_shape, *logpb.shape[1:])
        logpt = logpt.reshape(*batch_shape, *logpt.shape[1:])
        return w, logpf, logpb, logpt
    
    @torch.no_grad()
    def sample_exploration_trajectories(self, x, p_explore_dynamics=None, p_explore_discretizer=None,
                                        return_w_only=False,
                                        avoid_sampled_wT=False, temperature_w=1, argmax_w=False, 
                                        temperature_traj=1, argmax_traj=False,
                                        correct_forward=True, correction_beta=.5):
        if p_explore_dynamics is None:
            p_explore_dynamics = self.config.p_explore_dynamics
        if p_explore_discretizer is None:
            p_explore_discretizer = self.config.p_explore_discretizer

        # Sample forward trajectory and its associated rewards
        z0 = self.get_z0(x)
        z_traj = self.sample_forward_trajectory(z0, p_explore=p_explore_dynamics)
        w, _, _, _ = self.sample_w(z_traj, z0, p_explore=p_explore_discretizer, temperature=temperature_w, argmax=argmax_w)
        z_hat = self.get_z_hat(w)
        z0_ = einops.repeat(z0, 'b z -> (b t) z', t=z_traj.shape[1])
        log_reward, metrics, score_result = self.get_log_reward(z0_, z_traj.flatten(0, 1), w.flatten(0, 1), return_score_result=True)
        log_reward = log_reward.view(z_traj.shape[:-1])

        # Sample w based on the log_rewards
        w_ = w.bool().cpu().numpy()
        w_counts = []
        for w_i in w_:
            w_unique, inverse, counts = np.unique(w_i, axis=0, return_inverse=True, return_counts=True)
            w_counts.append(counts[inverse])
        w_counts = np.array(w_counts)
        logits = log_reward - torch.tensor(w_counts, device=w.device).log()
        logits = logits / temperature_traj
        if avoid_sampled_wT:
            mask = (w == w[:,-1:]).all(-1)
            logits = logits.masked_fill(mask, -1e8)
        if argmax_traj:
            indices = logits.argmax(-1)
        else:
            indices = Categorical(logits=logits).sample()
        indices = einops.repeat(indices, 'b -> b 1 w', w=w.shape[-1])
        wT = w.gather(1, indices).squeeze(1)
        
        if return_w_only:
            return wT

        same_wT = (wT == w[:,-1]).all(-1)
        z_hat = self.get_z_hat(wT)
        # Sample and correct backward trajectories
        z_traj_b = self.sample_backward_trajectory(z_hat[~same_wT], z0[~same_wT])
        z_traj_b = self.correct_backward_trajectory(z_traj_b, z0[~same_wT], beta=correction_beta)

        # Forward trajectories
        if correct_forward:
            z_traj_f = tu.correct_trajectory(z_traj[same_wT], z_hat[same_wT], beta=correction_beta)
        else:
            z_traj_f = z_traj[same_wT]

        # Log log_rewards
        lr_new = log_reward.gather(1, indices[:,:,0]).squeeze()
        lr_old = log_reward[:,-1]
        lr_diff = (lr_new - lr_old).mean().item()
        self.log_metrics({'replay_buffer/exploration_score_diff': lr_diff})

        return z0[same_wT], z_traj_f, wT[same_wT], z0[~same_wT], z_traj_b, wT[~same_wT]
    
    @torch.no_grad()
    def add_exploration_trajectories_to_buffer(self, x):
        z0_f, z_traj_f, w_f, z0_b, z_traj_b, w_b = self.sample_exploration_trajectories(x)
        z0 = torch.cat([z0_f, z0_b], dim=0)
        z_traj = torch.cat([z_traj_f, z_traj_b], dim=0)
        w = torch.cat([w_f, w_b], dim=0)
        self.e_step_replay_buffer.add(z0, z_traj, w)

    @torch.no_grad()
    def add_sleep_trajectories_to_buffer(self, n):
        w = self.sample_w_from_prior(n, dtype=torch.float32)
        z0, z_hat = self.m_model.sample(w)
        zT = z_hat + torch.randn_like(z_hat) * self.config.attractor_sd
        z_traj = self.sample_backward_trajectory(zT, z0)
        self.log_metrics({'replay_buffer/sleep_backward_traj_correction': (z_traj[:,0] - z0).norm(dim=-1).mean()})
        z_traj = self.correct_backward_trajectory(z_traj, z0)
        self.e_step_replay_buffer.add(z0, z_traj, w)
        
    @torch.no_grad()
    def add_trajectories_to_buffer(self, x):
        z0 = self.get_z0(x)
        z_traj = self.sample_forward_trajectory(z0, p_explore=self.config.p_explore_dynamics)
        w, _, _, _ = self.sample_w(z_traj[:,-1], z0)
        self.e_step_replay_buffer.add(z0, z_traj, w)

    @torch.no_grad()
    def populate_e_step_buffer(self, n, mode=None):
        if mode is None:
            r = np.random.rand()
            if r < self.config.p_sleep_phase:
                mode = 'sleep'
            elif r < self.config.p_sleep_phase + self.config.p_add_exploration_trajectory:
                mode = 'explore'
            else:
                mode = 'wake'
        if mode == 'sleep':
            self.add_sleep_trajectories_to_buffer(n)
            return
        
        indices = np.random.choice(self.data_module.train_indices, n)
        batch = self.data_module.create_batch(indices, device=self.device)
        x = batch['x']
        
        if mode == 'explore':
            self.add_exploration_trajectories_to_buffer(x)
        else:
            self.add_trajectories_to_buffer(x)

    @torch.no_grad()
    def populate_m_step_buffer(self, n):
        indices = np.random.choice(self.data_module.train_indices, n)
        batch = self.data_module.create_batch(indices, device=self.device)
        x = batch['x']
        
        w = self.sample_exploration_trajectories(x, return_w_only=True, argmax_traj=True)
        
        if self.config.m_step_substrings:
            x_indices = []
            subw = []
            for i, w_i in enumerate(w.cpu().numpy()):
                indices = w_i.nonzero()[0]
                np.random.shuffle(indices)
                for k in indices:
                    x_indices.append(i)
                    subw.append(w_i)
                    w_i = w_i.copy()
                    w_i[k] = 0
            x = x[x_indices]
            w = torch.tensor(np.array(subw), dtype=torch.float32, device=w.device)
        self.m_step_replay_buffer.add(x, w)
        
    def get_dynamics_loss(self, batch_size):
        z0, z1, z2, w, t = self.e_step_replay_buffer.sample(batch_size)
        batch_size = len(z0)
        z2, logpf, logpb, mu_f, sigma_f, mu_b, sigma_b = self.dynamics_model.step(z1, z0, t, target=z2, return_params=True)
        logpb = logpb.masked_fill(t == 0, 0)
        z_ = torch.cat([z1, z2], dim=0)
        z0_ = torch.cat([z0, z0], dim=0)
        t_ = torch.cat([t, t+1], dim=0)
        w, logpf_w, logpb_w, logpt_w = self.sample_w(z_, z0_)
        logpf_w = logpf_w.sum(-1)
        logpb_w = logpb_w.sum(-1)
        z_hat = self.get_z_hat(w).detach()
        logpb_T, mu_T, sigma_T = self.get_logpb_z_z_hat(z0_, z_, z_hat)

        log_reward, metrics = self.get_log_reward(z0_, z_, w)
        g = self.get_g(z_, z0_, t_)
        logF = log_reward + logpb_T + logpb_w + g - logpf_w - logpt_w

        logF1 = logF[:batch_size]
        logF2 = logF[batch_size:]
        loss = (logF1 + logpf - logF2 - logpb).pow(2).mean()

        metrics.update({
            'dynamics/logF': logF.mean().item(),
            'dynamics/logpf': logpf.mean().item(),
            'dynamics/logpb': logpb.mean().item(),
            'dynamics/logpf_w': logpf_w.mean().item(),
            'dynamics/logpb_w': logpb_w.mean().item(),
            'dynamics/logpt_w': logpt_w.mean().item(),
            'dynamics/logpb_T': logpb_T.mean().item(),

            'dynamics/g': g.mean().item(),
            'dynamics/log_reward': log_reward.mean().item(),
            'dynamics/mu_f': mu_f.norm(dim=-1).mean().item(),
            'dynamics/sigma_f': sigma_f.norm(dim=-1).mean().item(),
            'dynamics/mu_b': mu_b.norm(dim=-1).mean().item(),
            'dynamics/sigma_b': sigma_b.norm(dim=-1).mean().item(),
            'dynamics/mu_T': mu_T.norm(dim=-1).mean().item(),
            'dynamics/sigma_T': sigma_T.norm(dim=-1).mean().item(),
            'dynamics/loss': loss.item()
        })
        return loss, metrics
    
    def get_discretizer_loss(self, batch_size):
        z0, _, z, w, t = self.e_step_replay_buffer.sample(batch_size)
        w, logpf, logpb, logpt = self.sample_w(z, z0=z0, p_explore=self.config.p_explore_discretizer)
            
        with torch.no_grad():
            z_hat = self.get_z_hat(w).detach()
            logpb_T, mu_T, sigma_T = self.get_logpb_z_z_hat(z0, z, z_hat)
            log_reward, reward_metrics = self.get_log_reward(z0, z, w)
            log_reward += logpb_T

        logZ = self.discretizer_model.flow_model(torch.cat([z0, z], dim=-1))
        h = torch.cat([z0, z], dim=-1) if self.config.z0_dependent_discretizer else z
        loss, metrics = self.discretizer_model.get_tb_loss(h, logpf, logpb, logpt, log_reward, logZ=logZ)
        metrics.update(reward_metrics)
        metrics['logpb_T'] = logpb_T.mean().item()
        metrics['mu_T'] = mu_T.norm(dim=-1).mean().item()
        metrics['sigma_T'] = sigma_T.norm(dim=-1).mean().item()
        metrics['log_reward'] = log_reward.mean().item()
        metrics['loss'] = loss.item()
        metrics = {f'discretizer/{k}': v for k, v in metrics.items()}
        return loss, metrics
    
    def get_e_step_loss(self):
        """
        x: (batch_size, ...)
        mode: 'wake', 'x_discretizer'
        """
        
        if (self.num_mode_updates > 0) & self.num_mode_updates % self.config.e_step_buffer_update_interval == 0:
            self.populate_e_step_buffer(256)
            
        dynamics_loss, dynamics_metrics = self.get_dynamics_loss(self.config.e_step_batch_size)
        discretizer_loss, discretizer_metrics = self.get_discretizer_loss(self.config.e_step_batch_size)
        loss = dynamics_loss + discretizer_loss #+ discretizerT_loss
        metrics = {**dynamics_metrics, **discretizer_metrics}#, **discretizerT_metrics}
        return loss, metrics
    
    def get_m_step_loss(self):
        if (self.num_mode_updates > 0) & self.num_mode_updates % self.config.m_step_buffer_update_interval == 0:
            self.populate_e_step_buffer(256)
            
        x, w = self.m_step_replay_buffer.sample(256)
        loss, metrics = self.m_model.get_loss(x=x, w=w)
        metrics = {f"m_step/{k.replace('/', '_')}": v for k, v in metrics.items()}
        metrics['m_step/loss'] = loss.item()
        return loss, metrics
    
    def get_em_loss(self):
        """
        Main training code should go here.
        training_step() will call this function.
        """
        if self.e_step: # E-step
            optimizer = self.get_optimizer('e_step')
            loss, metrics = self.get_e_step_loss()
            metrics['training/e_step_loss_threshold'] = self.e_step_loss_threshold
            if self.check_and_exit_e_step(loss):
                self.exit_e_step()
            
        else: # M-step
            optimizer = self.get_optimizer('m_step')
            loss, metrics = self.get_m_step_loss()
            self.num_m_steps += 1
            metrics['training/num_m_steps'] = self.num_m_steps
            if self.check_and_exit_m_step(loss):
                self.exit_m_step()
        return loss, optimizer, metrics
    
    def exit_e_step(self):
        self.num_mode_updates = 0
        self.e_step = False
        self.e_step_loss_threshold = min(self.e_step_loss_threshold, self.e_step_losses.mean().item())
        self.e_step_losses = np.zeros(self.config.e_step_loss_window)
        self.m_step_replay_buffer.reset()
        self.populate_m_step_buffer(self.config.m_step_start_rollouts)
        
    def exit_m_step(self):
        self.num_mode_updates = 0
        self.e_step = True
        self.m_model.freeze_encoder()
        self.e_step_replay_buffer.reset()
        self.populate_e_step_buffer(self.config.e_step_start_rollouts)

    @torch.no_grad()
    def check_and_exit_e_step(self, loss):
        """
        Returns whether or not the model should exit E-step.
        """
        if self.config.num_m_steps <= 0:
            return False
        
        self.num_mode_updates += 1
        self.e_step_losses = np.roll(self.e_step_losses, 1)
        self.e_step_losses[0] = loss.item()
        avg_loss = self.e_step_losses.mean()

        if self.config.num_m_steps <= 0:
            return False
        elif self.global_step <= self.config.start_e_steps:
            return False
        elif self.num_mode_updates < self.config.min_e_steps:
            return False
        elif self.config.max_e_steps is not None and self.num_mode_updates >= self.config.max_e_steps:
            pass
        elif self.config.e_loss_relaxation_rate is not None and avg_loss > self.e_step_loss_threshold:
            self.e_step_loss_threshold = self.e_step_loss_threshold * self.config.e_loss_relaxation_rate
            return False
        elif self.config.e_loss_improvement_rate is not None:
            avg_loss_1 = self.e_step_losses[:len(self.e_step_losses)//2].mean()
            avg_loss_2 = self.e_step_losses[len(self.e_step_losses)//2:].mean()
            if 1 - (avg_loss_2 / avg_loss_1) < self.config.e_loss_improvement_rate:
                return False
        elif self.global_step > self.config.e_step_unique_w_start and (self.config.e_step_min_unique_w > 0 or self.config.e_step_min_unique_tokens > 0):
            if self.data_module.batch_size < len(self.data_module.train_indices):
                indices = np.random.choice(self.data_module.train_indices, self.data_module.batch_size, replace=False)
            else:
                indices = self.data_module.train_indices
            batch = self.data_module.create_batch(indices, device=self.device)
            x = batch['x']

            # w = self.sample_w_from_x(x)
            z0 = self.get_z0(x)
            z_traj = self.sample_forward_trajectory(z0)
            w, _, _, _ = self.sample_w(z_traj[:,self.config.num_steps], z0)

            num_unique_w = len(w.unique(dim=0))
            num_unique_tokens = (w > 0).any(dim=0).sum()
            if num_unique_w < self.config.e_step_min_unique_w or num_unique_tokens < self.config.e_step_min_unique_tokens:
                return False

            z_hat = self.get_z_hat(w)
            distance = (z_hat - z_traj[:,self.config.num_steps]).norm(dim=-1)
            if distance.mean() > self.config.e_step_max_distance:
                return False

        return True
    
    def training_step(self, batch, batch_idx=None):
        loss, optimizer, metrics = self.get_em_loss()

        optimizer.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        
        if self.global_step%100 == 0:
            metrics.update(self.get_performance_metrics())
        if self.config.create_plots_every is not None and self.global_step % self.config.create_plots_every == 0 and isinstance(self.logger, WandbLogger):
            self.create_plots()

        self.log_metrics(metrics)
        if self.config.save_dir is not None and self.global_step % self.config.save_checkpoint_every == 0:
            self.save()
    
    @torch.no_grad()
    def get_performance_metrics(self, n=256):
        indices = np.random.choice(self.data_module.train_indices, n)
        batch = self.data_module.create_batch(indices, device=self.device)
        x = batch['x']
        
        z0 = self.get_z0(x)
        z_traj = self.sample_forward_trajectory(z0)
        w, _, _, _ = self.sample_w(z_traj[:,-1], z0)
        log_reward, reward_metrics, score_results = self.get_log_reward(z=z_traj[:,-1], w=w, z0=z0, return_score_result=True)
        distance = (score_results.z_hat - z_traj[:,-1]).norm(dim=-1)
        num_unique_sentences = w.unique(dim=0).shape[0]
        num_unique_tokens = (w > 0).any(dim=0).sum().item()
        w_length = w.sum(-1).mean().item()
        num_unique_e_step_replay_buffer_sentences = self.e_step_replay_buffer.w.unique(dim=0).shape[0]
        num_unique_e_step_replay_buffer_tokens = (self.e_step_replay_buffer.w > 0).any(dim=0).sum().item()
        e_step_w_length = self.e_step_replay_buffer.w.sum(-1).mean().item()
        num_unique_m_step_replay_buffer_sentences = self.m_step_replay_buffer.w.unique(dim=0).shape[0]
        num_unique_m_step_replay_buffer_tokens = (self.m_step_replay_buffer.w > 0).any(dim=0).sum().item()
        m_step_w_length = self.m_step_replay_buffer.w.sum(-1).mean().item()
        
        
        metrics = {
            'training/zT_score': score_results.score.mean().item(),
            'training/zT_log_reward': log_reward.mean().item(),
            'training/zT_distance': distance.mean().item(),
            'training/num_unique_sentences': num_unique_sentences,
            'training/num_unique_tokens': num_unique_tokens,
            'training/w_length': w_length,
            'replay_buffer/e_step_unique_sentences': num_unique_e_step_replay_buffer_sentences,
            'replay_buffer/e_step_unique_tokens': num_unique_e_step_replay_buffer_tokens,
            'replay_buffer/e_step_w_length': e_step_w_length,
            'replay_buffer/m_step_unique_sentences': num_unique_m_step_replay_buffer_sentences,
            'replay_buffer/m_step_unique_tokens': num_unique_m_step_replay_buffer_tokens,
            'replay_buffer/m_step_w_length': m_step_w_length
        }
        return metrics
    
    @torch.no_grad()
    def sanity_test(self, n=100, plot: bool = False):
        indices = np.random.choice(np.arange(len(self.data_module)), n)
        batch = self.data_module.create_batch(indices, device=self.device)
        x = batch['x']
        
        print("Populating replay buffer")        
        self.populate_e_step_buffer(n, 'wake')
        self.populate_e_step_buffer(n, 'explore')
        self.populate_e_step_buffer(n, 'sleep')
        self.populate_m_step_buffer(n)
        
        print("E-step")
        loss, metrics = self.get_e_step_loss()
        print(metrics)
        
        print("M-step")
        loss, metrics = self.get_m_step_loss()
        print(metrics)
        
        print("Performance metrics")
        print(self.get_performance_metrics(n))
        
        if plot:
            print("Creating plots")
            self.create_plots()
            
    def create_plots(self):
        images = self.create_pca_gif(500)
        self.log_gif('pca', images)

    ####################################################################################################
    ######################################### Analyses #################################################
    ####################################################################################################
    
    def get_svd(self, x, pca_mode='z0'):
        z0 = self.get_z0(x)
        if pca_mode == 'z0':
            return torch.linalg.svd(z0)
        
        z_traj_f = self.sample_forward_trajectory(z0)
        if pca_mode == 'zT':
            u, s, v = torch.linalg.svd(z_traj_f[:,-1])
        elif pca_mode == 'z_traj':
            u, s, v = torch.linalg.svd(z_traj_f.flatten(0, 1))
        elif pca_mode == 'z_hat':
            w, _, _, _ = self.sample_w(z_traj_f[:,-1], z0)
            z_hat = self.get_z_hat(w)
            u, s, v = torch.linalg.svd(z_hat)
        else:
            raise ValueError(f'pca_mode={pca_mode}')
        return u, s, v
    
    @torch.no_grad()
    def plot_2d_step(self, df_traj, step, df_zhat=None, color: str = None, shape: str = None, scale=1):
        """
        df_traj: DataFrame with columns ['step', 'pc1', 'pc2']
        df_zhat: DataFrame with columns ['pc1', 'pc2']
        color: str or list of str containing the column names in df_traj
        shape: str or list of str containing the column names in df_traj
        """
        xlim = scale * df_traj.pc1.min(), scale * df_traj.pc1.max()
        ylim = scale * df_traj.pc2.min(), scale * df_traj.pc2.max()
        df_traj = df_traj[df_traj.step == step].copy()

        kwargs = {}
        if color is not None:
            if isinstance(color, str):
                kwargs['color'] = color
            else:
                df_traj['color'] = [', '.join(f"{k}: {r[k]}" for k in color) for r in df_traj.to_records()]
                kwargs['color'] = 'color'
        if shape is not None:
            if isinstance(shape, str):
                kwargs['shape'] = shape
            else:
                df_traj['shape'] = [', '.join(f"{k}: {r[k]}" for k in shape) for r in df_traj.to_records()]
                kwargs['shape'] = 'shape'
        plot = (ggplot(df_traj, aes(x='pc1', y='pc2'))
                + geom_point(aes(**kwargs), size=1, alpha=.5))
        
        if df_zhat is not None:
            plot = plot + geom_point(data=df_zhat, size=1, color='black')
        plot = (plot 
                + coord_cartesian(xlim=xlim, ylim=ylim)
                + labs(title=f"Step {step}")
                + theme_bw())
        return plot
    
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


# def suppress_distance(distance: torch.Tensor, odds: float, log_base: float = 2):
#     """
#     Given a distance tensor, suppresses the distance by clamping it to a and taking the log of the remaining distance.

#     d: tensor with shape (...)
#     odds: float
#     log_base: float
#     returns: tensor with shape (...)
#     """
#     a = np.log(odds) ** 2
#     return d.clamp(0, a) + (d - a + 1).log() / torch.log(log_base)
