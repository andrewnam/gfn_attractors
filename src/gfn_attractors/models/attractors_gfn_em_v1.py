import numpy as np
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
import einops
import pytorch_lightning as pl
from plotnine import *

from .gfn_em import GFNEM, GFNEMConfig
from .bitmap_gfn import BitmapGFN
from ..misc import torch_utils as tu, image_utils as iu
from ..misc.replay_buffer import TrajectoryReplayBuffer
from .helpers import MLP, PositionalEncoding
from .dynamics import MLPMeanBoundedDynamics, MLPLangevinMeanBoundedDynamics


@dataclass
class AttractorsGFNEMConfig(GFNEMConfig):

    attractor_sd: float = 1.

    distance_linear_odds: float = 10000 # After 10000x less likely, the distance is suppressed by log_base
    distance_log_base: float = 2

    dim_t: int = 10
    dynamics_dim_h: int = 256
    dynamics_num_layers: int = 3
    num_steps: int = 25
    num_extra_steps: int = 0
    max_mean: float = 0.15 # If None, uses max_travel to calculate max_mean. Otherwise, uses this value.
    zT_max_mean: float|None = None
    max_sd: float = 1.
    min_sd: float = 0.
    z0_dependent_forward: bool = False
    z0_dependent_backward: bool = True
    t_dependent_forward: bool = False
    t_dependent_backward: bool = True
    directional_dynamics: bool = False
    langevin_dynamics: bool = False

    # Discretizer
    z0_dependent_discretizer: bool = False

    # Replay buffer
    replay_buffer_inv_freq_sequence: bool = False
    replay_buffer_inv_freq_token: bool = False
    replay_buffer_size: int = 100000
    dynamics_batch_size: int = 2000
    discretizer_batch_size: int = 2000
    buffer_update_interval: int = 1

    # Training
    e_step_max_distance: float = 1000
    p_sample_x_discretizer: float = 0.
    lr_dynamics: float = 1e-4
    p_explore_dynamics = 0.05
    f_half_sleep: float = 0.
    p_exploration_path: float = 0.
    p_explore_exploration_path: float = .3



class AttractorsGFNEM(GFNEM):

    def __init__(self, config: AttractorsGFNEMConfig, data_module):
        super().__init__(config, data_module)
        self.config = config
        self.temporal_encoding = PositionalEncoding(config.dim_t) if config.dim_t is not None else None
        self.g_model = MLP(2 * config.dim_z + config.dim_t, 1, hidden_dim=config.flow_dim_h, n_layers=config.flow_num_layers, nonlinearity=nn.ReLU())
        self.dynamics_model = self.init_dynamics()
        self.discretizer_model = self.init_discretizer()
        self.replay_buffer = TrajectoryReplayBuffer(config.dim_z, size=100000, vocab_size=config.vocab_size,
                                                    inv_freq_sequence=self.config.replay_buffer_inv_freq_sequence, 
                                                    inv_freq_token=self.config.replay_buffer_inv_freq_token)
        
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


    def init_dynamics(self):
        if self.config.langevin_dynamics:
            energy_func = lambda z, z0: self.get_discretizer_log_Z(z0, z)
            return MLPLangevinMeanBoundedDynamics(dim_x=self.config.dim_z, dim_z=self.config.dim_z, max_mean=self.config.max_mean,
                                                  energy_func=energy_func, 
                                                  directional=self.config.directional_dynamics,
                                                  dim_t=self.config.dim_t, dim_h=self.config.dynamics_dim_h, 
                                                  num_layers=self.config.dynamics_num_layers, 
                                                  min_sd=self.config.min_sd, 
                                                  max_sd=self.config.max_sd, 
                                                  fixed_sd=None, 
                                                  t_dependent_forward=self.config.t_dependent_forward,
                                                  t_dependent_backward=self.config.t_dependent_backward,
                                                  x_dependent_forward=self.config.z0_dependent_forward,
                                                  x_dependent_backward=self.config.z0_dependent_backward)
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
                         dim_flow_input=dim_input)
    
    def add_optimizers(self):
        e_step = [{'params': self.dynamics_model.parameters(), 'lr': self.config.lr_dynamics},
                  {'params': self.discretizer_model.parameters(), 'lr': self.config.lr_discretizer},
                  {'params': [*self.g_model.parameters()], 'lr': self.config.lr_flows}]
        if self.config.zT_max_mean is not None:
            e_step.append({'params': self.zT_dynamics_model.parameters(), 'lr': self.config.lr_dynamics})
        m_step = []
        others = {}
        return e_step, m_step, others
    
    def get_z0(self, x):
        return self.m_model.get_z0(x)

    def get_z_hat(self, w):
        return self.m_model.get_z_hat(w)
    
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
    
    def get_discretizer_log_Z(self, z0, z):
        """
        z0: tensor with shape (batch_size, dim_z)
        z: tensor with shape (batch_size, ..., dim_z)
        returns: tensor with shape (batch_size, ...)
        """
        if not self.config.z0_dependent_discretizer:
            return self.discretizer_model.flow_model(z)
        if z0.ndim == z.ndim:
            return self.discretizer_model.flow_model(torch.cat([z0, z], dim=-1))
        if z0.ndim != 2:
            raise ValueError(f"z0 should have 2 dimensions or equal dimensions as z, but got {z0.ndim} and {z.ndim}.")
        batch_size = z0.shape[0]
        batch_shape = z.shape[:-1]
        z = z.view(batch_size, -1, self.config.dim_z)
        z0 = einops.repeat(z0, 'b z -> (b k) z', k=z.shape[1])
        z = z.view(-1, self.config.dim_z)
        log_Z = self.discretizer_model.flow_model(torch.cat([z0, z], dim=-1))
        return log_Z.view(*batch_shape)
    
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

    # @torch.no_grad()
    # def get_log_reward(self, z0, z, w, distance_weight=1, return_score_result=False):
    #     """
    #     z0: (batch_size, dim_z)
    #     z: (batch_size, dim_z)
    #     w: (batch_size, max_w_length)
    #     """
    #     metrics = {}
    #     score_result = self.m_model(x=None, w=w, z=z, z0=z0)
    #     # decodability = self.get_logpw_z(score_result.z_hat, score_result.z_hat, score_result.z_hat_sd)
    #     decodability = score_result.logpz_zhat
    #     log_reward = self.config.score_weight * score_result.score + self.config.decodability_weight * decodability

    #     # if self.config.contrasts_weight > 0:
    #     #     log_reward += score_result.contrasts_score
    #     #     metrics['reward/contrasts_score'] = score_result.contrasts_score.mean().item()

    #     log_reward = log_reward.clamp(self.config.min_log_reward, self.config.max_log_reward)
    #     logpz_w = Normal(score_result.z_hat, self.config.attractor_sd).log_prob(value=z).sum(-1)
    #     log_reward = log_reward + distance_weight * logpz_w
    #     metrics.update({'reward/score': score_result.score.mean().item(),
    #                'reward/logpz_w': logpz_w.mean().item(),
    #                'reward/decodability': decodability.mean().item(),
    #                'reward/total': log_reward.mean().item()})
    #     if return_score_result:
    #         return log_reward, metrics, score_result
    #     return log_reward, metrics
    
    def correct_backward_trajectory(self, z_traj, z0, delay=0, beta=.5):
        if delay == 0:
            return tu.correct_trajectory(z_traj.flip(dims=[1]), z0, beta=beta).flip(dims=[1])
        else:
            corrected = tu.correct_trajectory(z_traj[:,:-delay].flip(dims=[1]), z0, beta=beta).flip(dims=[1])
            corrected = torch.cat([corrected, z_traj[:,-delay:]], dim=1)
            return corrected

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
    
    # @torch.no_grad()
    # def add_exploration_to_buffer(self, x):
    #     """
    #     Given a z_traj, sample a w at each step and sample among the unique w's according to their scores.
    #     """
    #     z0 = self.get_z0(x)
    #     z_traj = self.sample_forward_trajectory(z0, p_explore=self.config.p_explore_exploration_path)
    #     w, _, _, _ = self.sample_w(z_traj, z0, p_explore=self.config.p_explore_discretizer)
    #     log_reward, _ = self.get_log_reward(z0, z_traj, w, distance_weight=0)

    #     ws = []
    #     lrs = []
    #     for w_i, lr_i in zip(w.cpu().numpy(), log_reward.cpu().numpy()):
    #         w_unique, idx = np.unique(w_i, axis=0, return_index=True)
    #         j = Categorical(logits=torch.tensor(lr_i[idx])).sample()
    #         w_i = w_unique[j]
    #         ws.append(w_i)
    #         lrs.append(lr_i[j])
    #     w = torch.tensor(np.array(ws), dtype=torch.float32, device=z_traj.device)
    #     z_hat = self.get_z_hat(w)
    #     zT = z_hat + torch.randn_like(z_hat) * self.config.attractor_sd
    #     z_traj = self.sample_backward_trajectory(zT, z0)
    #     z_traj = self.correct_backward_trajectory(z_traj, z0)
    #     lrs = torch.tensor(np.array(lrs), dtype=torch.float32, device=z_traj.device)
    #     self.replay_buffer.add(z0, z_traj, w)
    #     self.log_metrics({
    #         'replay_buffer/sampled_exploration_score': lrs.median().item(),
    #         'replay_buffer/exploration_score': log_reward.median().item()
    #     })

    @torch.no_grad()
    def add_exploration_paths(self, x):
        z0 = self.get_z0(x)
        z_traj = self.sample_forward_trajectory(z0, p_explore=self.config.p_explore_exploration_path)
        w, _, _, _ = self.sample_w(z_traj, z0, p_explore=self.config.p_explore_discretizer)
        z_hat = self.get_z_hat(w)
        z0_ = einops.repeat(z0, 'b z -> (b t) z', t=z_traj.shape[1])
        log_reward, metrics, score_result = self.get_log_reward(z0_, z_traj.flatten(0, 1), w.flatten(0, 1), return_score_result=True)
        log_reward = log_reward.view(z_traj.shape[:-1])

        w_ = w.bool().cpu().numpy()
        w_counts = []
        for w_i in w_:
            w_unique, inverse, counts = np.unique(w_i, axis=0, return_inverse=True, return_counts=True)
            w_counts.append(counts[inverse])
        w_counts = np.array(w_counts)
        logits = log_reward - torch.tensor(w_counts, device=w.device).log()
        indices = Categorical(logits=logits).sample()
        indices = einops.repeat(indices, 'b -> b 1 w', w=w.shape[-1])
        wT = w.gather(1, indices).squeeze(1)

        z_hat = self.get_z_hat(wT)
        z_traj_b = self.sample_backward_trajectory(z_hat, z0)
        z_traj_b = self.correct_backward_trajectory(z_traj_b, z0, beta=.5)
        self.replay_buffer.add(z0, z_traj_b, wT)

        lr_new = log_reward.gather(1, indices[:,:,0]).squeeze()
        lr_old = log_reward[:,-1]
        lr_diff = (lr_new - lr_old).mean().item()
        self.log_metrics({'training/exploration_score_diff': lr_diff})


    @torch.no_grad()
    def add_trajectories_to_buffer(self, x):
        batch_size = len(x)
        num_steps = self.config.num_steps + self.config.num_extra_steps
        metrics = {}

        z0s = []
        z_trajs = []
        ws = []
        z_hats = []
        sleep_idx = int(batch_size * self.config.p_sleep_phase)
        x_discretizer_idx = sleep_idx + int(batch_size * self.config.p_sample_x_discretizer)
        # path_explore_idx = x_discretizer_idx + int(batch_size * self.config.f_exploration_path)

        if x_discretizer_idx > 0:
            if self.config.p_sleep_phase > 0: # Sleep phase, sample w from prior
                w = self.sample_w_from_prior(sleep_idx, dtype=torch.float32)
                z0, z_hat = self.m_model.sample(w)
                z0s.append(z0)
                z_hats.append(z_hat)
                ws.append(w)
            if self.config.p_sample_x_discretizer > 0: # Half-sleep phase, sample w from x_discretizer
                x_ = x[sleep_idx:x_discretizer_idx]
                w, _, _, _ = self.sample_w_x(x_)
                z0 = self.get_z0(x_)
                z_hat = self.get_z_hat(w)
                z0s.append(z0)
                z_hats.append(z_hat)
                ws.append(w)
            z0 = torch.cat(z0s, dim=0)
            z_hat = torch.cat(z_hats, dim=0)
            z_terminal = z_hat + torch.randn_like(z_hat) * self.config.attractor_sd
            z_traj = self.sample_backward_trajectory(z_terminal, z0)
            metrics['backward_traj_correction'] = (z_traj[:,0] - z0).norm(dim=-1).mean()
            z_traj = self.correct_backward_trajectory(z_traj, z0)
            z_trajs.append(z_traj)
        
        z0 = self.get_z0(x[x_discretizer_idx:])
        z0s.append(z0)
        z_traj = self.sample_forward_trajectory(z0, num_steps=num_steps, p_explore=self.config.p_explore_dynamics)
        w, _, _, _ = self.sample_w(z_traj[:,self.config.num_steps], z0)
        ws.append(w)
        z_trajs.append(z_traj)
        z_traj = torch.cat(z_trajs, dim=0)
        z0 = torch.cat(z0s, dim=0)
        w = torch.cat(ws, dim=0)
        self.replay_buffer.add(z0, z_traj, w)

    def get_discretizer_loss(self, batch_size):
        z0, _, z, w, t = self.replay_buffer.sample(batch_size)
        w, logpf, logpb, logpt = self.sample_w(z, z0=z0, p_explore=self.config.p_explore_discretizer)
            
        with torch.no_grad():
            z_hat = self.get_z_hat(w).detach()
            logpb_T, mu_T, sigma_T = self.get_logpb_z_z_hat(z0, z, z_hat)
            log_reward, reward_metrics = self.get_log_reward(z0, z, w)
            log_reward += logpb_T

        h = torch.cat([z0, z], dim=-1)
        loss, metrics = self.discretizer_model.get_tb_loss(h, logpf, logpb, logpt, log_reward)
        metrics.update(reward_metrics)
        metrics['logpb_T'] = logpb_T.mean().item()
        metrics['mu_T'] = mu_T.norm(dim=-1).mean().item()
        metrics['sigma_T'] = sigma_T.norm(dim=-1).mean().item()
        metrics['log_reward'] = log_reward.mean().item()
        metrics['loss'] = loss.item()
        metrics = {f'discretizer/{k}': v for k, v in metrics.items()}
        return loss, metrics
    
    def get_dynamics_loss(self, batch_size):
        z0, z1, z2, w, t = self.replay_buffer.sample(batch_size)
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
        # _, _, logpb_T, _, _, mu_T, sigma_T = self.zT_dynamics_model.step(z_hat, z0_, t_, target=z_, return_params=True)

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
    
    def get_e_step_loss(self, batch):
        """
        x: (batch_size, ...)
        mode: 'wake', 'x_discretizer'
        """
        x = batch['x']
        if len(self.replay_buffer) < self.replay_buffer.size or self.global_step % self.config.buffer_update_interval == 0:
            if np.random.rand() < self.config.p_exploration_path:
                self.add_exploration_paths(x)
            else:
                self.add_trajectories_to_buffer(x)
        dynamics_loss, dynamics_metrics = self.get_dynamics_loss(self.config.dynamics_batch_size)
        discretizer_loss, discretizer_metrics = self.get_discretizer_loss(self.config.discretizer_batch_size)
        loss = dynamics_loss + discretizer_loss #+ discretizerT_loss
        metrics = {**dynamics_metrics, **discretizer_metrics}#, **discretizerT_metrics}
        return loss, metrics
    
    def get_m_step_loss(self, batch, p_explore=None, argmax=None):
        x = batch['x']
        if p_explore is None:
            p_explore = self.config.m_step_p_explore
        if argmax is None:
            argmax = self.config.m_step_argmax
        with torch.no_grad():
            z0 = self.get_z0(x)
            z_traj = self.sample_forward_trajectory(z0)
            z_T = z_traj[:,self.config.num_steps]
            
            if self.config.m_step_max_w_length:
                w, logpf, logpb, logpt = self.sample_w(z_T, z0,
                                                    min_steps=self.config.max_w_length, 
                                                    p_explore=p_explore,
                                                    temperature=self.config.m_step_temperature,
                                                    allow_terminate=self.config.m_step_substrings,
                                                    argmax=argmax)
            else:
                w, logpf, logpb, logpt = self.sample_w(z_T, z0,
                                                       p_explore=p_explore,
                                                       temperature=self.config.m_step_temperature,
                                                       allow_terminate=True,
                                                       argmax=argmax)
                if self.config.m_step_substrings:
                    subw = np.zeros((len(w), w.sum(-1).max().long().item(), w.shape[-1]))
                    for i, w_i in enumerate(w.cpu().numpy()):
                        indices = w_i.nonzero()[0]
                        np.random.shuffle(indices)
                        for j, k in enumerate(indices):
                            subw[i, j] = w_i
                            w_i = w_i.copy()
                            w_i[k] = 0
                    w = torch.tensor(subw, dtype=torch.float32, device=w.device)
        loss, metrics = self.m_model.get_loss(w=w, **batch)

        metrics = {f"m_step/{k.replace('/', '_')}": v for k, v in metrics.items()}
        metrics['m_step/loss'] = loss.item()
        return loss, metrics

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

        self.num_mode_updates = 0
        self.e_step = False
        self.e_step_loss_threshold = min(self.e_step_loss_threshold, avg_loss.item())
        self.e_step_losses = np.zeros(self.config.e_step_loss_window)
        self.replay_buffer.reset()
        return True
    
    def get_em_loss(self, batch):
        """
        Main training code should go here.
        training_step() will call this function.
        """
        if self.e_step: # E-step
            optimizer = self.get_optimizer('e_step')
            loss, metrics = self.get_e_step_loss(batch)
            self.check_and_exit_e_step(loss)
            metrics['training/e_step_loss_threshold'] = self.e_step_loss_threshold
        else: # M-step
            optimizer = self.get_optimizer('m_step')
            loss, metrics = self.get_m_step_loss(batch)
            self.check_and_exit_m_step(loss)
            self.num_m_steps += 1
            metrics['training/num_m_steps'] = self.num_m_steps
        return loss, optimizer, metrics
    
    @torch.no_grad()
    def get_performance_metrics(self, batch):
        x = batch['x']
        z0 = self.get_z0(x)
        z_traj = self.sample_forward_trajectory(z0)
        w, _, _, _ = self.sample_w(z_traj[:,self.config.num_steps], z0, temperature=self.config.m_step_temperature)
        log_reward, reward_metrics, score_results = self.get_log_reward(z=z_traj[:,self.config.num_steps], w=w, z0=z0, return_score_result=True)
        distance = (score_results.z_hat - z_traj[:,self.config.num_steps]).norm(dim=-1)
        num_unique_sentences = w.unique(dim=0).shape[0]
        num_unique_tokens = (w > 0).any(dim=0).sum().item()
        metrics = {
            'training/zT_score': score_results.score.mean().item(),
            'training/zT_log_reward': log_reward.mean().item(),
            'training/zT_distance': distance.mean().item(),
            'training/num_unique_sentences': num_unique_sentences,
            'training/num_unique_tokens': num_unique_tokens,
            # 'training/min_attractor_sd': self.attractor_sd.min().item(),
            # 'training/mean_attractor_sd': self.attractor_sd.mean().item(),
        }
        if self.config.langevin_dynamics:
            metrics['training/alpha'] = self.dynamics_model.last_alpha.mean().item()
        return metrics
    

    @torch.no_grad()
    def sanity_test(self, n=100, plot: bool = False):
        indices = np.random.choice(np.arange(len(self.data_module)), n)
        batch = self.data_module.create_batch(indices)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        x = batch['x']
        print("Populating replay buffer")
        self.add_trajectories_to_buffer(x)
        self.add_exploration_paths(x)
        print("E-step")
        loss, metrics = self.get_e_step_loss(batch)
        print(metrics)
        print("M-step")
        loss, metrics = self.get_m_step_loss(batch)
        print(metrics)
        print("Performance metrics")
        print(self.get_performance_metrics(batch))
        if plot:
            print("Creating plots")
            self.create_plots(batch)

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
