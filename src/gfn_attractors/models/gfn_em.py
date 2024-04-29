from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import os
import yaml
from functools import cached_property
from datetime import datetime
import io
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
import wandb

import torch
import einops
import pytorch_lightning as pl
from tqdm.auto import trange

from .bitmap_gfn import BitmapGFN
from ..misc.config import Config
from ..misc import torch_utils as tu


@dataclass
class GFNEMConfig(Config):

    # Shared
    dim_z: int
    vocab_size: int
    max_w_length: int
    vocab_group_size: int|None = None
    min_w_length: int = 1

    seed: int|None = None
    save_dir: str|None = None # Where to save the model checkpoints. If None, doesn't save.
    save_prefix: str|None = None # Prefix for the saved model files.
    save_checkpoint_every: int = 1000 # How often to save checkpoints (in number of updates)
    create_plots_every: int|None = None # How often to create plots (in number of updates)

    # Reward
    score_weight: float = 1.
    decodability_weight: float = 1.
    contrasts_weight: float = 1.
    length_penalty: float = 0.
    max_log_reward: float = 1e10
    min_log_reward: float = -1e10

    # Discretizer
    discretizer_model: str = 'mlp'
    discretizer_objective: str = 'tb' # 'tb', 'subtb', 'db'
    discretizer_fixed_backward_policy: bool = False
    discretizer_dim_h: int = 128
    discretizer_num_layers: int = 3

    # Flow models
    flow_dim_h: int = 128
    flow_num_layers: int = 3

    # M-Model
    m_model_dim_h: int = 256
    m_model_num_layers: int = 3
    m_model_num_w_embedding_layers: int = 1
    m_model_vae_beta: float = 1.
    m_model_cvae_beta: float = 1.

    # Training
    lr_discretizer: float = 1e-3
    lr_flows: float = 1e-3
    lr_m_step: float = 1e-4
    p_sleep_phase: float = 0. # probability of sleeping during training
    p_explore_discretizer: float = .05

    # EM
    start_e_steps: int = 0 # number of e-steps to start with, ignoring all other conditions
    start_m_steps: int = 0
    min_e_steps: int = 1 # number of e-steps to perform before checking loss-based conditions
    max_e_steps: int|None = None # If number of e-steps exceeds this, switch to M-step regardless of loss-based conditions.
    e_step_unique_w_start: int = 10000 # track unique_w after this many updates
    e_step_min_unique_w: int = 0 # If the number of unique w is less than this, keep updating E-step
    e_step_min_unique_tokens: int = None # If the number of unique tokens is less than this, keep updating E-step. If None, defaults to vocab_size
    # e_step_unique_w_margin: int|None = None # If not None, switch to M-step only if the number of
    #                                         # unique w is at most this much less than the maximum number of unique w seen so far
    e_step_loss_window: int = 1 # number of steps to average loss over
    e_loss_improvement_rate: float|None = None # if the average loss improves by at least this much, keep updating E-step
    e_loss_relaxation_rate: float|None = None # If not None, at the end of each E-step, records the average loss as the threshold.
                                         # During the next E-step, check if the current average loss is less than the threshold.
                                         # If not, keep performing E-step and increase the threshold by this rate.
    num_m_steps: int = 1
    m_step_temperature: float = 1. # temperature for sampling P(w|z) during M-step
    m_step_argmax: bool = False # if True, uses argmax instead of sampling P(w|z) during M-step
    m_step_p_explore: float = 0. # probability of exploring during M-step
    m_step_substrings: bool = True
    m_step_max_w_length: bool = False

    def __post_init__(self):
        if self.length_penalty > 0:
            raise ValueError("Length penalty should not be positive.")
        if self.e_step_min_unique_tokens is None:
            self.e_step_min_unique_tokens = self.vocab_size


class GFNEM(pl.LightningModule, ABC):

    def __init__(self, 
                 config: GFNEMConfig, 
                 data_module):
        super().__init__()
        self.config = config
        self.data_module = data_module
        self.automatic_optimization = False

        self.x_discretizer_model = self.init_x_discretizer()
        self.m_model = self.init_m_model()
        self.p_w_z_model = BitmapGFN(self.config.dim_z, num_bits=self.config.vocab_size, group_size=self.config.vocab_group_size,
                                     dim_h=self.config.discretizer_dim_h, num_layers=self.config.discretizer_num_layers)

        self.e_step = self.config.max_e_steps is None or self.config.max_e_steps > 0
        if self.config.start_e_steps == 0 and self.config.start_m_steps > 0:
            self.e_step = False
        self.num_mode_updates = 0
        self.num_m_steps = 0
        self.e_step_loss_threshold = np.inf
        self.e_step_losses = np.zeros(config.e_step_loss_window)
        self.e_step_update_step = 0
        self.max_last_num_unique_w = 0
        self.metrics = {'training/num_e_steps': 0}

        self.num_unique_w = 0

    @classmethod
    def load_latest(cls, config_class, data_module, save_dir):
        paths = list(Path(save_dir).glob('*.pt'))
        paths.sort(key=os.path.getmtime, reverse=True)
        latest_save = paths[0]
        config = yaml.load(open(f'{save_dir}/config.yaml', 'r'), Loader=yaml.FullLoader)
        config = config_class(**config)
        model = cls(config, data_module)
        missing = tu.load_partial_state_dict(model, torch.load(latest_save))    
        print(f"Loaded {latest_save}")
        if len(missing) == 0:
            print("All keys found in the model.")
        else:
            print(f"{len(missing)} mismatched keys were found.")
        return model, missing

    @cached_property
    def optimizer_indices(self):
        return {k: i for i, k in enumerate(sorted(self.optimizers_dict))}
    
    @abstractmethod
    def init_m_model(self, **kwargs):
        raise NotImplementedError
    
    def init_x_discretizer(self):
        raise NotImplementedError
    
    def add_optimizers(self):
        e_step = []
        m_step = []
        others = {}
        return e_step, m_step, others
    
    def init_optimizers(self):
        other_e_step, other_m_step, others = self.add_optimizers()
        e_step = torch.optim.Adam([{'params': [*self.x_discretizer_model.parameters(),
                                               *self.p_w_z_model.parameters()], 'lr': self.config.lr_discretizer},
                                   *other_e_step])
        m_step = torch.optim.Adam([{'params': self.m_model.parameters(), 'lr': self.config.lr_m_step},
                                   *other_m_step])
        optimizers = {
            'e_step': e_step,
            'm_step': m_step,
            **others
        }

        self.optimizers_dict = optimizers
    
    def configure_optimizers(self):
        return [self.optimizers_dict[k] for k in sorted(self.optimizers_dict)]
    
    def get_optimizer(self, optimizer_name):
        if self._trainer is None:
            return None
        index = self.optimizer_indices[optimizer_name]
        return self.optimizers()[index]
    
    def sample_substrings(self, w):
        return self.x_discretizer_model.sample_substrings(w)

    def sample_w_from_prior(self, n, dtype=bool):
        return self.x_discretizer_model.sample_from_prior(n, self.config.max_w_length, dtype=dtype, device=self.device)
    
    @torch.no_grad()
    def sample_w_from_x(self, x, *args, **kwargs):
        w, _, _, _ = self.sample_w(x, *args, **kwargs)
        return w

    def sample_w(self, x, min_steps=None, max_steps=None, target=None, temperature=1., argmax=False, allow_terminate=True, p_explore=0.):
        """
        x: tensor with shape [batch_size, *input_shape]
        target: tensor with shape (batch_size, num_bits)
        returns:
            w: (batch_size, num_bits) or (batch_size, 1 + max_steps, num_bits)
            logpf: (batch_size, max_steps)
            logpb: (batch_size, max_steps)
            logpt: (batch_size) or (batch_size, 1 + max_steps)
        """
        if min_steps is None:
            min_steps = self.config.min_w_length
        if max_steps is None:
            max_steps = self.config.max_w_length
        return self.x_discretizer_model.sample(x, min_steps=min_steps, max_steps=max_steps, target=target, temperature=temperature, argmax=argmax, 
                                               allow_terminate=allow_terminate, p_explore=p_explore)
    
    def get_z0(self, x):
        return self.m_model.get_z0(x)
    
    def get_z_hat(self, w):
        return self.m_model.get_z_hat(w)

    def get_logpw_z(self, z, z_hat, z_hat_sd):
        """
        z: tensor with shape [batch_size, dim_z] or [batch_size, ..., dim_z]
        z_hat: tensor with shape [batch_size, ..., dim_z]
        z_hat_sd: tensor with shape [batch_size, ..., dim_z]
        returns: tensor with shape [batch_size, ...]
        """
        logpz_zhat = self.m_model.get_logpz_zhat(z, z_hat, z_hat_sd)
        batch_shape = logpz_zhat.shape
        if z.ndim < logpz_zhat.ndim:
            logpz_zhat = logpz_zhat.view(z.shape[0], -1)
            z = einops.repeat(z, 'b z -> (b k) z', k=logpz_zhat.shape[1])
        else:
            z = z.view(-1, self.config.dim_z)
        logpz_zhat = logpz_zhat.flatten()
        logpw_z = self.p_w_z_model.get_logpw(z, logpz_zhat).view(*batch_shape)
        return logpw_z

    def get_logpw_z_loss(self, z):
        """
        z: tensor with shape [batch_size, dim_z]
        """
        w, logpf, logpb, logpt = self.p_w_z_model.sample(z.detach(), max_steps=self.config.max_w_length,
                                                         p_explore=self.config.p_explore_discretizer,
                                                         allow_terminate=self.config.discretizer_objective == 'tb')
        z_hat, z_hat_sd = self.m_model.get_z_hat(w, return_sd=True)
        with torch.no_grad():
            log_reward = self.m_model.get_logpz_zhat(z_hat, z_hat, z_hat_sd).clamp(self.config.min_log_reward, self.config.max_log_reward)
            # log_reward = self.m_model.get_logpz_zhat(z, z_hat, z_hat_sd).clamp(self.config.min_log_reward, self.config.max_log_reward)
        loss, metrics =  self.p_w_z_model.get_tb_loss(z, logpf, logpb, logpt, log_reward)
        metrics = {f'logpw_z/{k}': v for k, v in metrics.items()}
        return loss, metrics

    @torch.no_grad()
    def get_log_reward(self, x, w, z0=None, return_score_result=False):
        """
        x: tensor with shape [batch_size, *input_shape]
        w: tensor with shape [batch_size, ..., vocab_size]
        z0: tensor with shape [batch_size, dim_z]
        returns: tensor with shape [batch_size, ...]
        """
        score_results = self.m_model(x, w, z0)
        log_reward = self.config.score_weight * score_results.score
        metrics = {'score': score_results.score.mean().item()}
        if self.config.decodability_weight > 0:
            logpw_z = self.get_logpw_z(score_results.z_hat, score_results.z_hat, score_results.z_hat_sd)
            log_reward += self.config.decodability_weight * logpw_z
            metrics['logpw_z'] = logpw_z.mean().item()
        if self.config.length_penalty < 0:
            length = w.sum(-1)
            log_reward += self.config.length_penalty * length
            metrics['length'] = length.mean().item()
        # if self.config.contrasts_weight > 0:
        #     log_reward += score_results.contrasts_score
        #     metrics['contrasts_score'] = score_results.contrasts_score.mean().item()
        
        log_reward = log_reward.clamp(self.config.min_log_reward, self.config.max_log_reward)
        metrics['log_reward'] = log_reward.mean().item()
        if return_score_result:
            return log_reward, metrics, score_results
        return log_reward, metrics
    
    def get_e_step_loss(self, batch, sleep=False):
        """
        x: (batch_size, ...)
        """
        x = batch['x']
        batch_size = len(x)

        if sleep:
            w = self.sample_w_from_prior(batch_size, dtype=float)
            x = self.m_model.sample(w)
            w, logpf, logpb, logpt = self.sample_w(x, target=w, max_steps=self.config.max_w_length, allow_terminate=self.config.discretizer_objective == 'tb')
        else:
            w, logpf, logpb, logpt = self.sample_w(x, p_explore=self.config.p_explore_discretizer, 
                                                    max_steps=self.config.max_w_length,
                                                    allow_terminate=self.config.discretizer_objective == 'tb')

        with torch.no_grad():
            score, metrics, score_results = self.get_log_reward(x, w, return_score_result=True)
        if self.config.discretizer_objective == 'db':
            loss, disc_metrics = self.x_discretizer_model.get_db_loss(logpf, logpb, logpt, score)
        else:
            loss, disc_metrics = self.x_discretizer_model.get_tb_loss(x, logpf, logpb, logpt, score)
        metrics.update(disc_metrics)

        if self.config.decodability_weight > 0:
            logpw_z_loss, logpw_z_metrics = self.get_logpw_z_loss(score_results.z_hat.view(-1, self.config.dim_z))
            loss += logpw_z_loss
            metrics.update(logpw_z_metrics)

        metrics['loss'] = loss.item()
        mode = 'sleep' if sleep else 'wake'
        metrics = {f'{mode}/{k}': v for k, v in metrics.items()}
        return loss, metrics
    
    def get_m_step_loss(self, batch, p_explore=None, argmax=None):
        if p_explore is None:
            p_explore = self.config.m_step_p_explore
        if argmax is None:
            argmax = self.config.m_step_argmax
        x = batch['x']
        with torch.no_grad():
            if self.config.m_step_max_w_length:
                w, logpf, logpb, logpt = self.sample_w(x,
                                                        min_steps=self.config.max_w_length, 
                                                        p_explore=p_explore,
                                                        temperature=self.config.m_step_temperature,
                                                        # allow_terminate=False,
                                                        allow_terminate=self.config.m_step_substrings,
                                                        argmax=argmax)
            else:
                w, logpf, logpb, logpt = self.sample_w(x, p_explore=p_explore, temperature=self.config.m_step_temperature,
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
        loss, metrics = self.m_model.get_loss(x, w)
        metrics = {f"m_step/{k.replace('/', '_')}": v for k, v in metrics.items()}
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
        # elif self.config.e_step_unique_w_margin is not None and self.global_step > self.config.e_step_unique_w_start:
            indices = np.random.choice(self.data_module.train_indices, self.data_module.batch_size, replace=False)
            batch = self.data_module.create_batch(indices, device=self.device)
            x = batch['x']
            w = self.sample_w_from_x(x)
            num_unique_w = len(w.unique(dim=0))
            num_unique_tokens = (w > 0).any(dim=0).sum()
            if num_unique_w < self.config.e_step_min_unique_w or num_unique_tokens < self.config.e_step_min_unique_tokens:
                return False
            # self.max_last_num_unique_w = max(self.max_last_num_unique_w, num_unique_w)
            # self.num_unique_w = .95 * self.num_unique_w + .05 * num_unique_w
            # n = min(self.num_unique_w, num_unique_w)
            # if n < self.max_last_num_unique_w - self.config.e_step_unique_w_margin:
            #     return False
            # num_unique_w = self.sample_w_from_x(x).unique(dim=0).shape[0]
            # self.max_last_num_unique_w = max(self.max_last_num_unique_w, num_unique_w)
            # if num_unique_w < self.max_last_num_unique_w - self.config.e_step_unique_w_margin:
            #     return False

        self.num_mode_updates = 0
        self.e_step = False
        self.e_step_loss_threshold = min(self.e_step_loss_threshold, avg_loss.item())
        self.e_step_losses = np.zeros(self.config.e_step_loss_window)
        return True
    
    def check_and_exit_m_step(self, loss):
        """
        Returns whether or not the model should exit M-step.
        """
        self.num_mode_updates += 1
        exit_m_step = True
        if self.config.start_m_steps > 0: 
            if self.global_step < self.config.start_e_steps + self.config.start_m_steps:
                exit_m_step = False
                # print(f"Continuing at {self.global_step} (step {self.num_mode_updates})")
            else:
                # print(f"Stopping at {self.global_step} (step {self.num_mode_updates})")
                self.config.start_m_steps = 0
        elif (self.config.max_e_steps is not None and self.config.max_e_steps <= 0) or self.num_mode_updates < self.config.num_m_steps:
            # print("lalala")
            exit_m_step = False
        if exit_m_step:
            # print("Exiting M-step")
            self.num_mode_updates = 0
            self.e_step = True
            self.m_model.freeze_encoder()
        return exit_m_step

    def get_em_loss(self, batch):
        """
        Main training code should go here.
        training_step() will call this function.
        """
        if self.e_step: # E-step
            optimizer = self.get_optimizer('e_step')
            p = np.random.rand()
            sleep = p < self.config.p_sleep_phase
            loss, metrics = self.get_e_step_loss(batch, sleep=sleep)
            if not sleep:
                self.check_and_exit_e_step(loss)
            self.metrics['training/num_e_steps'] += 1
            metrics['training/e_step_loss_threshold'] = self.e_step_loss_threshold
        else: # M-step
            optimizer = self.get_optimizer('m_step')
            loss, metrics = self.get_m_step_loss(batch)
            self.check_and_exit_m_step(loss)
            self.num_m_steps += 1
            metrics['training/num_m_steps'] = self.num_m_steps
        return loss, optimizer, metrics
    
    def training_step(self, batch, batch_idx=None):
        loss, optimizer, metrics = self.get_em_loss(batch)

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

    def get_more_performance_metrics(self, batch, w, score_results):
        return {}
    
    @torch.no_grad()
    def create_plots(self):
        pass
            
    @torch.no_grad()
    def get_performance_metrics(self, n=256):
        batch = self.data_module.create_batch(np.random.choice(self.data_module.train_indices, n, replace=False), device=self.device)
        x = batch['x']
        w, logpf, logpb, logpt = self.sample_w(x)
        score_results = self.m_model(x, w)
        num_unique_tokens = (w > 0).any(dim=0).sum().item()
        num_unique_w = len(w.unique(dim=0))
        # contrast_accuracy = self.m_model.discriminator.get_accuracy(score_results.z_hat, score_results.z0)
        metrics = {
            'training/score': score_results.score.mean().item(),
            'training/num_unique_tokens': num_unique_tokens,
            'training/num_unique_sentences': num_unique_w,
            'training/w_length': w.sum(-1).mean().item(),
            # 'training/contrast_accuracy': contrast_accuracy,
        }
        metrics.update({f'training/{k}': v for k, v in self.get_more_performance_metrics(batch, w, score_results).items()})
        return metrics

    @torch.no_grad()
    def sanity_test(self, n=100, plot: bool = False):
        indices = np.random.choice(np.arange(len(self.data_module)), n)
        batch = self.data_module.create_batch(indices, device=self.device)
        x = batch['x']
        print("E-step")
        loss, metrics = self.get_e_step_loss(batch)
        print(metrics)
        print("M-step")
        loss, metrics = self.get_m_step_loss(batch)
        print(metrics)
        print("Performance metrics")
        print(self.get_performance_metrics())
        if plot:
            print("Creating plots")
            self.create_plots()
    
    def train_vae(self, num_updates, batch_size, save_name):
        optimizer = self.optimizers_dict['m_step']

        pbar = trange(num_updates)
        for i in pbar:
            indices = np.random.choice(self.data_module.train_indices, batch_size)
            batch = self.data_module.create_batch(indices, device=self.device)
            x = batch['x']
            loss, metrics = self.m_model.get_vae_loss(x)
            desc = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            pbar.set_description(desc)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        pbar.close()
        save_path = self.save(save_name=save_name, save_config=False)
        print(f"Saved VAE model to {save_path}")
    
    def save(self, save_path=None, save_dir=None, save_name=None, save_config=True):
        prefix = "" if self.config.save_prefix is None else self.config.save_prefix + '_'
        if save_path is None:
            if save_dir is None:
                save_dir = self.config.save_dir
            if save_name is None:
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f'{prefix}{self.global_step}_{now}.pt'
            save_path = f'{save_dir}/{save_name}'
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), save_path)
        if save_config:
            with open(save_dir / 'config.yaml', 'w') as f:
                f.write(yaml.dump(asdict(self.config)))
        return save_path
    
    def log_metrics(self, metrics: dict):
        self.metrics.update(metrics)
        self.log_dict(self.metrics, prog_bar=True)

    def log_image(self, key, image):
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(key=key, images=[image], caption=[f'{key} ({self.global_step})'])

    # def log_figure(self, key, p):
    #     if isinstance(self.logger, WandbLogger):
    #         img_buf = io.BytesIO()
    #         p.save(img_buf, format='png', verbose = False)
    #         im = Image.open(img_buf).copy()
    #         plt.close()
    #         img_buf.close()
    #         self.logger.log_image(key=key, images=[im], caption=[f'{key} ({self.global_step})'], commit=True)

    def log_gif(self, key, images):
        if isinstance(self.logger, WandbLogger):
            imarrays = np.array([np.transpose(np.array(im), (2, 0, 1)) for im in images])
            wandb.log({key: wandb.Video(imarrays, fps=1, format='gif', caption=f"{key} ({self.global_step})")}, commit=True)


