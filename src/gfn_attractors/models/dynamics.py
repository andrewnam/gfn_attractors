from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.distributions import Normal
import einops

from .helpers import PositionalEncoding, MLP


class MeanBoundedDynamics(ABC, nn.Module):
    """
    Abstract class for continuous dynamics where the transition probabilities are Gaussians with bounded means.
    This forces the model to learn small steps and prevents it from jumping to the attractor, while still maintaining full
    support over z-space.

    Can specify whether the forward and backward models are dependent on t.
    Assumes that both forward and backward policies are dependent on x, but the implementing class can ignore it.

    Note that this model has a slightly simplified interface compared to DualStochasticDynamics.
    Exploration policy is also simplified so that p_explore % of the time, it samples a mean uniformly within [-max_mean, max_mean].
    """

    def __init__(self, dim_x: int, dim_z: int, max_mean, t_dependent_forward=True, t_dependent_backward=True, dim_t=10):
        """
        dim_x: dimension of the input (or some encoded vector of the input)
        dim_z: dimension of the latent space
        max_mean: maximum mean of predicted deltas, i.e. the average maximum speed of the dynamics
        t_dependent_forward: whether the forward model is dependent on t.
        t_dependent_backward: whether the backward model is dependent on t.
            In general, this should be true, even if the forward model is not dependent on t.

        dim_t: dimension of the temporal encoding
        """
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_t = dim_t
        self.max_mean = max_mean
        self.t_dependent_forward = t_dependent_forward
        self.t_dependent_backward = t_dependent_backward

        self.temporal_encoding = PositionalEncoding(dim_t, concat=True)

    def log_prob(self, z_traj, x, return_params=False):
        """
        z_traj: (batch_size, num_steps, dim_z)
        x: (batch_size, dim_x)
        returns:
            logpf, logpb: (batch_size, num_steps-1)
            if return_params, also returns:
                mu_f, sigma_f, mu_b, sigma_b: (batch_size, num_steps-1, dim_z)
        """
        batch_size, num_steps, dim_z = z_traj.shape
        x = einops.repeat(x, 'b x -> (b t) x', t=num_steps-1)
        
        h = self.temporal_encoding(z_traj) if self.t_dependent_forward else z_traj
        mu_f, sigma_f = self.transition_forward_delta(h[:,:-1].flatten(0, 1), x)
        mu_f = mu_f.view(batch_size, num_steps-1, dim_z)
        sigma_f = sigma_f.view(batch_size, num_steps-1, dim_z)
        dist_f = Normal(mu_f, sigma_f)
        
        h = self.temporal_encoding(z_traj) if self.t_dependent_backward else z_traj
        mu_b, sigma_b = self.transition_backward_delta(h[:,1:].flatten(0, 1), x)
        mu_b = mu_b.view(batch_size, num_steps-1, dim_z)
        sigma_b = sigma_b.view(batch_size, num_steps-1, dim_z)
        dist_b = Normal(mu_b, sigma_b)

        deltas = z_traj[..., 1:, :] - z_traj[..., :-1, :]
        logpf = dist_f.log_prob(deltas).sum(dim=-1)
        logpb = dist_b.log_prob(-deltas).sum(dim=-1)

        if return_params:
            return logpf, logpb, mu_f, sigma_f, mu_b, sigma_b
        return logpf, logpb
        
    @torch.no_grad()
    def sample_trajectory(self, z, x, num_steps: int, forward=True, p_explore=0., explore_mean=False, explore_sd=1.):
        """
        Samples the trajectory of z for num_steps.
        The returned tensor is sorted in increasing t, even if sampling backwards.

        z: (batch_size, dim_z)
            If x is None, this is z0 and a forward trajectory is generated from it.
            If x is defined, this is z_T and a backward trajectory is generated from it.
        x: (batch_size, dim_x_encoded)
        explore_mean: if True, explores by sampling mean between [-max_mean, max_mean]
        explore_sd: samples by multiplying the sd by this factor
        returns: (batch_size, num_steps, dim_z)
        """
        assert not (p_explore > 0 and not explore_mean and explore_sd == 1.) # if exploring, must explore either mean or sd
        assert explore_sd >= 1.

        was_training = self.training
        self.eval()

        z_t = z
        z_traj = [z_t]
        for t in range(num_steps):
            if forward:
                h = self.temporal_encoding(z_t, t=t) if self.t_dependent_forward else z_t
                mu, sigma = self.transition_forward_delta(h, x)
            else: # backward
                h = self.temporal_encoding(z_t, t=num_steps - t) if self.t_dependent_backward else z_t
                mu, sigma = self.transition_backward_delta(h, x)

            if p_explore > 0:
                explore = torch.rand(z_t.shape[:-1], device=z_t.device) < p_explore
                if explore_mean:
                    rand_mu = 2 * self.max_mean * (torch.rand(z_t.shape, device=z_t.device) - .5)
                    mu = torch.where(explore[..., None], rand_mu, mu)
                if explore_sd > 1.:
                    sigma = sigma + explore.unsqueeze(-1) * (explore_sd - 1)

            delta = Normal(mu, sigma).sample()
            z_t = z_t + delta
            z_traj.append(z_t)

        self.train(was_training)
        z_traj = torch.stack(z_traj, dim=-2)
        if not forward:
            z_traj = z_traj.flip(dims=[-2])
        return z_traj
    
    @abstractmethod
    def transition_forward_delta(self, z_t: torch.Tensor, x: torch.Tensor):
        """
        z_t: 
            if self.t_dependent_forward: tensor with shape (batch_size, dim_z + dim_t)
            else: tensor with shape (batch_size, dim_z)
        x: tensor with shape (batch_size, dim_x_encoded)
        returns:
            mu, sigma: tensors with shape (batch_size, dim_z)

        Note: assumes that mu is bounded
        """
        raise NotImplementedError

    @abstractmethod
    def transition_backward_delta(self, z_t: torch.Tensor, x: torch.Tensor):
        """
        z_t:
            if self.t_dependent_backward: tensor with shape (batch_size, dim_z + dim_t)
            else: tensor with shape (batch_size, dim_z)
        x: tensor with shape (batch_size, dim_x_encoded)
        returns:
            mu, sigma: tensors with shape (batch_size, dim_z)
        
        Note: assumes that mu is bounded
        """
        raise NotImplementedError


class MLPMeanBoundedDynamics(MeanBoundedDynamics):

    def __init__(self, dim_x: int, dim_z: int, max_mean: float,
                 dim_t=10, 
                 dim_h=128, 
                 num_layers=2, 
                 nonlinearity=nn.ReLU(),
                 max_sd: float|None = None, 
                 fixed_sd: float|None = None, 
                 t_dependent_forward=True,
                 t_dependent_backward=True,
                 x_dependent_forward=True,
                 x_dependent_backward=True):
        """
        max_sd: if not None, then sd = max_sd * sigmoid(log_sd)
        fixed_sd: if not None, then sd = fixed_sd
            Note: max_sd and fixed_sd should not both be None
            If both are None, sd is unbounded.
        x_dependent_forward: if True, then the mean and sd of the forward transition are functions of x
            Note that if the forward policy does not explicitly depend on x, then the forward policy learns the geometric expectation w.r.t x.
        x_dependent_backward: if True, then the mean and sd of the backward transition are functions of x
            In general, the backward policy should depend on x.
        """
        assert not (max_sd is not None and fixed_sd is not None)
        super().__init__(dim_x=dim_x, dim_z=dim_z, max_mean=max_mean, dim_t=dim_t, 
                         t_dependent_forward=t_dependent_forward, t_dependent_backward=t_dependent_backward)
        self.max_sd = max_sd
        self.fixed_sd = fixed_sd
        self.dim_h = dim_h
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.x_dependent_forward = x_dependent_forward
        self.x_dependent_backward = x_dependent_backward
        
        output_dim = 2 * dim_z if fixed_sd is None else dim_z
        dim_f_in = dim_z + (t_dependent_forward * dim_t) + (x_dependent_forward * dim_x)
        dim_b_in = dim_z + (t_dependent_backward * dim_t) + (x_dependent_backward * dim_x)
        self.forward_mlp = MLP(dim_f_in, output_dim, hidden_dim=dim_h, n_layers=num_layers, nonlinearity=nonlinearity)
        self.backward_mlp = MLP(dim_b_in, output_dim, hidden_dim=dim_h, n_layers=num_layers, nonlinearity=nonlinearity)

    def transition_forward_delta(self, z_t: torch.Tensor, x: torch.Tensor):
        h = torch.cat([z_t, x], dim=-1) if self.x_dependent_forward else z_t
        if self.fixed_sd is not None:
            mu = self.forward_mlp(h)
            sd = torch.full_like(mu, self.fixed_sd)
        else:
            mu, log_sd = self.forward_mlp(h).chunk(2, dim=-1)
            if self.max_sd is not None:
                sd = self.max_sd * torch.sigmoid(log_sd)
            else:
                sd = log_sd.exp()
        mu = self.max_mean * mu.tanh()
        return mu, sd
    
    def transition_backward_delta(self, z_t: torch.Tensor, x: torch.Tensor):
        h = torch.cat([z_t, x], dim=-1) if self.x_dependent_backward else z_t
        if self.fixed_sd is not None:
            mu = self.backward_mlp(h)
            sd = torch.full_like(mu, self.fixed_sd)
        else:
            mu, log_sd = self.backward_mlp(h).chunk(2, dim=-1)
            if self.max_sd is not None:
                sd = self.max_sd * torch.sigmoid(log_sd)
            else:
                sd = log_sd.exp()
        mu = self.max_mean * mu.tanh()
        return mu, sd
    