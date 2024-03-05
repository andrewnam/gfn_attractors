from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F
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

    def __init__(self, dim_x: int, dim_z: int, max_mean, allow_terminate: bool, flag_z0=False, t_dependent_forward=True, t_dependent_backward=True, dim_t=10):
        """
        dim_x: dimension of the input (or some encoded vector of the input)
        dim_z: dimension of the latent space
        max_mean: maximum mean of predicted deltas, i.e. the average maximum speed of the dynamics
        t_dependent_forward: whether the forward model is dependent on t.
        t_dependent_backward: whether the backward model is dependent on t.
            In general, this should be true, even if the forward model is not dependent on t.

        dim_t: dimension of the temporal encoding
        """
        if flag_z0 and not t_dependent_backward:
            raise ValueError("If flag_z0 is True, then t_dependent_backward must be True.")
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_t = dim_t
        self.max_mean = max_mean
        self.allow_terminate = allow_terminate
        self.flag_z0 = flag_z0
        self.t_dependent_forward = t_dependent_forward
        self.t_dependent_backward = t_dependent_backward

        self.temporal_encoding = PositionalEncoding(dim_t, concat=True)
        if flag_z0:
            self.z0_flag_vector = nn.Parameter(torch.randn(dim_z))
        else:
            self.z0_flag_vector = nn.Parameter(torch.zeros(dim_z), requires_grad=False)

    def encode_z_traj(self, z_traj, forward: bool):
        num_steps = z_traj.shape[1]
        if forward:
            z0_flag = self.z0_flag_vector.view(1, 1, -1)
            z0_flag = F.pad(z0_flag, (0, 0, 0, num_steps-1), value=0)
            h0 = z_traj + z0_flag
            h = self.temporal_encoding(h0) if self.t_dependent_forward else h0
            return h
        else:
            return self.temporal_encoding(z_traj) if self.t_dependent_backward else z_traj

    def log_prob(self, z_traj, x, zero_pb_t1=True, return_params=False):
        """
        z_traj: (batch_size, num_steps, dim_z)
        x: (batch_size, dim_x)
        returns:
            logpf, logpb: (batch_size, num_steps-1)
            if allow_terminate, also returns:
                logpt: (batch_size, num_steps)
            if return_params, also returns:
                mu_f, sigma_f, mu_b, sigma_b: (batch_size, num_steps-1, dim_z)
        """
        batch_size, num_steps, dim_z = z_traj.shape
        x = einops.repeat(x, 'b x -> (b t) x', t=num_steps-1)
        
        # z0_flag = self.z0_flag_vector.view(1, 1, -1)
        # z0_flag = F.pad(z0_flag, (0, 0, 0, num_steps-1), value=0)
        # h0 = z_traj + z0_flag
        # h = self.temporal_encoding(h0) if self.t_dependent_forward else h0

        h = self.encode_z_traj(z_traj, forward=True)
        mu_f, sigma_f = self.transition_forward_delta(z_traj[:,:-1].flatten(0, 1), h[:,:-1].flatten(0, 1), x)
        mu_f = mu_f.view(batch_size, num_steps-1, dim_z)
        sigma_f = sigma_f.view(batch_size, num_steps-1, dim_z)
        dist_f = Normal(mu_f, sigma_f)
        
        # h = self.temporal_encoding(h0) if self.t_dependent_backward else h0
        h = self.encode_z_traj(z_traj, forward=False)
        mu_b, sigma_b = self.transition_backward_delta(z_traj[:,1:].flatten(0, 1), h[:,1:].flatten(0, 1), x)
        mu_b = mu_b.view(batch_size, num_steps-1, dim_z)
        sigma_b = sigma_b.view(batch_size, num_steps-1, dim_z)
        dist_b = Normal(mu_b, sigma_b)

        deltas = z_traj[..., 1:, :] - z_traj[..., :-1, :]
        logpf = dist_f.log_prob(deltas).sum(dim=-1)
        logpb = dist_b.log_prob(-deltas).sum(dim=-1)
        if zero_pb_t1:
            logpb[:,0] = 0

        retval = (logpf, logpb)

        if self.allow_terminate:
            h = self.encode_z_traj(z_traj, forward=True)
            logit = self.get_terminate_logit(z_traj, h[:,:-1].flatten(0, 1), x)
            logpt = F.logsigmoid(logit).view(batch_size, num_steps-1)
            logp_continue = F.logsigmoid(-logit).view(batch_size, num_steps-1)
            logpf = logpf + logp_continue
            logpt = torch.cat([logpt, torch.zeros(batch_size, 1, device=logpt.device)], dim=-1)
            retval = (*retval, logpt)
        if return_params:
            retval = (*retval, mu_f, sigma_f, mu_b, sigma_b)
        return retval
        
    @torch.no_grad()
    def sample_trajectory(self, z, x, num_steps: int, forward=True, deterministic=False, p_explore=0., explore_mean=False, explore_sd=1.):
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

        was_training = self.training
        self.eval()

        z_t = z
        z_traj = [z_t]
        for t in range(num_steps):
            if forward:
                h = z_t
                if t == 0:
                    h = h + self.z0_flag_vector.unsqueeze(0)
                h = self.temporal_encoding(h, t=t) if self.t_dependent_forward else z_t
                mu, sigma = self.transition_forward_delta(z_t, h, x)
            else: # backward
                h = self.temporal_encoding(z_t, t=num_steps - t) if self.t_dependent_backward else z_t
                mu, sigma = self.transition_backward_delta(z_t, h, x)

            if p_explore > 0:
                explore = torch.rand(z_t.shape[:-1], device=z_t.device) < p_explore
                if explore_mean:
                    rand_mu = 2 * self.max_mean * (torch.rand(z_t.shape, device=z_t.device) - .5)
                    mu = torch.where(explore[..., None], rand_mu, mu)
                if explore_sd > 1.:
                    sigma = sigma + explore.unsqueeze(-1) * (explore_sd - 1)

            if deterministic:
                delta = mu
            else:
                delta = Normal(mu, sigma).sample()
            z_t = z_t + delta
            z_traj.append(z_t)

        self.train(was_training)
        z_traj = torch.stack(z_traj, dim=-2)
        if not forward:
            z_traj = z_traj.flip(dims=[-2])
        return z_traj
    
    @abstractmethod
    def get_terminate_logit(self, z: torch.Tensor, z_t: torch.Tensor, x: torch.Tensor ):
        """
        z: tensor with shape (batch_size, dim_z)
        z_t: 
            if self.t_dependent_forward: tensor with shape (batch_size, dim_z + dim_t)
            else: tensor with shape (batch_size, dim_z)
        x: tensor with shape (batch_size, dim_x_encoded)
        returns: tensor with shape (batch_size,)
        """
        raise NotImplementedError
    
    @abstractmethod
    def transition_forward_delta(self, z: torch.Tensor, z_t: torch.Tensor, x: torch.Tensor):
        """
        z: tensor with shape (batch_size, dim_z)
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
    def transition_backward_delta(self, z: torch.Tensor, z_t: torch.Tensor, x: torch.Tensor):
        """
        z: tensor with shape (batch_size, dim_z)
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

    def __init__(self, dim_x: int, dim_z: int, max_mean: float, allow_terminate: bool,
                 flag_z0=False, 
                 dim_t=10, 
                 dim_h=128, 
                 num_layers=2, 
                 nonlinearity=nn.ReLU(),
                 min_sd: float = 0.,
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
        super().__init__(dim_x=dim_x, dim_z=dim_z, max_mean=max_mean, allow_terminate=allow_terminate, dim_t=dim_t, 
                         flag_z0=flag_z0,
                         t_dependent_forward=t_dependent_forward, t_dependent_backward=t_dependent_backward)
        self.min_sd = min_sd
        self.max_sd = max_sd
        self.fixed_sd = fixed_sd
        self.dim_h = dim_h
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.x_dependent_forward = x_dependent_forward
        self.x_dependent_backward = x_dependent_backward
        
        output_dim = 2 * dim_z if fixed_sd is None else dim_z
        self.forward_mlp = MLP(self.dim_f_in, output_dim, hidden_dim=dim_h, n_layers=num_layers, nonlinearity=nonlinearity)
        self.backward_mlp = MLP(self.dim_b_in, output_dim, hidden_dim=dim_h, n_layers=num_layers, nonlinearity=nonlinearity)

        if allow_terminate:
            self.terminate_mlp = MLP(self.dim_f_in, 1, hidden_dim=dim_h, n_layers=num_layers, nonlinearity=nonlinearity)

    @property
    def dim_f_in(self):
        return self.dim_z + (self.t_dependent_forward * self.dim_t) + (self.x_dependent_forward * self.dim_x)
    
    @property
    def dim_b_in(self):
        return self.dim_z + (self.t_dependent_backward * self.dim_t) + (self.x_dependent_backward * self.dim_x)

    def transition_forward_delta(self, z: torch.Tensor, z_t: torch.Tensor, x: torch.Tensor):
        h = torch.cat([z_t, x], dim=-1) if self.x_dependent_forward else z_t
        if self.fixed_sd is not None:
            mu = self.forward_mlp(h)
            sd = torch.full_like(mu, self.fixed_sd)
        else:
            mu, log_sd = self.forward_mlp(h).chunk(2, dim=-1)
            if self.max_sd is not None:
                sd = self.min_sd + (self.max_sd - self.min_sd) * torch.sigmoid(log_sd)
            else:
                sd = self.min_sd + log_sd.exp()
        mu = self.max_mean * mu.tanh()
        return mu, sd
    
    def transition_backward_delta(self, z: torch.Tensor, z_t: torch.Tensor, x: torch.Tensor):
        h = torch.cat([z_t, x], dim=-1) if self.x_dependent_backward else z_t
        if self.fixed_sd is not None:
            mu = self.backward_mlp(h)
            sd = torch.full_like(mu, self.fixed_sd)
        else:
            mu, log_sd = self.backward_mlp(h).chunk(2, dim=-1)
            if self.max_sd is not None:
                sd = self.min_sd + (self.max_sd - self.min_sd) * torch.sigmoid(log_sd)
            else:
                sd = self.min_sd + log_sd.exp()
        mu = self.max_mean * mu.tanh()
        return mu, sd
    
    def get_terminate_logit(self, z: torch.Tensor, z_t, x):
        """
        z_t: 
            if self.t_dependent_forward: tensor with shape (batch_size, dim_z + dim_t)
            else: tensor with shape (batch_size, dim_z)
        x: tensor with shape (batch_size, dim_x_encoded)
        returns: tensor with shape (batch_size,)
        """
        h = torch.cat([z_t, x], dim=-1) if self.x_dependent_forward else z_t
        return self.terminate_mlp(h)


class MLPLangevinMeanBoundedDynamics(MLPMeanBoundedDynamics):

    def __init__(self, dim_x: int, dim_z: int, max_mean: float, energy_func, *args, **kwargs):
        """
        energy_func: a differentiable function that takes z and x, and returns the energy at (z, x), 
            where the model wants to go towards the direction of increasing energy.
            z: tensor with shape (batch_size, dim_z)
            x: tensor with shape (batch_size, dim_x_encoded)
            returns energy with shape (batch_size)
        """
        super().__init__(dim_x=dim_x, dim_z=dim_z, max_mean=max_mean, allow_terminate=False, *args, **kwargs)
        self.energy_func = energy_func
        self.alpha_mlp = MLP(self.dim_f_in, 1, hidden_dim=self.dim_h, n_layers=self.num_layers, nonlinearity=self.nonlinearity)
        
    def transition_forward_delta(self, z: torch.Tensor, z_t: torch.Tensor, x: torch.Tensor, create_graph=False):
        mu, sd = super().transition_forward_delta(z, z_t, x)
        h = torch.cat([z_t, x], dim=-1) if self.x_dependent_forward else z_t
        alpha = self.alpha_mlp(h).sigmoid().unsqueeze(-1)
        with torch.enable_grad():
            z = z.clone()
            if not z.requires_grad:
                z.requires_grad = True
            energy = self.energy_func(z, x.detach())
            grad = torch.autograd.grad(energy.sum(), z, create_graph=create_graph)[0]
            grad = (grad / grad.norm(dim=-1, keepdim=True)).nan_to_num()
        mu = (1 - alpha) * mu + alpha * self.max_mean * grad
        return mu, sd

    def get_alpha(self, z_traj, x):
        z_traj = self.encode_z_traj(z_traj, forward=True)
        h = torch.cat([z_traj, x], dim=-1) if self.x_dependent_forward else z_traj
        alpha = self.alpha_mlp(h).sigmoid().unsqueeze(-1)
        return alpha
