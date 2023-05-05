import torch
import numpy as np
from typing import Optional


class RunningMeanStd(torch.nn.Module):
    def __init__(self, dim: int, clamp: float=0):
        super().__init__()
        self.epsilon = 1e-5
        self.clamp = clamp
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def forward(self, x, unnorm=False):
        mean = self.mean.to(torch.float32)
        var = self.var.to(torch.float32)+self.epsilon
        if unnorm:
            if self.clamp:
                x = torch.clamp(x, min=-self.clamp, max=self.clamp)
            return mean + torch.sqrt(var) * x
        x = (x - mean) * torch.rsqrt(var)
        if self.clamp:
            return torch.clamp(x, min=-self.clamp, max=self.clamp)
        return x
    
    @torch.no_grad()
    def update(self, x):
        x = x.view(-1, x.size(-1))
        var, mean = torch.var_mean(x, dim=0, unbiased=True)
        count = x.size(0)
        count_ = count + self.count
        delta = mean - self.mean
        m = self.var * self.count + var * count + delta**2 * self.count * count / count_
        self.mean.copy_(self.mean+delta*count/count_)
        self.var.copy_(m / count_)
        self.count.copy_(count_)


class DiagonalPopArt(torch.nn.Module):
    def __init__(self, dim: int, weight: torch.Tensor, bias: torch.Tensor, momentum:float=0.1):
        super().__init__()
        self.epsilon = 1e-5

        self.momentum = momentum
        self.register_buffer("m", torch.zeros((dim,), dtype=torch.float64))
        self.register_buffer("v", torch.full((dim,), self.epsilon, dtype=torch.float64))
        self.register_buffer("debias", torch.zeros(1, dtype=torch.float64))

        self.weight = weight
        self.bias = bias

    def forward(self, x, unnorm=False):
        debias = self.debias.clip(min=self.epsilon)
        mean = self.m/debias
        var = (self.v - self.m.square()).div_(debias)
        if unnorm:
            std = torch.sqrt(var)
            return (mean + std * x).to(x.dtype)
        x = ((x - mean) * torch.rsqrt(var)).to(x.dtype)
        return x

    @torch.no_grad()
    def update(self, x):
        x = x.view(-1, x.size(-1))
        running_m = torch.mean(x, dim=0)
        running_v = torch.mean(x.square(), dim=0)
        new_m = self.m.mul(1-self.momentum).add_(running_m, alpha=self.momentum)
        new_v = self.v.mul(1-self.momentum).add_(running_v, alpha=self.momentum)
        std = (self.v - self.m.square()).sqrt_()
        new_std_inv = (new_v - new_m.square()).rsqrt_()

        scale = std.mul_(new_std_inv)
        shift = (self.m - new_m).mul_(new_std_inv)

        self.bias.data.mul_(scale).add_(shift)
        self.weight.data.mul_(scale.unsqueeze_(-1))

        self.debias.data.mul_(1-self.momentum).add_(1.0*self.momentum)
        self.m.data.copy_(new_m)
        self.v.data.copy_(new_v)


class Discriminator(torch.nn.Module):
    def __init__(self, disc_dim, latent_dim=256):
        super().__init__()
        self.rnn = torch.nn.GRU(disc_dim, latent_dim, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32)
        )
        if self.rnn is not None:
            i = 0
            for n, p in self.mlp.named_parameters():
                if "bias" in n:
                    torch.nn.init.constant_(p, 0.)
                elif "weight" in n:
                    gain = 1 if i == 2 else 2**0.5 
                    torch.nn.init.orthogonal_(p, gain=gain)
                    i += 1
        self.ob_normalizer = RunningMeanStd(disc_dim)
        self.all_inst = torch.arange(0)
        
    def forward(self, s, seq_end_frame, normalize=True):
        if normalize: s = self.ob_normalizer(s)
        if self.rnn is None:
            s = s.view(s.size(0), -1)
        else:
            n_inst = s.size(0)
            if n_inst > self.all_inst.size(0):
                self.all_inst = torch.arange(n_inst, 
                    dtype=seq_end_frame.dtype, device=seq_end_frame.device)
            s, _ = self.rnn(s)
            s = s[(self.all_inst[:n_inst], torch.clip(seq_end_frame, max=s.size(1)-1))]
        return self.mlp(s)


class ACModel(torch.nn.Module):

    class Critic(torch.nn.Module):
        def __init__(self, state_dim, goal_dim, value_dim=1, latent_dim=256):
            super().__init__()
            self.rnn = torch.nn.GRU(state_dim, latent_dim, batch_first=True)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(latent_dim+goal_dim, 1024),
                torch.nn.ReLU6(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU6(),
                torch.nn.Linear(512, value_dim)
            )
            i = 0
            for n, p in self.mlp.named_parameters():
                if "bias" in n:
                    torch.nn.init.constant_(p, 0.)
                elif "weight" in n:
                    torch.nn.init.uniform_(p, -0.0001, 0.0001)
                    i += 1
            self.all_inst = torch.arange(0)

        def forward(self, s, seq_end_frame, g=None):
            if self.rnn is None:
                s = s.view(s.size(0), -1)
            else:
                n_inst = s.size(0)
                if n_inst > self.all_inst.size(0):
                    self.all_inst = torch.arange(n_inst, 
                        dtype=seq_end_frame.dtype, device=seq_end_frame.device)
                s, _ = self.rnn(s)
                s = s[(self.all_inst[:n_inst], torch.clip(seq_end_frame, max=s.size(1)-1))]
            if g is not None:
                s = torch.cat((s, g), -1)
            return self.mlp(s)


    class Actor(torch.nn.Module):
        def __init__(self, state_dim, act_dim, goal_dim, latent_dim=256, init_mu=None, init_sigma=None):
            super().__init__()
            self.rnn = torch.nn.GRU(state_dim, latent_dim, batch_first=True)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(latent_dim+goal_dim, 1024),
                torch.nn.ReLU6(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU6()
            )
            self.mu = torch.nn.Linear(512, act_dim)
            self.log_sigma = torch.nn.Linear(512, act_dim)
            with torch.no_grad():
                if init_mu is not None:
                    if torch.is_tensor(init_mu):
                        mu = torch.ones_like(self.mu.bias)*init_mu
                    else:
                        mu = np.ones(self.mu.bias.shape, dtype=np.float32)*init_mu
                        mu = torch.from_numpy(mu)
                    self.mu.bias.data.copy_(mu)
                    torch.nn.init.uniform_(self.mu.weight, -0.00001, 0.00001)
                if init_sigma is None:
                    torch.nn.init.constant_(self.log_sigma.bias, -3)
                    torch.nn.init.uniform_(self.log_sigma.weight, -0.0001, 0.0001)
                else:
                    if torch.is_tensor(init_sigma):
                        log_sigma = (torch.ones_like(self.log_sigma.bias)*init_sigma).log_()
                    else:
                        log_sigma = np.log(np.ones(self.log_sigma.bias.shape, dtype=np.float32)*init_sigma)
                        log_sigma = torch.from_numpy(log_sigma)
                    self.log_sigma.bias.data.copy_(log_sigma)
                    torch.nn.init.uniform_(self.log_sigma.weight, -0.00001, 0.00001)
                self.all_inst = torch.arange(0)

        def forward(self, s, seq_end_frame, g=None):
            if self.rnn is None:
                s = s.view(s.size(0), -1)
            else:
                n_inst = s.size(0)
                if n_inst > self.all_inst.size(0):
                    self.all_inst = torch.arange(n_inst, 
                        dtype=seq_end_frame.dtype, device=seq_end_frame.device)
                s, _ = self.rnn(s)
                s = s[(self.all_inst[:n_inst], torch.clip(seq_end_frame, max=s.size(1)-1))]
            if g is not None:
                s = torch.cat((s, g), -1)
            latent = self.mlp(s)
            mu = self.mu(latent)
            sigma = torch.exp(self.log_sigma(latent)) + 1e-8
            return torch.distributions.Normal(mu, sigma)


    def __init__(self, state_dim: int, act_dim: int, goal_dim: int=0, value_dim: int=1, 
        normalize_value: bool=True,
        init_mu:Optional[torch.Tensor or float]=None, init_sigma:Optional[torch.Tensor or float]=None
    ):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.actor = self.Actor(state_dim, act_dim, self.goal_dim, init_mu=init_mu, init_sigma=init_sigma)
        self.critic = self.Critic(state_dim, goal_dim, value_dim)
        self.ob_normalizer = RunningMeanStd(state_dim, clamp=5.0)
        if normalize_value:            
            self.value_normalizer = DiagonalPopArt(value_dim, 
                self.critic.mlp[-1].weight, self.critic.mlp[-1].bias)
        else:
            self.value_normalizer = None
            
    def observe(self, obs, norm=True):
        if self.goal_dim > 0:
            s = obs[:, :-self.goal_dim]
            g = obs[:, -self.goal_dim:]
        else:
            s = obs
            g = None
        s = s.view(*s.shape[:-1], -1, self.state_dim)
        return self.ob_normalizer(s) if norm else s, g

    def eval_(self, s, seq_end_frame, g, unnorm):
        v = self.critic(s, seq_end_frame, g)
        if unnorm and self.value_normalizer is not None:
            v = self.value_normalizer(v, unnorm=True)
        return v

    def act(self, obs, seq_end_frame, stochastic=None, unnorm=False):
        if stochastic is None:
            stochastic = self.training
        s, g = self.observe(obs)
        pi = self.actor(s, seq_end_frame, g)
        if stochastic:
            a = pi.sample()
            lp = pi.log_prob(a)
            if g is not None:
                g = g[...,:self.goal_dim]
            return a, self.eval_(s, seq_end_frame, g, unnorm), lp
        else:
            return pi.mean,

    def evaluate(self, obs, seq_end_frame, unnorm=False):
        s, g = self.observe(obs)
        if g is not None:
            g = g[...,:self.goal_dim]
        return self.eval_(s, seq_end_frame, g, unnorm)
    
    def forward(self, obs, seq_end_frame, unnorm=False):
        s, g = self.observe(obs)
        pi = self.actor(s, seq_end_frame, g)
        if g is not None:
            g = g[...,:self.goal_dim]
        return pi, self.eval_(s, seq_end_frame, g, unnorm)
