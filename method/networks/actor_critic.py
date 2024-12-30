from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym.spaces

from .distributions import (
    FixedCategorical,
    FixedNormal,
    Identity,
    MixedDistribution,
)
from .utils import MLP, flatten_ac
from .encoder import Encoder
from ..utils.pytorch import to_tensor
from ..utils.logger import logger

from method.dtsn_sac.dtsn import PopSpikeEncoderRegularSpike
from method.dtsn_sac.dtsn import SpikeMLP
from method.dtsn_sac.dtsn import PopSpikeDecoder
import method.dtsn_sac.core_cuda as core
from torch.distributions.normal import Normal
import math
ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5   


LOG_STD_MAX = 2
LOG_STD_MIN = -10

class Actor(nn.Module):
    def __init__(self, config, ob_space, ac_space, tanh_policy, encoder=None):
        super().__init__()
        self._config = config
        self._ac_space = ac_space
        self._activation_fn = getattr(F, config.policy_activation)
        self._tanh = tanh_policy
        self._gaussian = config.gaussian_policy
      
        if encoder:
            self.encoder1 = encoder
        else:
            self.encoder1 = Encoder(config, ob_space)

        self.fc = MLP(
            config, self.encoder1.output_dim, config.policy_mlp_dim[-1], config.policy_mlp_dim[:-1]
        )
        obs_dim = np.concatenate((ob_space['robot_ob'].shape, ob_space['object_ob'].shape)).sum()
        act_dim=ac_space.spaces['default'].shape[0]
        act_limit=ac_space.spaces['default'].high_repr
        self.act_limit = float(act_limit)
        self.encoder = PopSpikeEncoderRegularSpike(obs_dim, config.en_pop_dim, config.spike_ts, config.mean_range, config.std, config.device)
        self.snn = SpikeMLP(obs_dim*config.en_pop_dim, act_dim*config.de_pop_dim, config.hidden_sizes, config.spike_ts, config.device)
        self.decoder = PopSpikeDecoder(act_dim, config.de_pop_dim, output_activation=nn.Identity)
        # Use a complete separate deep MLP to predict log std
        self.log_std_network = core.mlp([obs_dim] + list(config.hidden_sizes) + [act_dim], nn.ReLU)

    @property
    def info(self):
        return {}

    def act(self, ob, deterministic=False,return_log_prob=False, detach_conv=False):
        """
        :param obs: observation
        :param batch_size: batch size
        :param deterministic: If true use deterministic action
        :param with_logprob: if true return log prob
        :return: action scale with action limit
        """
        ob = self.encoder1(ob, detach_conv=detach_conv)
        in_pop_spikes = self.encoder(ob, batch_size)
        out_pop_activity = self.snn(in_pop_spikes, batch_size)
        activations = self.decoder(out_pop_activity)
        log_std = self.log_std_network(ob)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        # Pre-squash distribution and sample
        pi_distribution = Normal(activations, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            actions = activations
        else:
            actions = pi_distribution.rsample()
        if return_log_prob:
            log_probs = pi_distribution.log_prob(actions).sum(axis=-1)
            log_probs -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(axis=1)
        else:
            log_probs = None
        actions = torch.tanh(actions)

        # Calculate entropy for each action
        entropy = 0.5 + 0.5 * math.log(2 * math.pi * math.e) + log_std
        entropy = entropy.sum()
        actions = self.act_limit * actions

        # OrderedDict
        actions = OrderedDict({'default': actions})
        activations = OrderedDict({'default': activations})
        return actions, activations, log_probs, entropy


class Critic(nn.Module):
    def __init__(self, config, ob_space, ac_space=None, encoder=None):
        super().__init__()
        self._config = config

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = Encoder(config, ob_space)

        input_dim = self.encoder.output_dim
        if ac_space is not None:
            input_dim += gym.spaces.flatdim(ac_space)

        self.fcs = nn.ModuleList()

        for _ in range(config.critic_ensemble):
            self.fcs.append(MLP(config, input_dim, 1, config.critic_mlp_dim))

    def forward(self, ob, ac=None, detach_conv=False):
        out = self.encoder(ob, detach_conv=detach_conv)

        if ac is not None:
            out = torch.cat([out, flatten_ac(ac)], dim=-1)
        assert len(out.shape) == 2

        out = [fc(out) for fc in self.fcs]
        if len(out) == 1:
            return out[0]
        return out

# if __name__ == '__main__':
#     model = Actor(args).to(device)

#     flop,params = get_model_complexity_info(model,(1,64),as_strings=True,print_per_layer_stat=True)
#     print(f"FLOPs: {flop}, Parameters: {params}")
