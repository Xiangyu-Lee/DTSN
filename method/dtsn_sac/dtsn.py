import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import sys
from functools import reduce
import operator

sys.path.append('../../')
import method.dtsn_sac.core_cuda as core

"""
Parameters for DTSN
"""

ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
Pos_threshold = 0.2
Neg_threshold = -0.2
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5

class PseudoEncoderSpikeRegular(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Regular Spike for encoder """
    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_REGULAR_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class PopSpikeEncoderRegularSpike(nn.Module):
    """ Learnable Population Coding Spike Encoder with Regular Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply
        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Regular Spike Trains
        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt)
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_REGULAR_VTH
        return pop_spikes


class PopSpikeDecoder(nn.Module):
    """ Population Coding Spike Decoder """
    def __init__(self, act_dim, pop_dim, output_activation=nn.Tanh):
        """
        :param act_dim: action dimension
        :param pop_dim:  population dimension
        :param output_activation: activation function added on output
        """
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, pop_act):
        """
        :param pop_act: output population activity
        :return: raw_act
        """
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        raw_act = self.output_activation(self.decoder(pop_act).view(-1, self.act_dim))
        return raw_act
        
class PseudoSpikeRect(torch.autograd.Function):
    """ Double threshold spiking neuron characteristics """
    @staticmethod
    def forward(ctx, input): 
        ctx.save_for_backward(input)
        spike_output = torch.where(input >= Pos_threshold, torch.ones_like(input), 
                                   torch.where(input <= Neg_threshold, -torch.ones_like(input), torch.zeros_like(input)))
        #Outputs 1 if above positive threshold, -1 if below negative threshold, otherwise 0.
        return spike_output
        
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors 
        grad_input = grad_output.clone()  
        spike_pseudo_grad = ((input >= Neg_threshold) & (input <= Pos_threshold)).float()
        return grad_input * spike_pseudo_grad

        
class SpikeMLP(nn.Module):
    """ Spike MLP with double threshold spiking neurons """
    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device):
        """
        :param in_pop_dim: input population dimension
        :param out_pop_dim: output population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoSpikeRect.apply
        # Define Layers (Hidden Layers + Output Population)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0])])
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.extend([nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer])])
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim)

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        DTSN Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY + current - torch.where(
            spike == 1, Pos_threshold, torch.where(spike == -1, Neg_threshold, 0) 
        )       

        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, in_pop_spikes, batch_size):
        """
        :param in_pop_spikes: input population spikes
        :param batch_size: batch size
        :return: out_pop_act
        """
        # Define LIF Neuron states: Current, Voltage, and Spike
        hidden_states = []
        for layer in range(self.hidden_num):
            hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                  for _ in range(3)])
        out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device)
                          for _ in range(3)]
        out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)
        # Start Spike Timestep Iteration
        for step in range(self.spike_ts):
            in_pop_spike_t = in_pop_spikes[:, :, step]
            hidden_states[0][0], hidden_states[0][1], hidden_states[0][2] = self.neuron_model(
                self.hidden_layers[0], in_pop_spike_t,
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2]
            )
            if self.hidden_num > 1:
                for layer in range(1, self.hidden_num):
                    hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2] = self.neuron_model(
                        self.hidden_layers[layer], hidden_states[layer-1][2],
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2]
                    )
            out_pop_states[0], out_pop_states[1], out_pop_states[2] = self.neuron_model(
                self.out_pop_layer, hidden_states[-1][2],
                out_pop_states[0], out_pop_states[1], out_pop_states[2]
            )
            out_pop_act += out_pop_states[2]
        out_pop_act = out_pop_act / self.spike_ts
        return out_pop_act
