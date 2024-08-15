from metrics.visualization_metrics import visualization
from metrics.predictive_metrics import predictive_score_metrics,predictive_score_metrics2
from metrics.discriminative_metrics import discriminative_score_metrics
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import (
    create_regularization_fns,
    get_regularization,
    append_regularization_to_log,
    build_model_tabular,
    build_model_tabular_nonlinear,
    build_model_tabular_original
)
from ctfp_tools import run_latent_ctfp_model5_2 as run_model, parse_arguments
from ctfp_tools import build_augmented_model_tabular
import lib.utils as utils
import controldiffeq
from torchdiffeq import odeint
import gru_ode_bayes.data_utils as data_utils
from gru_ode_bayes.models import FullGRUODECell_Autonomous
import os
import sys
import pathlib
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
from time_dataset import *
from torch.optim import optimizer
from torch.nn import functional as F
from torch import nn, optim
from itertools import chain

import numpy as np
import torch
import tensorflow as tf
random_seed = 7777
torch.manual_seed(random_seed)
np.random.seed(random_seed)

class Net(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        rnn=nn.GRU,
        activation_fn=torch.sigmoid,
    ):
        super().__init__()
        self.rnn = rnn(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class Net2(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        rnn=nn.GRU,
        activation_fn=torch.sigmoid,
    ):
        super().__init__()
        self.rnn = rnn(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        _, x = self.rnn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class ContinuousRNNConverter(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, model):
        super(ContinuousRNNConverter, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model

        out_base = torch.zeros(self.input_channels +
                               self.hidden_channels, self.input_channels)
        for i in range(self.input_channels):
            out_base[i, i] = 1
        self.register_buffer('out_base', out_base)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}".format(self.input_channels, self.hidden_channels)

    def forward(self, z):
        # z is a tensor of shape (..., input_channels + hidden_channels)
        x = z[..., :self.input_channels]
        h = z[..., self.input_channels:]
        # In theory the hidden state must lie in this region. And most of the time it does anyway! Very occasionally
        # it escapes this and breaks everything, though. (Even when using adaptive solvers or small step sizes.) Which
        # is kind of surprising given how similar the GRU-ODE is to a standard negative exponential problem, we'd
        # expect to get absolute stability without too much difficulty. Maybe there's a bug in the implementation
        # somewhere, but not that I've been able to find... (and h does only escape this region quite rarely.)
        h = h.clamp(-1, 1)
        # model_out is a tensor of shape (..., hidden_channels)
        model_out = self.model(x, h)
        batch_dims = model_out.shape[:-1]
        out = self.out_base.repeat(*batch_dims, 1, 1).clone()
        out[..., self.input_channels:, 0] = model_out
        return out


class FinalTanh(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(
            hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(
            hidden_hidden_channels, input_channels * hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels,
                         self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):

        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(
            *z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z


class NeuralCDE(torch.nn.Module):
    """A Neural CDE model. Provides a wrapper around the lower-level cdeint function, to get a flexible Neural CDE
    model.

    Specifically, considering the CDE
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where X is determined by the data, and given some terminal time t_N, then this model first computes z_{t_N}, then
    performs a linear function on it, and then outputs the result.

    It's known that linear functions on CDEs are universal approximators, so this is a very general type of model.
    """

    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        if isinstance(func, ContinuousRNNConverter):  # ugly hack
            hidden_channels = hidden_channels + input_channels

        super(NeuralCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.func = func
        self.initial = initial
        if initial and not isinstance(func, ContinuousRNNConverter):  # very ugly hack
            self.initial_network = torch.nn.Linear(
                input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

        self.activation_fn = torch.sigmoid

    def extra_repr(self):
        return "input_channels={}, hidden_channels={}, output_channels={}, initial={}" \
               "".format(self.input_channels, self.hidden_channels,
                         self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index, z0=None, stream=True, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t f``or which there was data.
        """
        # Extract the sizes of the batch dimensions from the coefficients
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, "coeff.shape[:-2] must be the same as final_index.shape. " \
                                                    "coeff.shape[:-2]={}, final_index.shape={}" \
                                                    "".format(
                                                        batch_dims, final_index.shape)

        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)

        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            if isinstance(self.func, ContinuousRNNConverter):  # still an ugly hack
                z0 = torch.zeros(*batch_dims, self.hidden_channels,
                                 dtype=coeff.dtype, device=coeff.device)
            else:
                z0 = self.initial_network(cubic_spline.evaluate(times[0]))
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
            # continuing adventures in ugly hacks
            if isinstance(self.func, ContinuousRNNConverter):
                z0_extra = torch.zeros(
                    *batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)
        # Figure out what times we need to solve for

        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(
                sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat(
                [times[0].unsqueeze(0), times[sorted_final_index], times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()
        # Actually solve the CDE
        z_t = controldiffeq.cdeint(dX_dt=cubic_spline.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=t,
                                   **kwargs)

        # Organise the output

        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(
                -1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        #final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0).type(torch.int64)
        #z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)
        # Linear map and return
        pred_y = self.linear(z_t)
        pred_y = self.activation_fn(pred_y)
        return pred_y


class RecoveryODENetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, last_activation='identity', solver='euler'):
        ''' 24 24 6 24 48
        Arguments:
            input_size: input shape
            hidden_size: shape of hidden state of GRUODE and GRU
            output_size: output shape
            gru_input_size: input size of GRU (raw input will pass through x_model which change shape input_size to gru_input_size)
            x_hidden: shape going through x_model
            delta_t: integration time step for fixed integrator
            solver: ['euler','midpoint','dopri5']
        '''
        super(RecoveryODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True

        self.x_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.x_hidden, bias=self.bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_out),
            torch.nn.Linear(self.x_hidden, self.gru_input_size, bias=self.bias)
        )
        self.gru_layer = FullGRUODECell_Autonomous(
            self.hidden_size, bias=self.bias)
        self.gru_obs = torch.nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.hidden_size)

        if last_activation == 'identity':
            self.last_layer = None
        elif last_activation == 'softplus':
            self.last_layer = torch.nn.Softplus()
        elif last_activation == 'tanh':
            self.last_layer = torch.nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last_layer = torch.nn.Sigmoid()
        self.rec_linear = torch.nn.Linear(
            self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert self.impute == False
            solution, eval_times, eval_vals = odeint(
                func, h, torch.tensor([0, delta_t]))
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        HH = self.x_model(H)
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to('cuda')
        current_time = times[0, 0] - 1
        final_time = times[0, -1]
        for idx, obs_time in enumerate(times[0]):
            while current_time < (obs_time-0.001*self.delta_t):
                if self.solver == 'dopri5':
                    h, current_time = self.ode_step(
                        h, self.gru_layer, obs_time-current_time, current_time)
                else:
                    h, current_time = self.ode_step(
                        h, self.gru_layer, self.delta_t, current_time)
            current_out, tmp = self.gru_obs(torch.reshape(
                HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])), h[None, :, :])
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + \
                current_out.reshape(HH.shape[0], HH.shape[-1])
        X_tilde = self.rec_linear(out)
        if self.last_layer != None:
            X_tilde = self.last_layer(X_tilde)
        return X_tilde


class First_ODENetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, solver='euler'):
        super(First_ODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True

        self.x_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.x_hidden, bias=self.bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_out),
            torch.nn.Linear(self.x_hidden, self.gru_input_size, bias=self.bias)
        )
        self.gru_layer = FullGRUODECell_Autonomous(
            self.hidden_size, bias=self.bias)
        self.gru_obs = torch.nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.hidden_size)
        # self.rec_linear = torch.nn.Linear(self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert self.impute == False
            solution, eval_times, eval_vals = odeint(
                func, h, torch.tensor([0, delta_t]))
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        HH = self.x_model(H)
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to('cuda')
        current_time = times[0, 0] - 1
        final_time = times[0, -1]
        for idx, obs_time in enumerate(times[0]):
            while current_time < (obs_time-0.001*self.delta_t):
                if self.solver == 'dopri5':
                    h, current_time = self.ode_step(
                        h, self.gru_layer, obs_time-current_time, current_time)
                else:
                    h, current_time = self.ode_step(
                        h, self.gru_layer, self.delta_t, current_time)
            current_out, tmp = self.gru_obs(torch.reshape(
                HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])), h[None, :, :])
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + \
                current_out.reshape(HH.shape[0], HH.shape[-1])
        # X_tilde = self.rec_linear(out)
        # return X_tilde
        return out


class Mid_ODENetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, solver='euler'):
        super(Mid_ODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True

        # self.x_model = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_size, self.x_hidden, bias = self.bias),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p = self.drop_out),
        #     torch.nn.Linear(self.x_hidden, self.gru_input_size, bias = self.bias)
        # )
        self.gru_layer = FullGRUODECell_Autonomous(
            self.hidden_size, bias=self.bias)
        self.gru_obs = torch.nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.hidden_size)
        # self.rec_linear = torch.nn.Linear(self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert self.impute == False
            solution, eval_times, eval_vals = odeint(
                func, h, torch.tensor([0, delta_t]))
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        # HH = self.x_model(H)
        HH = H
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to('cuda')
        current_time = times[0, 0] - 1
        final_time = times[0, -1]
        for idx, obs_time in enumerate(times[0]):
            while current_time < (obs_time-0.001*self.delta_t):
                if self.solver == 'dopri5':
                    h, current_time = self.ode_step(
                        h, self.gru_layer, obs_time-current_time, current_time)
                else:
                    h, current_time = self.ode_step(
                        h, self.gru_layer, self.delta_t, current_time)
            current_out, tmp = self.gru_obs(torch.reshape(
                HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])), h[None, :, :])
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + \
                current_out.reshape(HH.shape[0], HH.shape[-1])
        # X_tilde = self.rec_linear(out)
        # return X_tilde
        return out


class Last_ODENetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, last_activation='identity', solver='euler'):
        super(Last_ODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True

        # self.x_model = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_size, self.x_hidden, bias = self.bias),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p = self.drop_out),
        #     torch.nn.Linear(self.x_hidden, self.gru_input_size, bias = self.bias)
        # )
        self.gru_layer = FullGRUODECell_Autonomous(
            self.hidden_size, bias=self.bias)
        self.gru_obs = torch.nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.hidden_size)
        if last_activation == 'identity':
            self.last_layer = None
        elif last_activation == 'softplus':
            self.last_layer = torch.nn.Softplus()
        elif last_activation == 'tanh':
            self.last_layer = torch.nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last_layer = torch.nn.Sigmoid()
        self.rec_linear = torch.nn.Linear(
            self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert self.impute == False
            solution, eval_times, eval_vals = odeint(
                func, h, torch.tensor([0, delta_t]))
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        # HH = self.x_model(H)
        HH = H
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to('cuda')
        current_time = times[0, 0] - 1
        final_time = times[0, -1]
        for idx, obs_time in enumerate(times[0]):
            while current_time < (obs_time-0.001*self.delta_t):
                if self.solver == 'dopri5':
                    h, current_time = self.ode_step(
                        h, self.gru_layer, obs_time-current_time, current_time)
                else:
                    h, current_time = self.ode_step(
                        h, self.gru_layer, self.delta_t, current_time)
            current_out, tmp = self.gru_obs(torch.reshape(
                HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])), h[None, :, :])
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + \
                current_out.reshape(HH.shape[0], HH.shape[-1])
        X_tilde = self.rec_linear(out)
        if self.last_layer != None:
            X_tilde = self.last_layer(X_tilde)
        return X_tilde


class Multi_Layer_ODENetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, num_layer, last_activation='identity', solver='euler'):
        super(Multi_Layer_ODENetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.drop_out = 0
        self.solver = solver
        self.impute = False
        self.bias = True
        self.num_layer = num_layer
        self.last_activation= last_activation

        if num_layer == 1:
            self.model = RecoveryODENetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                                            gru_input_size=gru_input_size, x_hidden=x_hidden, last_activation=self.last_activation,delta_t=delta_t, solver=solver)
        elif num_layer == 2:
            self.model = torch.nn.ModuleList(
                [
                    First_ODENetwork(input_size=input_size, hidden_size=hidden_size, output_size=hidden_size,
                                     gru_input_size=gru_input_size, x_hidden=x_hidden, delta_t=delta_t, solver=solver),
                    Last_ODENetwork(input_size=hidden_size, hidden_size=hidden_size, output_size=output_size,
                                    gru_input_size=gru_input_size, x_hidden=x_hidden, last_activation=self.last_activation,delta_t=delta_t, solver=solver)
                ]
            )
        else:
            self.model = torch.nn.ModuleList()
            for i in range(num_layer):
                if i == 0:
                    self.model.append(First_ODENetwork(input_size=input_size, hidden_size=hidden_size, output_size=hidden_size,
                                      gru_input_size=gru_input_size, x_hidden=x_hidden, delta_t=delta_t, solver=solver))
                elif i == num_layer-1:
                    self.model.append(Last_ODENetwork(input_size=hidden_size, hidden_size=hidden_size, output_size=output_size,
                                      gru_input_size=gru_input_size, x_hidden=x_hidden,last_activation=self.last_activation, delta_t=delta_t, solver=solver))
                else:
                    self.model.append(Mid_ODENetwork(input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size,
                                      gru_input_size=gru_input_size, x_hidden=x_hidden, delta_t=delta_t, solver=solver))

    def forward(self, H, times):
        if self.num_layer == 1:
            out = self.model(H, times)
        else:
            out = H
            for model in self.model:
                out = model(out, times)
        return out


def train(
    args,
    batch_size,
    max_steps,
    dataset,
    device,
    embedder,
    generator,
    supervisor,
    recovery,
    discriminator,
    gamma,
):
    def _loss_e_t0(x_tilde, x):
        return F.mse_loss(x_tilde, x)

    def _loss_e_0(loss_e_t0):
        return torch.sqrt(loss_e_t0) * 10

    def _loss_e(loss_e_0, loss_s):
        return loss_e_0 + 0.1 * loss_s

    def _loss_s(h_hat_supervise, h):
        return F.mse_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])

    def _loss_g_u(y_fake):
        return F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))

    def _loss_g_u_e(y_fake_e):
        return F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))

    def _loss_g_v(x_hat, x):
        loss_g_v1 = torch.mean(
            torch.abs(torch.sqrt(torch.var(x_hat, 0) + 1e-6) -
                      torch.sqrt(torch.var(x, 0) + 1e-6))
        )
        loss_g_v2 = torch.mean(
            torch.abs(torch.mean(x_hat, 0) - torch.mean(x, 0)))
        return loss_g_v1 + loss_g_v2

    def _loss_g(loss_g_u, loss_g_u_e, loss_s, loss_g_v):
        return loss_g_u + gamma * loss_g_u_e + 100 * torch.sqrt(loss_s) + 100 * loss_g_v

    def _loss_g2(loss_g_u, loss_s, loss_g_v):
        return loss_g_u + loss_s + 100 * loss_g_v

    def _loss_g3(loss_g_u, loss_g_v):
        return loss_g_u + 100 * loss_g_v

    def _loss_d(y_real, y_fake, y_fake_e):
        loss_d_real = F.binary_cross_entropy_with_logits(
            y_real, torch.ones_like(y_real))
        loss_d_fake = F.binary_cross_entropy_with_logits(
            y_fake, torch.zeros_like(y_fake))
        loss_d_fake_e = F.binary_cross_entropy_with_logits(
            y_fake_e, torch.zeros_like(y_fake_e))
        return loss_d_real + loss_d_fake + gamma * loss_d_fake_e

    def _loss_d2(y_real, y_fake):
        loss_d_real = F.binary_cross_entropy_with_logits(
            y_real, torch.ones_like(y_real))
        loss_d_fake = F.binary_cross_entropy_with_logits(
            y_fake, torch.zeros_like(y_fake))
        return loss_d_real + loss_d_fake

    optimizer_er = optim.Adam(chain(embedder.parameters(), recovery.parameters()))
    optimizer_gs = optim.Adam(generator.parameters())
    optimizer_d = optim.Adam(discriminator.parameters())

    embedder.train()
    generator.train()
    supervisor.train()
    recovery.train()
    discriminator.train()
    
    print("Start Embedding Network Training")
    for step in range(1, args.first_epoch + 1):
        batch = dataset[batch_size]
        x = batch['data'].to(device)
        train_coeffs = batch['inter']
        original_x = batch['original_data'].to(device)
        obs = x[:, :, -1]
        x = x[:, :, :-1]
        time = torch.FloatTensor(list(range(24))).cuda()
        final_index = (torch.ones(batch_size) * 23).cuda()
        h = embedder(time, train_coeffs, final_index)
        x_tilde = recovery(h, obs)
        x_no_nan = x[~torch.isnan(x)]
        x_tilde_no_nan = x_tilde[~torch.isnan(x)]
        loss_e_t0 = _loss_e_t0(x_tilde_no_nan, x_no_nan)
        loss_e_0 = _loss_e_0(loss_e_t0)
        optimizer_er.zero_grad()
        loss_e_0.backward()
        optimizer_er.step()
        torch.cuda.empty_cache()
        print(
            "step: "
            + str(step)
            + "/"
            + str(args.first_epoch)
            + ", loss_e: "
            + str(np.round(np.sqrt(loss_e_t0.item()), 4))
        )
    print("Finish Embedding Network Training")
    
    print("Start Joint Training")
    for step in range(1, max_steps+1):
        for _ in range(2):
            generator.train()
            supervisor.train()
            recovery.train()

            batch = dataset[batch_size]
            x = batch['data'].to(device)
            train_coeffs = batch['inter']
            original_x = batch['original_data'].to(device)
            obs = x[:, :, -1]
            x = x[:, :, :-1]
            z = torch.randn(batch_size, x.size(1), args.effective_shape).to(device)
            time = torch.FloatTensor(list(range(24))).cuda()
            final_index = (torch.ones(batch_size) * 23).cuda()
            h = embedder(time, train_coeffs, final_index)
            times = time
            times = times.unsqueeze(0)
            times = times.unsqueeze(2)
            times = times.repeat(obs.shape[0], 1, 1)
            h_hat = run_model(args, generator, z, times, device, z=True)
            x_real = recovery(h, obs)
            x_fake = recovery(h_hat, obs)
            y_fake = discriminator(x_fake, obs)
            y_real = discriminator(x_real, obs)
            loss_d = _loss_d2(y_real, y_fake)

            if loss_d.item() > 0.15:
                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()
                torch.cuda.empty_cache()

            #############Recovery######################
            h = embedder(time, train_coeffs, final_index)
            x_tilde = recovery(h, obs)

            x_no_nan = x[~torch.isnan(x)]
            x_tilde_no_nan = x_tilde[~torch.isnan(x)]
            loss_e_t0 = _loss_e_t0(x_tilde_no_nan, x_no_nan)

            loss_e_0 = _loss_e_0(loss_e_t0)
            loss_e = loss_e_0
            optimizer_er.zero_grad()
            loss_e.backward()
            optimizer_er.step()
            torch.cuda.empty_cache()

        if step % args.log_time == 0:
            batch = dataset[batch_size]
            x = batch['data'].to(device)
            train_coeffs = batch['inter']#.to(device)
            original_x = batch['original_data'].to(device)
            obs = x[:, :, -1]
            x = x[:, :, :-1]
            time = torch.FloatTensor(list(range(24))).cuda()
            final_index = (torch.ones(batch_size) * 23).cuda()

            h = embedder(time, train_coeffs, final_index)
            times = time
            times = times.unsqueeze(0)
            times = times.unsqueeze(2)
            times = times.repeat(obs.shape[0], 1, 1)
            #################################################
            if args.kinetic_energy == None:
                loss_s, loss = run_model(
                    args, generator, h, times, device, z=False)
                optimizer_gs.zero_grad()
                loss_s.backward()
            else:
                loss_s, loss, reg_state = run_model(
                    args, generator, h, times, device, z=False)
                optimizer_gs.zero_grad()
                (loss_s+reg_state).backward()
            optimizer_gs.step()

        batch = dataset[batch_size]
        x = batch['data'].to(device)
        train_coeffs = batch['inter']#.to(device)
        original_x = batch['original_data'].to(device)
        obs = x[:, :, -1]
        x = x[:, :, :-1]
        time = torch.FloatTensor(list(range(24))).cuda()
        final_index = (torch.ones(batch_size) * 23).cuda()
        z = torch.randn(batch_size, x.size(1), args.effective_shape).to(device)
        h = embedder(time, train_coeffs, final_index)
        times = time.unsqueeze(0)
        times = times.unsqueeze(2)
        times = times.repeat(obs.shape[0], 1, 1)
        h_hat = run_model(args, generator, z, times, device, z=True)

        x_hat = recovery(h_hat, obs)

        x_no_nan = x[~torch.isnan(x)]
        x_hat_no_nan = x_hat[~torch.isnan(x)]

        y_fake = discriminator(x_hat, obs)
        loss_g_u = _loss_g_u(y_fake)
        loss_g_v = _loss_g_v(x_no_nan, x_hat_no_nan)
        loss_g = _loss_g3(loss_g_u, loss_g_v)
        optimizer_gs.zero_grad()
        loss_g.backward()
        optimizer_gs.step()
        if step > 4:
            print(
                "step: "
                + str(step)
                + "/"
                + str(max_steps)
                + ", loss_d: "
                + str(np.round(loss_d.item(), 4))
                + ", loss_g_u: "
                + str(np.round(loss_g_u.item(), 4))
                + ", loss_g_v: "
                + str(np.round(loss_g_v.item(), 4))
                + ", loss_s: "
                + str(np.round(loss_s.item(), 4))
                + ", loss_e_t0: "
                + str(np.round(np.sqrt(loss_e_t0.item()), 4))
            )
        if step % 500 == 0:
            path = args.save_dir
            torch.save(embedder.state_dict(), path +
                       "/embedder{}.pt".format(str(step)))
            torch.save(recovery.state_dict(), path +
                       "/recovery{}.pt".format(str(step)))
            torch.save(generator.state_dict(), path +
                       "/generator{}.pt".format(str(step)))
            torch.save(discriminator.state_dict(), path +
                       "/discriminator{}.pt".format(str(step)))

            seq_len = x.shape[1]
            input_size = x.shape[2] - 1
            dataset_size = dataset.size

            with torch.no_grad():
                batch = dataset[dataset_size]
                x = batch['data'].to(device)
                train_coeffs = batch['inter']#.to(device)
                original_x = batch['original_data'].to(device)
                obs = x[:, :, -1]
                x = x[:, :, :-1]
                z = torch.randn(dataset_size, x.size(
                    1), args.effective_shape).to(device)
                time = torch.FloatTensor(list(range(24))).cuda()

                final_index = (torch.ones(dataset_size) * 23).cuda()

                ###########################################
                h = embedder(time, train_coeffs, final_index)
                times = time
                times = times.unsqueeze(0)
                times = times.unsqueeze(2)
                times = times.repeat(obs.shape[0], 1, 1)
                ###########################################
                h_hat = run_model(args, generator, z, times, device, z=True)
                ###########################################
                x_hat = recovery(h_hat, obs)
                x = original_x
                
            generated_data_curr = x.cpu().numpy()
            generated_data1 = list()
            for i in range(dataset_size):
                temp = generated_data_curr[i, :, :]
                generated_data1.append(temp)

            generated_data_curr = x_hat.cpu().numpy()
            generated_data2 = list()
            for i in range(dataset_size):
                temp = generated_data_curr[i, :, :]
                generated_data2.append(temp)
                
            name = 'third'+str(step)
            visualization(generated_data1, generated_data2,
                          "tsne", args)

            metric_results = dict()
            tf.compat.v1.disable_eager_execution()
            discriminative_score = list()
            for _ in range(args.max_steps_metric):
                temp_disc = discriminative_score_metrics(
                    generated_data1, generated_data2, True)
                discriminative_score.append(temp_disc)

            metric_results['discriminative'] = np.mean(discriminative_score)
            metric_results['discriminative_std'] = np.std(discriminative_score)

            predictive_score = list()
            for tt in range(args.max_steps_metric):
                if args.missing_value == 0.0:
                    temp_pred = predictive_score_metrics2(
                        generated_data1, generated_data2)
                else:
                    temp_pred = predictive_score_metrics(
                        generated_data1, generated_data2)  
                predictive_score.append(temp_pred)

            metric_results['predictive'] = np.mean(predictive_score)
            metric_results['predictive_std'] = np.std(predictive_score)
            print(metric_results)
    print("Finish Joint Training")

    generator.eval()
    supervisor.eval()
    recovery.eval()
    
    generated_data_curr = x_hat.cpu().numpy()
    generated_data = list()

    return generated_data


def train_test_divide(data_x, data_x_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat


def visualize(dataset, device, generated_data, args, name=''):
    dataset = np.delete(dataset, -1, axis=2)
    visualization(dataset, generated_data, "tsne", args)


def main():
    parser = parse_arguments()
    parser.add_argument("--data", type=str, default="stock")
    parser.add_argument("--model1", type=str, default='gtgan')
    parser.add_argument("--model2", type=str, default='add_discriminator')
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--max-steps-metric", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--save_dir", type=str, default='test')
    parser.add_argument("--d_layer", type=int, default=2)
    parser.add_argument("--r_layer", type=int, default=2)
    parser.add_argument("--last_activation_r", type=str, default='sigmoid')
    parser.add_argument("--last_activation_d", type=str, default='identity')
    parser.add_argument("--log_time", type=int, default=2)
    parser.add_argument("--missing_value", type=float, default=0.0)
    parser.add_argument("--first_epoch",type=int,default=10000)
    here = pathlib.Path(__file__).resolve().parent
    args = parser.parse_args()
    args.effective_shape = args.hidden_dim
    # args.aug_mapping = True
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(device)
    if args.data == 'stock':
        data_path = here / 'datasets/stock_data.csv'
        dataset = TimeDataset(data_path, args.seq_len,'stock', args.missing_value)
        input_size = 6
        # dataset[100]
    if args.data == 'energy':
        data_path = here / 'datasets/energy_data.csv'
        dataset = TimeDataset(data_path, args.seq_len, 'energy', args.missing_value)
        # original_dataset = TimeDataset(data_path, args.seq_len)
        input_size = 28
    if args.data == 'sine':
        dataset = SineDataset(10000,args.seq_len,5,'sine',args.missing_value)
        input_size = 5
    if args.data == 'mujoco':
        data_path = here / 'mujoco_data'
        dataset = MujocoDataset(data_path, 'mujoco', args.missing_value)
        # original_dataset = MujocoDataset(data_path)
        input_size = 14 
    # min_val max_val for denormalization
    # dataset => normalization
    #dataset = TimeDataset(args.data_path, args.seq_len)
    # input_size = dataset[0].shape[1] - 1
    hidden_size = args.hidden_dim
    num_layers = 3
    x_hidden = 48

    path = here
    print(path)
    args.save_dir = str(path)+'/'+args.save_dir
    os.makedirs(args.save_dir,exist_ok=True)
    def cvt(x): return x.type(torch.float32).to(device, non_blocking=True)

    if args.model1 == 'new_embedding':
        ode_func = FinalTanh(input_size, hidden_size, hidden_size, num_layers)
        ####4###############
        #embedder = Net(input_size, hidden_size, hidden_size, num_layers).to(device)
        embedder = NeuralCDE(func=ode_func, input_channels=input_size,
                             hidden_channels=hidden_size, output_channels=hidden_size).to(device)
        ###################
        recovery = RecoveryODENetwork(
            input_size, hidden_size, num_layers, x_hidden, delta_t=0.5).to(device)
        # recovery = Net(hidden_size, hidden_size, input_size, num_layers).to(device)
        generator = Net(input_size, hidden_size,
                        hidden_size, num_layers).to(device)
        supervisor = Net(hidden_size, hidden_size,
                         hidden_size, num_layers - 1).to(device)
        discriminator = Net(hidden_size, hidden_size, 1,
                            num_layers, activation_fn=None).to(device)
        #discriminator_metric = Net2(6, int(6/2), 1, 1, activation_fn=None).to(device)
    elif args.model1 == 'add_generator':
        ode_func = FinalTanh(input_size, hidden_size, hidden_size, num_layers)
        embedder = NeuralCDE(func=ode_func, input_channels=input_size,
                             hidden_channels=hidden_size, output_channels=hidden_size).to(device)
        recovery = RecoveryODENetwork(
            input_size, hidden_size, num_layers, x_hidden, delta_t=0.5).to(device)
        generator = GeneratorODENetwork(
            input_size, hidden_size, num_layers, x_hidden, delta_t=1).to(device)
        supervisor = Net(hidden_size, hidden_size,
                         hidden_size, num_layers - 1).to(device)
        discriminator = Net(hidden_size, hidden_size, 1,
                            num_layers, activation_fn=None).to(device)
    elif args.model1 == 'gtgan':
        ode_func = FinalTanh(input_size, hidden_size, hidden_size, num_layers)
        embedder = NeuralCDE(func=ode_func, input_channels=input_size,
                             hidden_channels=hidden_size, output_channels=hidden_size).to(device)
        recovery = Multi_Layer_ODENetwork(input_size=hidden_size, hidden_size=hidden_size, output_size=input_size,gru_input_size=hidden_size, x_hidden=x_hidden, num_layer=args.r_layer, last_activation=args.last_activation_r,delta_t=0.5).to(device)
        regularization_fns, regularization_coeffs = create_regularization_fns(
            args)
        # generator = build_augmented_model_tabular(args,args.effective_shape,regularization_fns=regularization_fns,).to(device)
        generator = build_model_tabular_nonlinear(
            args, args.effective_shape, regularization_fns=regularization_fns,).to(device)
        # generator = build_model_tabular_original(args,args.effective_shape,regularization_fns=regularization_fns,).to(device)
        set_cnf_options(args, generator)
        # dummy
        supervisor = Net(hidden_size, hidden_size,
                         hidden_size, num_layers - 1).to(device)
        discriminator = Multi_Layer_ODENetwork(input_size=input_size, hidden_size=hidden_size, output_size=1,
                                               gru_input_size=hidden_size, x_hidden=x_hidden, last_activation=args.last_activation_d,num_layer=args.d_layer, delta_t=0.5).to(device)
    elif args.model1 == 'timegan':
        embedder = Net(input_size, hidden_size,
                       hidden_size, num_layers).to(device)
        recovery = Net(hidden_size, hidden_size,
                       input_size, num_layers).to(device)
        generator = Net(input_size, hidden_size,
                        hidden_size, num_layers).to(device)
        supervisor = Net(hidden_size, hidden_size,
                         hidden_size, num_layers - 1).to(device)
        discriminator = Net(hidden_size, hidden_size, 1,
                            num_layers, activation_fn=None).to(device)

    print('model created')
    pytorch_total_params = sum(p.numel()
                               for p in generator.parameters() if p.requires_grad)
    print(pytorch_total_params)
    pytorch_total_params = sum(
        p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(pytorch_total_params)

    if args.train == True:
        generated_data = train(
            args,
            args.batch_size,
            args.max_steps,
            dataset,
            device,
            embedder,
            generator,
            supervisor,
            recovery,
            discriminator,
            args.gamma,
        )
        # visualize(dataset, device, generated_data,args)
    else:                             
        if args.missing_value == 0.0:
            path = 'energy_model/energy'
            generator.load_state_dict(torch.load(
                path+"/generator.pt"))
            recovery.load_state_dict(torch.load(
                path+"/recovery.pt"))
        elif args.missing_value == 0.5:
            path = 'energy_model/energy_0.5'
            generator.load_state_dict(torch.load(
                path+"/generator.pt"))
            recovery.load_state_dict(torch.load(
                path+"/recovery.pt"))

        dataset_size = len(dataset)
        generator.eval()
        recovery.eval()

        with torch.no_grad():
            batch = dataset[dataset_size]
            x = batch['data'].to(device)
            train_coeffs = batch['inter']#.to(device)
            original_x = batch['original_data'].to(device)
            obs = x[:, :, -1]
            x = x[:, :, :-1]
            z = torch.randn(dataset_size, x.size(
                1), args.effective_shape).to(device)
            time = torch.FloatTensor(list(range(24))).cuda()
            times = time
            times = times.unsqueeze(0)
            times = times.unsqueeze(2)
            times = times.repeat(obs.shape[0], 1, 1)
            h_hat = run_model(args, generator, z, times, device, z=True)
            x_hat = recovery(h_hat, obs)
            x = original_x

        generated_data_curr = x.cpu().numpy()
        generated_data1 = list()
        for i in range(dataset_size):
            temp = generated_data_curr[i, :, :]
            generated_data1.append(temp)

        generated_data_curr = x_hat.cpu().numpy()
        generated_data2 = list()
        for i in range(dataset_size):
            temp = generated_data_curr[i, :, :]
            generated_data2.append(temp)
        visualization(generated_data1, generated_data2,"tsne",args)
        metric_results = dict()
        discriminative_score = list()
        for tt in range(args.max_steps_metric):
            temp_pred = discriminative_score_metrics(
                generated_data1, generated_data2, True)
            discriminative_score.append(temp_pred)

        metric_results['discriminative'] = np.mean(discriminative_score)
        metric_results['discriminative_std'] = np.std(discriminative_score)
        
        predictive_score = list()
        for tt in range(args.max_steps_metric):
            if args.missing_value == 0.0:
                temp_pred = predictive_score_metrics2(
                    generated_data1, generated_data2)
            else:
                temp_pred = predictive_score_metrics(
                    generated_data1, generated_data2)                
            predictive_score.append(temp_pred)

        metric_results['predictive'] = np.mean(predictive_score)
        metric_results['predictive_std'] = np.std(predictive_score)
        print(metric_results)
        return 0


if __name__ == "__main__":
    main()
