import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde
from torchdiffeq import odeint


class CDEFunction(nn.Module):
    def __init__(self, input_channels, hidden_channels, second_hidden_channels, num_layers=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.init_linear = nn.Linear(hidden_channels, second_hidden_channels)
        self.linear_layers = nn.ModuleList([nn.Linear(second_hidden_channels, second_hidden_channels) for _ in range(num_layers)])
        self.final_linear = nn.Linear(second_hidden_channels, input_channels * hidden_channels)

    def forward(self, t, z):            # t is needed for torchcde library, just a placeholder
        # z has shape (batch, hidden_channels)
        z = self.init_linear(z)
        z = F.relu(z)
        
        for linear in self.linear_layers:
            z = linear(z)
            z = F.relu(z)

        z = self.final_linear(z)
        z = F.tanh(z)           # recommended to use tanh by the author
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        
        # attributes
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        
        # neural networks
        self.cde_function = CDEFunction(
            input_channels=input_channels, 
            hidden_channels=hidden_channels, 
            second_hidden_channels=hidden_channels, 
        )
        self.fc_init = nn.Linear(input_channels, hidden_channels)
        self.fc_last = nn.Linear(hidden_channels, output_channels)
    
    def get_min_diff(self, times):
        time_diffs = times[1:] - times[:-1]
        return time_diffs.min().item()
    
    def forward(self, times, coeffs):
        spline = torchcde.CubicSpline(coeffs)
        x0 = spline.evaluate(spline.interval[0])
        z0 = self.fc_init(x0)
        
        min_diff = self.get_min_diff(times)
        
        zt = torchcde.cdeint(
            X = spline, 
            z0 = z0, 
            func = self.cde_function, 
            t = times,
            method = "rk4",
            options = {"step_size": min_diff},
        )
        
        out = self.fc_last(zt)
        out = F.sigmoid(out)                # NOTE: make sure output in [0, 1]
        return out


class FullGRUODECell_Autonomous(nn.Module):
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()

        #self.lin_xh = nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xz = nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xr = nn.Linear(input_size, hidden_size, bias=bias)

        #self.lin_x = nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        """
        Executes one step with autonomous GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            t        time of evaluation
            h        hidden state (current)

        Returns:
            Updated h
        """
        #xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh


class First_ODENetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, solver='euler'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.solver = solver
        self.impute = False
        self.bias = True

        self.x_model = nn.Sequential(
            nn.Linear(self.input_size, self.x_hidden, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.x_hidden, self.gru_input_size, bias=self.bias)
        )
        self.gru_layer = FullGRUODECell_Autonomous(self.hidden_size, bias=self.bias)
        self.gru_obs = nn.GRU(input_size=self.gru_input_size, hidden_size=self.hidden_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert self.impute == False
            event_times, solution = odeint(func, h, torch.tensor([0, delta_t]))
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        HH = self.x_model(H)            # input layer need to initially use x_model to obtain hidden states
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to('cuda')
        # h = torch.zeros(HH.shape[0], self.hidden_size).to('mps:0')
        current_time = times[0, 0]
        for idx, obs_time in enumerate(times[0]):

            while current_time < (obs_time-0.001*self.delta_t):
                if self.solver == 'dopri5':
                    h, current_time = self.ode_step(h, self.gru_layer, obs_time-current_time, current_time)
                else:
                    h, current_time = self.ode_step(h, self.gru_layer, self.delta_t, current_time)
            
            current_out, tmp = self.gru_obs(torch.reshape(HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])), h[None, :, :])
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + current_out.reshape(HH.shape[0], HH.shape[-1])

        return out          # pass to other layers does not need FC since it doesn't generate final time series


class Mid_ODENetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, solver='euler'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.solver = solver
        self.impute = False
        self.bias = True
        
        self.gru_layer = FullGRUODECell_Autonomous(self.hidden_size, bias=self.bias)
        self.gru_obs = nn.GRU(input_size=self.gru_input_size, hidden_size=self.hidden_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert self.impute == False
            solution, eval_times, eval_vals = odeint(func, h, torch.tensor([0, delta_t]))
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        # HH = self.x_model(H)
        HH = H      # for mid layers, just use the input
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to('cuda')
        current_time = times[0, 0]
        for idx, obs_time in enumerate(times[0]):

            while current_time < (obs_time-0.001*self.delta_t):
                if self.solver == 'dopri5':
                    h, current_time = self.ode_step(h, self.gru_layer, obs_time-current_time, current_time)
                else:
                    h, current_time = self.ode_step(h, self.gru_layer, self.delta_t, current_time)

            current_out, tmp = self.gru_obs(torch.reshape(HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])), h[None, :, :])
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + current_out.reshape(HH.shape[0], HH.shape[-1])
            
        return out          # for mid layers, just pass the output (no FC)


class Last_ODENetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, last_activation='identity', solver='euler'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.solver = solver
        self.impute = False
        self.bias = True

        self.gru_layer = FullGRUODECell_Autonomous(self.hidden_size, bias=self.bias)
        self.gru_obs = nn.GRU(input_size=self.gru_input_size, hidden_size=self.hidden_size)
        
        if last_activation == 'identity':
            self.last_layer = None
        elif last_activation == 'softplus':
            self.last_layer = nn.Softplus()
        elif last_activation == 'tanh':
            self.last_layer = nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last_layer = nn.Sigmoid()
            
        self.rec_linear = nn.Linear(self.gru_input_size, self.output_size)

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
        HH = H
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to('cuda')
        # h = torch.zeros(HH.shape[0], self.hidden_size).to('mps:0')
        current_time = times[0, 0]
        for idx, obs_time in enumerate(times[0]):
            
            while current_time < (obs_time-0.001*self.delta_t):
                if self.solver == 'dopri5':
                    h, current_time = self.ode_step(h, self.gru_layer, obs_time-current_time, current_time)
                else:
                    h, current_time = self.ode_step(h, self.gru_layer, self.delta_t, current_time)
            
            current_out, tmp = self.gru_obs(torch.reshape(HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])), h[None, :, :])
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))
            out[:, idx, :] = out[:, idx, :] + current_out.reshape(HH.shape[0], HH.shape[-1])
        X_tilde = self.rec_linear(out)
        if self.last_layer != None:
            X_tilde = self.last_layer(X_tilde)

        return X_tilde


class RecoveryODENetwork(nn.Module):
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
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.solver = solver
        self.impute = False
        self.bias = True

        self.x_model = nn.Sequential(
            nn.Linear(self.input_size, self.x_hidden, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.x_hidden, self.gru_input_size, bias=self.bias)
        )
        self.gru_layer = FullGRUODECell_Autonomous(self.hidden_size, bias=self.bias)
        self.gru_obs = nn.GRU(input_size=self.gru_input_size, hidden_size=self.hidden_size)

        if last_activation == 'identity':
            self.last_layer = None
        elif last_activation == 'softplus':
            self.last_layer = nn.Softplus()
        elif last_activation == 'tanh':
            self.last_layer = nn.Tanh()
        elif last_activation == 'sigmoid':
            self.last_layer = nn.Sigmoid()
            
        self.rec_linear = nn.Linear(self.gru_input_size, self.output_size)

    def ode_step(self, h, func, delta_t, current_time):
        if self.solver == "euler":
            h = h + delta_t * func(t=0, h=h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(h=h)
            h = h + delta_t * func(h=k)
        elif self.solver == "dopri5":
            # Dopri5 solver is only compatible with autonomous ODE.
            assert self.impute == False
            solution, eval_times, eval_vals = odeint(func, h, torch.tensor([0, delta_t]))
            h = solution[1, :, :]
        current_time += delta_t
        return h, current_time

    def forward(self, H, times):
        HH = self.x_model(H)
        out = torch.zeros_like(HH)
        h = torch.zeros(HH.shape[0], self.hidden_size).to('cuda')
        current_time = times[0, 0]
        for idx, obs_time in enumerate(times[0]):
            
            while current_time < (obs_time-0.001*self.delta_t):         # solve ODE, this part is GRU-ODE
                if self.solver == 'dopri5':
                    h, current_time = self.ode_step(h, self.gru_layer, obs_time-current_time, current_time)
                else:
                    h, current_time = self.ode_step(h, self.gru_layer, self.delta_t, current_time)
            
            current_out, tmp = self.gru_obs(torch.reshape(HH[:, idx, :], (1, HH.shape[0], HH.shape[-1])), h[None, :, :])            # use GRU to generate a sequence of outputs for decoding
            h = torch.reshape(tmp, (h.shape[0], h.shape[1]))                                                # new hidden state is used in next iteration
            out[:, idx, :] = out[:, idx, :] + current_out.reshape(HH.shape[0], HH.shape[-1])                # append generated sequence to output (time stamp is 2nd dimension)
            
        X_tilde = self.rec_linear(out)                  # one more FC for decoding to generate synthetic time series
        if self.last_layer != None:
            X_tilde = self.last_layer(X_tilde)
            
        return X_tilde


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gru_input_size, x_hidden, delta_t, num_layer, last_activation='identity', solver='euler'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.x_hidden = x_hidden
        self.gru_input_size = gru_input_size
        self.delta_t = delta_t
        self.solver = solver
        self.impute = False
        self.bias = True
        self.num_layer = num_layer
        self.last_activation= last_activation

        if num_layer == 1:
            self.model = RecoveryODENetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                                            gru_input_size=gru_input_size, x_hidden=x_hidden, last_activation=self.last_activation,delta_t=delta_t, solver=solver)
        elif num_layer == 2:
            self.model = nn.ModuleList(
                [
                    First_ODENetwork(input_size=input_size, hidden_size=hidden_size, output_size=hidden_size,
                                     gru_input_size=gru_input_size, x_hidden=x_hidden, delta_t=delta_t, solver=solver),
                    Last_ODENetwork(input_size=hidden_size, hidden_size=hidden_size, output_size=output_size,
                                    gru_input_size=gru_input_size, x_hidden=x_hidden, last_activation=self.last_activation,delta_t=delta_t, solver=solver)
                ]
            )
        else:
            self.model = nn.ModuleList()
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
        