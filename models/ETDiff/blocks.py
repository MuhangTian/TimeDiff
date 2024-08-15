import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde
from einops import rearrange

import models.ETDiff.utils as utils


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CDEFunction(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, second_hidden_channels: int, num_layers: int=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.init_linear = nn.Linear(hidden_channels, second_hidden_channels)
        self.linear_layers = nn.ModuleList([nn.Linear(second_hidden_channels, second_hidden_channels) for _ in range(num_layers)])
        self.final_linear = nn.Linear(second_hidden_channels, input_channels * hidden_channels)

    def forward(self, t, z):            # t is needed for torchcde library, just a placeholder
        z = self.init_linear(z)
        z = F.relu(z)
        
        for linear in self.linear_layers:
            z = linear(z)
            z = F.relu(z)

        z = self.final_linear(z)
        z = F.tanh(z)           # recommended to use tanh by Neural CDE author
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class FirstLinearBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, bias: bool=True, groups: int=8):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2),
        ) if utils.exists(time_dim) else None
        
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = nn.GroupNorm(groups, out_channels)
    
    def forward(self, x, time_embed):
        time_embed = self.time_mlp(time_embed)
        time_embed = rearrange(time_embed, 'b c -> b c 1')
        time_tuple = time_embed.chunk(2, dim=1)
        
        x = self.linear(x)
        x = self.norm(x)
        
        scaling, shifts = time_tuple
        x = x.unsqueeze(-1) * (scaling + 1) + shifts
        return x.squeeze(-1)


class LastLinearBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, bias: bool=True, groups: int=10):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2),
        ) if utils.exists(time_dim) else None
        
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = nn.GroupNorm(groups, out_channels)
    
    def forward(self, x, time_embed):
        time_embed = self.time_mlp(time_embed)
        time_embed = rearrange(time_embed, 'b c -> b c 1')
        time_tuple = time_embed.chunk(2, dim=1)
        
        x = self.linear(x)
        x = x.permute(0, 2, 1)              # permute for group norm
        x = self.norm(x)
        
        scaling, shifts = time_tuple
        x = x * (scaling + 1) + shifts
        x = x.permute(0, 2, 1)              # permute back
        return x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, utils.default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, utils.default(dim_out, dim), 4, 2, 1)

class NeuralCDE(nn.Module):
    def __init__(
        self,
        input_channels: int,
        seq_len: int,
        hidden_channels: int,
        second_hidden_channels: int,
        num_layers: int,
        output_channels: int,
        self_condition: bool=False,
        embed_dim: int= 64,
        time_dim: int= 256,
        solver: str = "rk4",
    ):
        super().__init__()
        assert all(isinstance(var, int) for var in [input_channels, hidden_channels, second_hidden_channels, num_layers, output_channels, embed_dim, time_dim, seq_len]), "must be integer!"
        
        # attributes
        self.self_condition = self_condition
        self.time_dim_partition = seq_len
        self.solver = solver
        
        # neural CDE networks
        self.cde_func = CDEFunction(
            input_channels = input_channels, 
            hidden_channels = hidden_channels, 
            second_hidden_channels = second_hidden_channels,
            num_layers = num_layers,
        )
        self.fc_init = FirstLinearBlock(input_channels, hidden_channels, time_dim, groups=8)                 # groups need to be adjusted based on input
        self.fc_last = LastLinearBlock(hidden_channels*8, output_channels, time_dim, groups=2)    
        
        # time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.upsample1 = Upsample(1, seq_len // 4)
        self.upsample2 = Upsample(seq_len // 4, seq_len // 2)
        self.upsample3 = Upsample(seq_len // 2, seq_len)
        # RNN for upsampling
        # self.rnn = nn.GRU()
    
    def get_min_diff(self, times: torch.Tensor):
        time_diffs = times[1:] - times[:-1]
        return time_diffs.min().item()
    
    def process_times(self, spline):
        if self.time_dim_partition is None or self.time_dim_partition == 2:
            return spline.interval          # use end time and start time
        elif self.time_dim_partition > 2:
            return torch.linspace(spline.interval[0], spline.interval[1], self.time_dim_partition)
        else:
            raise ValueError(f"Invalid time partition value!\nGot \"{self.time_dim_partition}\" but it can be only None or integer >= 2!")

    def forward(self, x, time, x_self_cond = None):
        """
        Defines forward pass for prediction, x is some noisy sample.
        x is of dimension (batch_size, timesteps, channels)
        """
        x = x.permute(0, 2, 1)                          # permute to fit neural CDE
        if self.self_condition:
            x_self_cond = utils.default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)           # slow, but it must be here
        spline = torchcde.CubicSpline(coeffs)
        x = spline.evaluate(spline.interval[0])
        
        t = self.time_emb(time)                         # this is for diffusion time steps
        
        x = self.fc_init(x, t)                          # initial value
        # times = self.process_times(spline).to(next(self.parameters()).device)              # this is for neural CDE solutions
        # min_diff = self.get_min_diff(times)
        
        # solve CDE
        x = torchcde.cdeint(
            X = spline,
            z0 = x,
            func = self.cde_func,
            t = spline.interval,
            method = self.solver,
            # options = {"step_size": min_diff},
        )
        x = x[:, 1].unsqueeze(1)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = F.silu(self.fc_last(x, t))
        return x.permute(0, 2, 1)                       # permute back to fit GaussianDiffusion()


class RMSNorm(nn.Module):
    """
    Reference
    ---------
        Zhang, Biao, and Rico Sennrich. "Root mean square layer normalization." Advances in Neural Information Processing Systems 32 (2019).
    """
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class EfficientAttention(nn.Module):
    """
    Reference
    ---------
        Shen, Zhuoran, et al. "Efficient attention: Attention with linear complexities." Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2021.
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)
        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)


class RNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, layers, batch_first, dropout, bidirectional, time_dim, model):
        super().__init__()
        if model == "lstm":
            self.rnn = nn.LSTM(input_channels, output_channels, layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif model == "gru":
            self.rnn = nn.GRU(input_channels, output_channels, layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError(f"Invalid model name \"{model}\"! It can be only \"lstm\" or \"gru\"!")
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, output_channels * 4 if bidirectional else output_channels * 2),
        )
        # self.norm = nn.GroupNorm(groups, output_channels * 2 if bidirectional else output_channels)
        # self.norm = RMSNorm(output_channels * 2 if bidirectional else output_channels)
        self.norm = nn.LayerNorm(output_channels * 2 if bidirectional else output_channels)
    
    def forward(self, x, time_embed):
        """input is (batch_size, channels, timesteps)"""
        x = x.permute(0, 2, 1)
        time_embed = self.time_mlp(time_embed)
        time_embed = rearrange(time_embed, 'b c -> b c 1')
        time_tuple = time_embed.chunk(2, dim=1)
        
        outputs = self.rnn(x)
        hidden = outputs[0]
        # hidden = hidden.permute(0, 2, 1)
        # x = self.norm(hidden)
        x = self.norm(hidden).permute(0, 2, 1)
        
        scaling, shifts = time_tuple
        x = x * (scaling + 1) + shifts
        return x.permute(0, 2, 1)


class RNN(nn.Module):
    def __init__(
        self, 
        input_channels, 
        hidden_channels, 
        output_channels,
        layers, 
        model = "lstm",
        dropout = 0, 
        bidirectional  = True, 
        self_condition = False,
        embed_dim = 64,
        time_dim = 256,
        ):
        super().__init__()
        self.self_condition = self_condition
        
        self.rnn = RNNBlock(
            input_channels, hidden_channels, layers, 
            time_dim=time_dim, model=model,
            batch_first=True, dropout=dropout, bidirectional=bidirectional
        )

        # time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        hidden_channels = utils.times_two_if_bidirectional(hidden_channels, bidirectional)
        self.fc = nn.Linear(hidden_channels, output_channels)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = utils.default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 2)
        
        t = self.time_emb(time)
        x = self.rnn(x, t)
        x = self.fc(x)
        return x.permute(0, 2, 1)


class EncoderRNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, layers, batch_first, dropout, bidirectional, time_dim, model):
        super().__init__()
        if model == "lstm":
            self.rnn = nn.LSTM(input_channels, output_channels, layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif model == "gru":
            self.rnn = nn.GRU(input_channels, output_channels, layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError(f"Invalid model name \"{model}\"! It can be only \"lstm\" or \"gru\"!")
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, output_channels * 2),
        )
        self.ht_norm = RMSNorm(output_channels)
        self.ct_norm = RMSNorm(output_channels)
    
    def forward(self, x, time_embed):
        """input is (batch_size, channels, timesteps)"""
        x = x.permute(0, 2, 1)
        time_embed = self.time_mlp(time_embed)
        time_embed = rearrange(time_embed, 'b c -> b c 1')
        time_tuple = time_embed.chunk(2, dim=1)
        
        outputs, (ht, ct) = self.rnn(x)
        ht, ct = self.ht_norm(ht.permute(0,2,1)).permute(0,2,1), self.ct_norm(ct.permute(0,2,1)).permute(0,2,1)
        
        scaling, shifts = time_tuple
        scaling, shifts = scaling.permute(2,0,1), shifts.permute(2,0,1)
        ht = ht * (scaling + 1) + shifts
        ct = ct * (scaling + 1) + shifts
        return (ht, ct)
    

class DecoderRNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, layers, batch_first, dropout, bidirectional, model):
        super().__init__()
        if model == "lstm":
            self.rnn = nn.LSTM(input_channels, output_channels, layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif model == "gru":
            self.rnn = nn.GRU(input_channels, output_channels, layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError(f"Invalid model name \"{model}\"! It can be only \"lstm\" or \"gru\"!")
    
    def forward(self, input, hidden):
        x = self.rnn(input, hidden)
        return x[0]         # return output only, not hidden state


class EncoderDecoderRNN(nn.Module):
    def __init__(
        self, 
        input_channels, 
        hidden_channels, 
        output_channels,
        layers, 
        model = "lstm",
        dropout = 0, 
        bidirectional  = True, 
        self_condition = False,
        embed_dim = 64,
        time_dim = 256,
        ):
        super().__init__()
        assert model in ["lstm", "gru"], "Invalid model name! It can be only \"lstm\" or \"gru\"!"
        
        self.self_condition = self_condition
        
        self.encode_rnn = EncoderRNNBlock(
            input_channels, hidden_channels, layers, 
            time_dim=time_dim, model=model,
            batch_first=True, dropout=dropout, bidirectional=bidirectional
        )
        self.decode_rnn = DecoderRNNBlock(
            input_channels, hidden_channels, layers,
            batch_first=True, dropout=dropout, bidirectional=bidirectional, model=model,
        )

        # time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        hidden_channels = utils.times_two_if_bidirectional(hidden_channels, bidirectional)
        self.fc = nn.Linear(hidden_channels, output_channels)
    
    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = utils.default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 2)
        
        t = self.time_emb(time)
        hidden = self.encode_rnn(x, t)
        noise = torch.randn_like(x).permute(0,2,1)
        x = self.decode_rnn(noise, hidden)
        x = self.fc(x)
        return x.permute(0,2,1)


        
        
        
        
    
    
        
        
    