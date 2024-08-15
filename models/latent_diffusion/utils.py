import math
import os

import numpy as np
import torch
import torchcde
from torch.utils.data import DataLoader, Dataset


class TSDataset(Dataset):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()

class IrregularTSDataset(Dataset):
    def __init__(self, data_name, indicator_cols, load_path=None):
        path = f"data/datasets/{data_name}"
        if os.path.exists(f"{path}/coeffs.pt") and os.path.exists(f"{path}/data.pt") and os.path.exists(f"{path}/min.pt") and os.path.exists(f"{path}/max.pt"):
            self.coeffs = torch.load(f"{path}/coeffs.pt")[:,:288,:]
            self.data = torch.load(f"{path}/data.pt")[:,:288,:]                     # to be compatible with diffusion model (must be divisible by 8)
            self.min = torch.load(f"{path}/min.pt")
            self.max = torch.load(f"{path}/max.pt")
        else:
            print("data not found in datasets directory, will load raw data and save to dataset folder")
            assert load_path is not None, "load_path must be specified if data does not exist in dataset folder"
            assert load_path.endswith(".pt"), "load_path must be a .pt file"
            assert os.path.exists(load_path), "load_path does not exist"
            
            if not os.path.exists(path):
                os.mkdir(path)
            data = torch.load(load_path)
            data, self.min, self.max = normalize_data(data, indicator_cols)                 # normalize along the feature dimension
            data = data.permute(0, 2, 1)           # original data is (num_samples, num_channels, num_timesteps), so transpose here to fit Neural ODE
            # num, length, channels = data.shape
            # indices = torch.arange(int(num * length))                   # used in decoder to reconstruct the original data
            # indices = indices.reshape(num, length)
            # data = torch.cat((data, indices.unsqueeze(2)), dim=2)       # add indices as an additional channel to the data
            self.data = data[:,:288,:]
            
            # train_data = data[:,:,:-1]                         # do not pass indices, time is first dimension here
            print("Computing cubic spline coefficients...")
            self.coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(self.data)[:,:288,:]
            
            torch.save(self.coeffs, f"{path}/coeffs.pt")
            torch.save(self.data, f"{path}/data.pt")
            torch.save(self.min, f"{path}/min.pt")
            torch.save(self.max, f"{path}/max.pt")
            
            print(f"data saved to {path} folder")
        
        print(f"Data dimension: {self.data.shape}, coeff dimension: {self.coeffs.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].clone(), self.coeffs[idx].clone()
    
    @property
    def channels(self):
        return self.data.shape[2] - 1               # -1 because exclude time (time is used to construct coeffs already)
    
    @property
    def seq_length(self):
        return self.data.shape[1]

def reverse_to_nonan_indicator(cum_nonan_indicator):        # Reverse cumulative sum using PyTorch
    if isinstance(cum_nonan_indicator, torch.Tensor):
        nonan_indicator = torch.cat([cum_nonan_indicator[:1], cum_nonan_indicator[1:] - cum_nonan_indicator[:-1]])
    elif isinstance(cum_nonan_indicator, np.ndarray):
        nonan_indicator = np.concatenate([cum_nonan_indicator[:1], cum_nonan_indicator[1:] - cum_nonan_indicator[:-1]])
    else:
        raise ValueError()
    return nonan_indicator

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def normalize_to_neg_one_to_one(sample):
    """
    Assumes the input is already in the range of [0,1]
    """
    return sample * 2 - 1

def unnormalize_to_zero_to_one(t):
    """
    Assumes the input is in range [-1, 1]
    """
    return (t + 1) * 0.5

def identity(t, *args, **kwargs):
    return t

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def cycle(dl):
    while True:
        for data in dl:
            yield data

def normalize_data(data: torch.Tensor, indicator_cols: list, eps=0):
    """
    Parameters
    ----------
    data : torch.Tensor
        assumes in dimension (num_samples, num_channels, num_timesteps)
    indicator_cols : list
        columns for missingness indicators, if provided, do not noramlize these, if None, normalize all except time
    eps : int, optional
        used when want to let values to be (0,1), by default 0

    Returns
    -------
    tuple
        normalized data, min, max
    """
    data = data.numpy()
    time = data[:, 0, :].copy()            # assumes time is in first entry in second dimension
    
    if indicator_cols is not None:
        indicators = data[:, indicator_cols, :].copy()
        assert not np.any(np.isnan(indicators)), 'indicator columns contain NaNs'
    
    assert not np.any(np.isnan(time)), 'time contains NaNs'
    
    min, max = np.nanmin(data, axis=(0, 2)), np.nanmax(data, axis=(0, 2))
    min, max = min[None, :, None], max[None, :, None]
    data = (data - min + eps) / (max - min + 2*eps)
    assert np.all((data[~np.isnan(data)] >= 0) & (data[~np.isnan(data)] <= 1)), 'failed to normalize data'
    data[:, 0, :] = time                            # do not normalize time, needed information for neural ODE
    
    if indicator_cols is not None:
        data[:, indicator_cols, :] = indicators         # same idea here
            
    return torch.tensor(data).float(), torch.tensor(min)[:,1:,:].float(), torch.tensor(max)[:,1:,:].float()             # drop stats for time for min and max

def reverse_normalize(data, min, max) -> np.ndarray:
    return (data * (max - min) + min)          # to turn back into real scale          