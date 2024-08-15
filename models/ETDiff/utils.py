import math
import os

import numpy as np
import torch
import torchcde
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data_name, categorical_cols, cut_length, load_path=None):
        path = f"data/datasets/{data_name}"
        self.categorical_cols = categorical_cols
        if os.path.exists(f"{path}/data.pt") and os.path.exists(f"{path}/min.pt") and os.path.exists(f"{path}/max.pt"):
            self.data = torch.load(f"{path}/data.pt")[:,:,:cut_length]                     # to be compatible with diffusion model (must be divisible by 8)
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
            data = replace_nan_with_mean(data)
            data, self.min, self.max = normalize_data(data, categorical_cols)                 # normalize along the feature dimension
            self.data = data[:,:,:cut_length]
            
            torch.save(self.data, f"{path}/data.pt")
            torch.save(self.min, f"{path}/min.pt")
            torch.save(self.max, f"{path}/max.pt")
            
            print(f"data saved to {path} folder")
        
        print(f"Data dimension: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].clone()
    
    @property
    def channels(self):
        return self.data.shape[1]
    
    @property
    def seq_length(self):
        return self.data.shape[2]

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

def check_is_integer(var):
    assert isinstance(var, int), "must be integer!"

def times_two_if_bidirectional(integer, bidirectional):
    return integer * 2 if bidirectional else integer

def normalize_data(data: torch.Tensor, categorical_cols: list, eps=0):
    """
    Parameters
    ----------
    data : torch.Tensor
        assumes in dimension (num_samples, num_channels, seq_length)
    categorical_cols : list
        columns for missingness indicators, if provided, do not noramlize these, if None, normalize all
    eps : int, optional
        used when want to let values to be (0,1), by default 0

    Returns
    -------
    tuple
        normalized data, min, max
    """
    data = data.numpy()
    
    if categorical_cols is not None:
        indicators = data[:, categorical_cols, :].copy()
        assert not np.any(np.isnan(indicators)), 'indicator columns contain NaNs'
    
    min, max = np.nanmin(data, axis=(0, 2)), np.nanmax(data, axis=(0, 2))
    min, max = min[None, :, None], max[None, :, None]
    data = (data - min + eps) / (max - min + 2*eps)
    assert np.all((data[~np.isnan(data)] >= 0) & (data[~np.isnan(data)] <= 1)), 'failed to normalize data'
    
    if categorical_cols is not None:
        data[:, categorical_cols, :] = indicators         # if passed, do not normalize these columns
            
    return torch.tensor(data).float(), torch.tensor(min).float(), torch.tensor(max).float()

def reverse_normalize(data, min, max) -> np.ndarray:
    return (data * (max - min) + min)          # to turn back into real scale

def replace_nan_with_mean(data: torch.Tensor):
    mean = data.nanmean(dim=(0, 2))[None, :, None]
    data = torch.where(torch.isnan(data), mean, data)
    assert torch.all(~torch.isnan(data)), 'failed to replace NaNs!'
    return data

def replace_nan_with(data: torch.Tensor, value):
    data = torch.where(torch.isnan(data), torch.tensor(value), data)
    assert torch.all(~torch.isnan(data)), 'failed to replace NaNs!'
    return data
    