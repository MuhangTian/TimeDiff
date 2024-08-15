import pickle

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class TimeSeriesDataset(Dataset):
    # def __init__(self, data_path, bptt, device):
    def __init__(self, data_path):
        super().__init__()
        self.data = smart_load(data_path)
        self.data = replace_nan_with_mean(self.data)
        self.data, self.min, self.max = normalize_data(self.data, None)
        self.data = self.data.permute(0,2,1)            # NOTE: assumes data is (batch_size, channels, seq_len)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
    @property
    def channels(self):
        return self.data.size(2)

def smart_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    else:
        raise ValueError("Unsupported type: %s" % type(data))

def smart_load(load_path):
    if load_path.split(".")[-1] == "pt":
        return torch.load(load_path)
    elif load_path.split(".")[-1] == "npy":
        return np.load(load_path)
    elif load_path.split(".")[-1] == "csv":
        return pd.read_csv(load_path)
    else:
        raise ValueError("Unsupported file type: %s" % load_path.split(".")[0])

def normalize_data(data: torch.Tensor, indicator_cols: list, eps=0):
    """
    Parameters
    ----------
    data : torch.Tensor
        assumes in dimension (num_samples, num_channels, seq_length)
    indicator_cols : list
        columns for missingness indicators, if provided, do not noramlize these, if None, normalize all
    eps : int, optional
        used when want to let values to be (0,1), by default 0

    Returns
    -------
    tuple
        normalized data, min, max
    """
    
    if indicator_cols is not None:
        indicators = data[:, indicator_cols, :].copy()
        assert not np.any(np.isnan(indicators)), 'indicator columns contain NaNs'
    
    min, max = np.nanmin(data, axis=(0, 2)), np.nanmax(data, axis=(0, 2))
    min, max = min[None, :, None], max[None, :, None]
    data = (data - min + eps) / (max - min + 2*eps)
    assert np.all((data[~np.isnan(data)] >= 0) & (data[~np.isnan(data)] <= 1)), 'failed to normalize data'
    
    if indicator_cols is not None:
        data[:, indicator_cols, :] = indicators         # if passed, do not normalize these columns
            
    return torch.tensor(data).float(), torch.tensor(min).float(), torch.tensor(max).float()

def replace_nan_with_mean(data: np.ndarray):
    mean = np.nanmean(data, axis=(0, 2))[None, :, None]
    data = np.where(np.isnan(data), mean, data)
    assert not np.any(np.isnan(data)), 'failed to replace NaNs!'
    return data

