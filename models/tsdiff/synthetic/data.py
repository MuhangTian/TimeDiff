import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import os
from pathlib import Path

class DataModule(LightningDataModule):
    def __init__(self, batch_size: int, load_path, include_time: bool=True, test_batch_size: int = None):
        super().__init__()
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.include_time = include_time

        dataset = self._load_dataset(load_path)
        self.trainset, self.valset, self.testset = self._split_train_val_test(dataset)

    @property
    def dim(self):
        return self.trainset[0][0].shape[-1]

    @property
    def x_mean(self):
        return torch.cat([x[0] for x in self.trainset], 0).mean(0)

    @property
    def x_std(self):
        return torch.cat([x[0] for x in self.trainset], 0).std(0).clamp(1e-4)

    @property
    def t_max(self):
        return torch.cat([x[1] for x in self.trainset], 0).max().item()

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.test_batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.test_batch_size, shuffle=False)

    def _load_dataset(self, load_path):
        data = torch.load(load_path)            # assumes the dimension is (bsz, num_channels, seq_len)
        data = replace_nan_with_mean(data)
        data, self.min, self.max = normalize_data(data, None)
        data = data.permute(0,2,1)
        # TODO: get time
        if self.include_time:
            raise NotImplementedError('time is not implemented yet')
        else:
            times = torch.FloatTensor(list(range(data.size(1))))
            times = times.unsqueeze(0).unsqueeze(2)
            times = times.expand(data.shape[0], -1, -1)
            
        dataset = TensorDataset(data, times)
        return dataset

    def _split_train_val_test(self, dataset):
        train_len, val_len = int(0.6 * len(dataset)), int(0.2 * len(dataset))
        return random_split(dataset, lengths=[train_len, val_len, len(dataset) - train_len - val_len])


def replace_nan_with_mean(data: torch.Tensor):
    mean = data.nanmean(dim=(0, 2))[None, :, None]
    data = torch.where(torch.isnan(data), mean, data)
    assert torch.all(~torch.isnan(data)), 'failed to replace NaNs!'
    return data

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
    data = data.numpy()
    
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