import numpy as np
import torch
import sys
import os
import pathlib
import urllib.request
# import torchaudio
import tarfile
import zipfile
import math
import csv
import sktime.utils.load_data
import collections as co
import torchcde

def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # numerator = data-np.min(data)
    # denominator = np.max(data) - np.min(data)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def to_tensor(data):
    return torch.from_numpy(data).float()


def batch_generator(dataset, batch_size):
    # import pdb;pdb.set_trace()
    dataset_size = len(dataset)
    idx = torch.randperm(dataset_size)
    batch_idx = idx[:batch_size]
    batch = torch.stack([to_tensor(dataset[i]) for i in batch_idx])
    return batch

class SineDataset(torch.utils.data.Dataset):
    def __init__(self, no, seq_len,dim,data_name,missing_rate):
        base_loc = here / 'datasets'
        loc = here / 'datasets'/(data_name+str(missing_rate))
        if os.path.exists(loc):
            tensors = load_data(loc)
            self.train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
            self.data = tensors['data']
            self.original_sample = tensors['original_data']
            self.original_sample = np.array(self.original_sample)
            self.data = np.array(self.data)
            self.size = len(self.data)
        else:
            if not os.path.exists(base_loc):
                os.mkdir(base_loc)    
            if not os.path.exists(loc):
                os.mkdir(loc)
            self.data = list()
            self.original_sample = list()
            generator = torch.Generator().manual_seed(56789)
            for i in range(no):
                tmp = list()
                for k in range(dim):
                    freq = np.random.uniform(0, 0.1)
                    phase = np.random.uniform(0, 0.1)
                    tmp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
                    tmp.append(tmp_data)
                tmp = np.transpose(np.asarray(tmp))
                tmp = (tmp + 1) * 0.5
                self.original_sample.append(tmp.copy())
                removed_points = torch.randperm(tmp.shape[0], generator=generator)[:int(tmp.shape[0] * missing_rate)].sort().values
                tmp[removed_points] = float('nan')
                idx = np.array(range(seq_len)).reshape(-1,1)
                tmp = np.concatenate((tmp,idx),axis=1)
                self.data.append(tmp)
            self.data = np.array(self.data)
            self.original_sample = np.array(self.original_sample)
            norm_data_tensor = torch.Tensor(self.data[:,:,:-1]).float().cuda()
            time = torch.FloatTensor(list(range(norm_data_tensor.size(1)))).cuda()
            self.last = torch.Tensor(self.data[:,:,-1][:,-1]).float()
            self.train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time, norm_data_tensor)
            self.original_sample = torch.tensor(self.original_sample)
            self.data = torch.tensor(self.data)
            save_data(loc,data=self.data,
                    original_data = self.original_sample,
                    train_a=self.train_coeffs[0], 
                    train_b=self.train_coeffs[1], 
                    train_c=self.train_coeffs[2],
                    train_d=self.train_coeffs[3],
                    )
            self.original_sample = np.array(self.original_sample)
            self.data = np.array(self.data)
            self.size = len(self.data)
    def __getitem__(self, batch_size):
        dataset_size = len(self.data)
        idx = torch.randperm(dataset_size)
        batch_idx = idx[:batch_size]
        original_batch = torch.stack([to_tensor(self.original_sample[i]) for i in batch_idx])
        batch = torch.stack([to_tensor(self.data[i]) for i in batch_idx])
        a, b, c, d = self.train_coeffs
        batch_a = torch.stack([a[i] for i in batch_idx])
        batch_b = torch.stack([b[i] for i in batch_idx])
        batch_c = torch.stack([c[i] for i in batch_idx])
        batch_d = torch.stack([d[i] for i in batch_idx])
        batch_coeff = (batch_a, batch_b, batch_c, batch_d)
        self.sample = {'data': batch , 'inter': batch_coeff, 'original_data':original_batch}
        return self.sample
    def __len__(self):
        return len(self.data)

class SineDataset_t(torch.utils.data.Dataset):
    def __init__(self, no, seq_len, dim, missing_rate = 0.0):
        self.data = list()
        generator = torch.Generator().manual_seed(56789)
        for i in range(no):
            tmp = list()
            for k in range(dim):
                freq = np.random.uniform(0, 0.1)
                phase = np.random.uniform(0, 0.1)

                tmp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
                tmp.append(tmp_data)
            tmp = np.transpose(np.asarray(tmp))
            tmp = (tmp + 1) * 0.5
            removed_points = torch.randperm(tmp.shape[0], generator=generator)[:int(tmp.shape[0] * missing_rate)].sort().values
            tmp[removed_points] = float('nan')
            idx = np.array(range(seq_len)).reshape(-1,1)
            tmp = np.concatenate((tmp,idx),axis=1)
            self.data.append(tmp)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

here = pathlib.Path(__file__).resolve().parent


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(f"{dir}/{filename}")
            tensors[tensor_name] = tensor_value
    return tensors

def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, f"{dir}/{tensor_name}.pt")

import pathlib
here = pathlib.Path(__file__).resolve().parent
    
def smart_load(load_path):
    if load_path.split(".")[-1] == "pt":
        return torch.load(load_path)
    elif load_path.split(".")[-1] == "npy":
        return np.load(load_path)
    elif load_path.split(".")[-1] == "csv":
        return pd.read_csv(load_path)
    else:
        raise ValueError("Unsupported file type: %s" % load_path.split(".")[0])

def smart_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    else:
        raise ValueError("Unsupported type: %s" % type(data))

def normalize_data(data: torch.Tensor, indicator_cols: list=None, eps=0):
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
    data = (data - min + eps) / (max - min + 2*eps + 1e-9)
    assert np.all((data[~np.isnan(data)] >= 0) & (data[~np.isnan(data)] <= 1)), 'failed to normalize data'
    
    if indicator_cols is not None:
        data[:, indicator_cols, :] = indicators         # if passed, do not normalize these columns
            
    return torch.tensor(data).float(), torch.tensor(min).float(), torch.tensor(max).float()


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_name, here, irregular: bool):
        loc = f"{here}/gt_gan/{data_name}"
        self.irregular = irregular
        if os.path.exists(f"{loc}/data.pt") and os.path.exists(f"{loc}/original_data.pt") and os.path.exists(f"{loc}/train_coeffs.pt") and os.path.exists(f"{loc}/min.pt") and os.path.exists(f"{loc}/max.pt"):
            tensors = load_data(loc)
            self.train_coeffs = tensors["train_coeffs"]
            self.samples = tensors['data']
            self.original_sample = tensors['original_data']
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)
            self.min = tensors['min']
            self.max = tensors['max']
        else:
            print("Didn't find saved data, will pre-process and save instead (may take very long time)...")
            if not os.path.exists(loc):
                os.mkdir(loc)
            data = smart_load(data_path)
            
            if irregular:
                time_dim = data[:, -1, :]
                data = data[:, :-1, :]              # assumes time is at last dimension for irregular data
                
            self.original_sample = data.permute(0,2,1)
            norm_data, self.min, self.max = normalize_data(data)
            
            if irregular:           # if irregular, just normalize features, and concat time back
                norm_data = torch.cat((norm_data, time_dim.unsqueeze(1)), dim=1)  
                  
            self.samples = norm_data.permute(0,2,1).float()
            
            if irregular:
                norm_data_tensor = self.samples[:,:,:-1].float().cuda()
            else:
                norm_data_tensor = self.samples.float().cuda()
                
            time = torch.FloatTensor(list(range(self.samples.size(1))))
            
            if not irregular:           # if it's regular data, append time as last dimension
                time_concat = time.unsqueeze(0).unsqueeze(2)
                time_concat = time_concat.expand(self.samples.shape[0], -1, -1)                
                self.samples = torch.cat((self.samples, time_concat), dim=2)
            # NOTE: assume irregular data has time as last dimension for every data point
            
            time = time.cuda()                
            self.train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(t=time, x=norm_data_tensor)
            
            save_data(
                loc,
                data = self.samples,
                original_data = self.original_sample,
                train_coeffs = self.train_coeffs,
                min = self.min,
                max = self.max,
            )
            print(f"Data saved to \"{loc}\"")
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)

    def __getitem__(self, batch_size):
        # import pdb;pdb.set_trace()
        dataset_size = len(self.samples)
        idx = torch.randperm(dataset_size)
        batch_idx = idx[:batch_size]
        original_batch = torch.stack([to_tensor(self.original_sample[i]) for i in batch_idx])
        batch = torch.stack([to_tensor(self.samples[i]) for i in batch_idx])

        # batch _idx -> batch 만큼 가져고 
        a = self.train_coeffs
        batch_a = torch.stack([a[i] for i in batch_idx])
        
        self.sample = {'data': batch , 'inter': batch_a, 'original_data':original_batch}

        return self.sample # self.samples[index]

    def __len__(self):
        return len(self.samples)
    
    @property
    def dim(self):
        if self.irregular:          # if irregular, it's 1 minus since original sample has time dimension
            return self.original_sample.shape[2] - 1
        else:                       # if regular, just use original sample since doesn't have time dimension
            return self.original_sample.shape[2]
    
    @property
    def seq_len(self):
        return self.samples.shape[1]

class TimeDataset_regular(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len):
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)
        total_length = len(data)
        data = data[::-1]
        
        self.min_val = np.min(data, 0)
        self.max_val = np.max(data, 0) - np.min(data, 0)
        
        norm_data = normalize(data)
        total_length = len(norm_data)
        idx = np.array(range(total_length)).reshape(-1,1)
        norm_data = np.concatenate((norm_data,idx),axis=1)#맨 뒤에 관측시간에 대한 정보 저장

        seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len]
            seq_data.append(x)
        self.samples = []
        idx = torch.randperm(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)