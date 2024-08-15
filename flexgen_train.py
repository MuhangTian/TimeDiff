from models.FlexGen.cvae import VariationalAutoencoder, vae_loss_fn
from models.FlexGen.ddpm import DDPM, ContextUnet
from models.FlexGen.flexgen import flexgen
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import argparse
from helpers.utils import is_file_on_disk, is_file_not_on_disk
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

COLUMNS_DICT = {
    "mimiciv": {"numerical": [0, 2, 4, 6, 8], "categorical": [1, 3, 5, 7, 9, 10], "categorical_num_classes": [2, 2, 2, 2, 2, 2]},
    "mimiciii": {"numerical": [0, 2, 4, 6, 8, 10, 12], "categorical": [1, 3, 5, 7, 9, 11, 13, 14], "categorical_num_classes": [2, 2, 2, 2, 2, 2, 2, 2]},
    "eicu": {"numerical": [0, 2, 4, 6], "categorical": [1, 3, 5, 7, 8], "categorical_num_classes": [2, 2, 2, 2, 2]},
    "hirid": {"numerical": [0, 1, 2, 3, 4, 5, 6], "categorical": [7], "categorical_num_classes": [2]},
}

def parse_arguments():
    prs = argparse.ArgumentParser(prog="flexgen_train.py", description="Train FlexGen")
    prs.add_argument("--data_name", type=str, required=True, help="name of the folder to save data in datasets")
    prs.add_argument("--load_path", type=is_file_on_disk, required=True, help="path to load data")
    prs.add_argument("--model_path", type=is_file_not_on_disk, required=True, help="path to save model")
    prs.add_argument("--sync_path", type=is_file_not_on_disk, required=True, help="path to save synthetic data")
    # prs.add_argument("--cut_length", type=int, default=272, help="how much length of the entire sequence to use")
    return prs.parse_args()

def train_vae_tmp(net, dataloader,  epochs=30):
    optim = torch.optim.Adam(net.parameters())
    for i in range(epochs):
        running_loss = 0

        for batch, y in dataloader:

            if y is not None:
                y = y.to(device)

            batch = batch.to(device)
            optim.zero_grad()
            x, mu, logvar, z = net(batch, y)
            loss = vae_loss_fn(batch, x, mu, logvar, numeric=True)
            loss.backward()
            optim.step()
            running_loss += loss.item()

        print(running_loss/512)
        # evaluate(validation_losses, net, test_dataloader, vae=True, title=f'VAE - Epoch {i}')
    # torch.save(net, 'saved_models/vae_tmp.pt')

class MIMICDATASET(Dataset):
    def __init__(self, x_t,x_s, ys, yt, train=None, transform=None):
        # Transform
        self.transform = transform
        self.train = train
        self.xt = x_t
        self.xs = x_s
        self.ys = ys
        self.yt = yt
        m = x_s.shape[0]
        num_miss = 500
        self.miss_t = np.random.permutation(np.arange(m))[0: num_miss]
        self.miss_s = np.random.permutation(np.arange(m))[0: num_miss]

    def return_data(self):
        return self.xt, self.xs, self.ys, self.yt

    def __len__(self):
        return len(self.xt)


    def __getitem__(self, idx):
        if idx in self.miss_t:
            sample = 0
        else:
            sample = self.xt[idx]
        if idx in self.miss_s:
            stat = 0
        else:
            stat = self.xs[idx]
        sample_ys = self.ys[idx]
        sample_yt = self.yt[idx]
        return sample, stat, sample_ys, sample_yt

class TimeSeriesDataset(Dataset):
    def __init__(self, data_name, categorical_cols, cut_length, load_path=None):
        path = f"data/datasets/{data_name}"
        self.categorical_cols = categorical_cols
        self.data_name = data_name
        self.use_cond = True if self.data_name in ["eicu", "mimiciv", "mimiciii", "hirid"] else False

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
        if self.use_cond:
            y = torch.round(torch.mean(self.data[idx, -1, :])).clone()        # either 1 or 0 for mortality indicator, this step is due to the preprocessing method used in our study
            x = self.data[idx, :-1].clone()
            return (x, y)
        else:
            return self.data[idx].clone(), torch.zeros(1).clone()        # if doesn't have flag, cond is 0
    
    @property
    def channels(self):
        return self.data.shape[1]
    
    @property
    def seq_length(self):
        return self.data.shape[2]

def replace_nan_with_mean(data: torch.Tensor):
    mean = data.nanmean(dim=(0, 2))[None, :, None]
    data = torch.where(torch.isnan(data), mean, data)
    assert torch.all(~torch.isnan(data)), 'failed to replace NaNs!'
    return data

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


if __name__ == '__main__':
    args = parse_arguments()
    dataset = TimeSeriesDataset(
        categorical_cols = COLUMNS_DICT[args.data_name]["categorical"] if args.data_name in ["eicu", "mimiciv", "mimiciii", "hirid"] else None,             # indicate columns that are not time series values (missing indicators)
        data_name = args.data_name, 
        load_path = args.load_path,
        cut_length = 276, # doesn't do anything, longest sequence length in all the datasets is 276
    )

    batch_size =  512
    device = torch.device("cuda")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, \
                              num_workers=1, drop_last=False)

    feature_dim = dataset.channels

    vae = VariationalAutoencoder(feature_dim, cond=True).to(device)
    train_vae_tmp(vae, train_loader, epochs=70)
    vae.eval()

    n_epoch = 1
    n_T = 50 
    device = "cuda"
    n_classes = 2
    n_feat = 256  
    lrate = 1e-4
    save_model = True
    w = 0.1
    
    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=2),
        betas=(1e-4, 0.02),
        n_T=n_T, 
        device=device, 
        drop_prob=0.1,
    )
    ddpm.to(device)
    trainer = flexgen(
        tvae=vae, ldm=ddpm, 
        trainloader=train_loader, epochs=n_epoch,
        model_path=args.model_path,
        synthetic_path=args.sync_path,

    )
    trainer.generate(5000, 0)