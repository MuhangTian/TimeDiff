import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(format=("[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.has_mps:
        return torch.device('mps')
    else:
        return torch.device('cpu')

def normalize_data(data: torch.Tensor, eps=0):
    min, max = data.amin(dim=(0, 2))[None, :, None], data.amax(dim=(0, 2))[None, :, None]
    data = (data - min + eps) / (max - min + 2*eps)
    return data.float(), min.float(), max.float()

def standardize_data(data: torch.Tensor):
    mean, std = data.mean(dim=(0, 2))[None, :, None], data.std(dim=(0, 2))[None, :, None]
    data = (data - mean) / std
    return data.float(), mean.float(), std.float()

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index) -> torch.Tensor:
        return self.X[index, :, :], self.y[index]

class FCN(nn.Module):
    """
    Implements vanilla fully convolutional network for time series classification. 
    This is intended for univariate time series classification problem only.
    
    References
    ----------
        https://arxiv.org/pdf/1611.06455.pdf
    """
    def __init__(self, in_channels, out):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=8, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(128, out),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, X):
        Z = self.conv1(X)
        Z = self.conv2(Z)
        Z = self.conv3(Z)
        Z = self.final(Z)
        return Z

class RNN(nn.Module):
    """
    RNN for multivariate time series classification for benchmarking purposes.
    Note that we are aware that more sophisticated/better methods exist, but this straightforward 
    implementation serves our purpose for evaluating real vs synthetic samples without too much computational demand.
    """
    def __init__(self, in_channels, hidden_dim=64, model="lstm"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = model
        
        if self.model == 'lstm':
            self.lstm = nn.LSTM(in_channels, hidden_dim, bidirectional=True, batch_first=True)
        elif self.model == 'gru':
            self.gru = nn.GRU(in_channels, hidden_dim, bidirectional=True, batch_first=True)
        else:
            raise ValueError("Not available")
        
        self.final = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, X):
        X = X.permute(0, 2, 1)              # this is to make the data compatible with RNN, which expects (batch_size, timesteps, num_features)
        if self.model == "lstm":
            out, (ht, ct) = self.lstm(X)
        elif self.model == "gru":
            out, ht = self.gru(X)
        ht = ht.permute(1, 0, 2).reshape(X.shape[0], -1)
        Z = self.final(ht)
        return Z.squeeze()
        
class nnTimeSeriesClassifier:
    def __init__(self, model, X_train, y_train, bsz, check_point_interval=100, loss_func="cross entropy", normalize="standardize", wandb=None, train_num_steps=5000):
        assert isinstance(X_train, torch.Tensor), "X_train must be a torch.Tensor!"
        assert isinstance(y_train, torch.Tensor), "y_train must be a torch.Tensor!"
        assert len(X_train.shape) == 3, "X_train must be 3D tensor!"            # assumes data is store in the dimension of (batch_size, num_features, timesteps)
        assert len(y_train.shape) == 1, "y_train must be 1D tensor!"
        assert X_train.shape[0] == len(y_train), "X and y must have the same number of samples!"
        assert torch.equal(torch.unique(y_train), torch.tensor([0, 1])), "y must be binary and only (0,1)!"
        
        self.device = get_device()
        self.model = model(in_channels=X_train.shape[1]).to(self.device)         # assumes binary classification and channel is second dimension
        self.optimizer = optim.AdamW(self.model.parameters())            # default paramter is same as ones used in paper
        
        if loss_func == "cross entropy":
            self.loss_func = F.binary_cross_entropy             # use binary version since we are doing binary classification
        
        if normalize == "standardize":
            X_train, _, _ = standardize_data(X_train)
        elif normalize == "normalize":
            X_train, _, _ = normalize_data(X_train)

        self.X_train, self.y_train = X_train.float().to(self.device), y_train.float().to(self.device)
        self.train_loader = cycle(DataLoader(TimeSeriesDataset(self.X_train, self.y_train), batch_size=bsz, shuffle=True))
        self.wandb = wandb
        self.check_point_interval = check_point_interval
        self.train_num_steps = train_num_steps
    
    def train(self):
        step = 0
        # with tqdm(initial = step, total = self.train_num_steps, desc = "Training", position=0, leave=True) as pbar:
            
        while step < self.train_num_steps:
                
            X_batch, y_batch = next(self.train_loader)
            y_prob = self.model(X_batch)
            loss = self.loss_func(y_prob, y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            step += 1
    
    def change_device(self, device):
        self.device = device
        self.model = self.model.to(self.device)

    def predict_proba(self, X):
        if X.device != self.device:
            X = X.to(self.device)
        y_prob = self.model(X)
        return y_prob.detach().cpu().numpy()            # return probability of positive class