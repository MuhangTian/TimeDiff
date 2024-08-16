import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from models.halo.model import HALOModel
from models.halo.config import HALOConfig
import argparse

COLUMNS_DICT = {
    "mimiciv": {"numerical": [0, 2, 4, 6, 8], "categorical": [1, 3, 5, 7, 9, 10], "categorical_num_classes": [2, 2, 2, 2, 2, 2]},
    "mimiciii": {"numerical": [0, 2, 4, 6, 8, 10, 12], "categorical": [1, 3, 5, 7, 9, 11, 13, 14], "categorical_num_classes": [2, 2, 2, 2, 2, 2, 2, 2]},
    "eicu": {"numerical": [0, 2, 4, 6], "categorical": [1, 3, 5, 7, 8], "categorical_num_classes": [2, 2, 2, 2, 2]},
    "hirid": {"numerical": [0, 1, 2, 3, 4, 5, 6], "categorical": [7], "categorical_num_classes": [2]},
}

def parse_arguments():
    prs = argparse.ArgumentParser(
        prog='halo_generate.py',
        description='Sample from trained HALO model',
        epilog='Copyright (C) 2023',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    prs.add_argument("--seed", type=int, default=4)
    prs.add_argument("--dataset", type=str, default="mimiciii")
    prs.add_argument("--data", type=str, default="data/mimic-iii/train.pt", help="Path to original data file")
    prs.add_argument("--dataset_path", type=str, default="data/halo/mimiciii_halo_processed.pt", help="Path to processed data file")
    prs.add_argument("--mask_path", type=str, default="data/halo/mimiciii_halo_mask.pt", help="Path to processed masks")
    prs.add_argument("--save_path", type=str, default="results/models/mimiciii_halo.pt", help="Path to saved trained models")
    prs.add_argument("--sample_path", type=str, default="results/samples/mimiciii_halo_samples.pt", help="Path to saved samples")
    prs.add_argument("--n_layer", type=int, default=12)
    prs.add_argument("--n_head", type=int, default=18)
    prs.add_argument("--n_embd", type=int, default=1440)
    prs.add_argument("--n_positions", type=int, default=150)
    prs.add_argument("--n_ctx", type=int, default=150)
    return prs.parse_args()

class HALOSampler:
    def __init__(self, args, config, model, n_samples, device, seq_length, data, bsz_sample=200):
        assert n_samples % bsz_sample == 0, "n_samples must be divisible by bsz_sample"

        self.args = args
        self.data = data        # original data (unprocessed)
        self.seq_length = seq_length
        self.model = model
        self.config = config
        self.n_samples = n_samples
        self.bsz_sample = bsz_sample
        self.sync_dataset = []
        self.stoken = np.zeros(config.total_vocab_size)
        self.stoken[-2] = 1      # start token
        self.device = device
        self.missing_indices = [-1 + 101*(k+1) for k in range(self.data.shape[1])]
        self.find_bins_per_feature()

    @property
    def is_stock_energy(self):
        return self.args.dataset in ["stocks", "energy"]
    
    @property
    def is_eicu_mimic_hirid(self):
        return self.args.dataset in ["eicu", "mimiciv", "mimiciii", "hirid"]
    
    def j_idx(self, idx):
        return idx - 2 if self.is_eicu_mimic_hirid else idx - 1

    def f_idx(self, idx):
        return idx - 3 if self.is_eicu_mimic_hirid else idx - 2
    
    def find_bins_per_feature(self):
        """
        Find the bins for each feature using quantiles
        """
        self.bins = []
        for i in range(self.data.shape[1]):
            nonnan_data = self.data[:, i, :][~np.isnan(self.data[:, i, :])]
            quantiles = np.quantile(nonnan_data, np.linspace(0, 1, 101))
            self.bins.append([(quantiles[i], quantiles[i+1]) for i in range(100)])
    
    def sample_sequence(self, model, length, context, batch_size, device='cuda', sample=True):
        empty = torch.zeros((1, 1, self.config.total_vocab_size), device=self.device, dtype=torch.float32).repeat(batch_size, 1, 1)
        context = torch.tensor(context, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        prev = context.unsqueeze(1)
        context = None

        with torch.no_grad():
            for _ in range(length-1):
                prev = model.sample(torch.cat((prev,empty), dim=1), sample)
                if torch.sum(torch.sum(prev[:, :, -1], dim=1).bool().int(), dim=0).item() == batch_size:
                    break

        ehr = prev.cpu().detach().numpy()
        prev = None
        empty = None
        return ehr

    def reverse_discretize(self, input_data: np.ndarray) -> np.ndarray:
        """
        Reverse the discretization process, samples uniformly in the given bin range [a, b]
        """
        data = np.zeros((len(input_data), self.data.shape[1], self.data.shape[2]))    # n_sample, n_features, seq_length
        if self.is_eicu_mimic_hirid:
            iterator = range(2, input_data.shape[1]-1)
        else:
            iterator = range(1, input_data.shape[1]-1)

        for i in range(len(input_data)):
            
            if self.is_eicu_mimic_hirid:
                mortality_flag = input_data[i, 1, -3]
                data[i, -1, :] = np.ones(self.data.shape[2]) * mortality_flag
            
            for j in iterator:
                one_indices = np.where(input_data[i, j, :] == 1)[0]

                if len(one_indices) == 0:
                    continue
                else:
                    for idx in one_indices:
                        feature_idx, status = self.find_feature_idx_given_idx(idx)
                        if status == "missing":
                            data[i, feature_idx, self.j_idx(j)] = np.nan
                        else:
                            a, b = self.bins[feature_idx][idx % 101]
                            data[i, feature_idx, self.j_idx(j)] = np.random.uniform(a, b)
        
        return data
    
    def find_feature_idx_given_idx(self, idx):
        """
        Find the feature index given the index of the one-hot encoded vector
        """
        if idx in self.missing_indices:
            return self.missing_indices.index(idx), "missing"
        else:
            return self.f_idx(idx) // 101, "not missing"
    
    def sample(self):
        for i in tqdm(range(0, self.n_samples, self.bsz_sample)):
            batch_sync = self.sample_sequence(self.model, self.seq_length, self.stoken, batch_size=self.bsz_sample, device=self.device, sample=True)
            batch_sync = self.reverse_discretize(batch_sync)
            self.sync_dataset.append(batch_sync)
        self.sync_dataset = np.concatenate(self.sync_dataset, axis=0)

if __name__ == "__main__":
    args = parse_arguments()
    processed_data = torch.load(args.dataset_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    local_rank = -1
    fp16 = False
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.dataset in ["stocks", "energy"]:
        data = torch.load(args.data).numpy()
    elif args.dataset in ["mimiciv", "mimiciii", "eicu", "hirid"]:
        features = [*COLUMNS_DICT[args.dataset]["numerical"], -1]
        data = torch.load(args.data)[:, features, :].numpy()
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported")
    
    config = HALOConfig(
        total_vocab_size = processed_data.shape[2],
        n_layer = args.n_layer,
        n_head = args.n_head,
        n_embd = args.n_embd,
        n_positions = args.n_positions,
        n_ctx = args.n_ctx,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    model = HALOModel(config).to(device)
    checkpoint = torch.load(args.save_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])

    sampler = HALOSampler(
        args, config, model, 
        device=device, seq_length=processed_data.shape[1], 
        data=data,
        n_samples=20000, bsz_sample=100,
    )
    sampler.sample()
    torch.save(torch.tensor(sampler.sync_dataset), args.sample_path)
    print(f"Sampling complete! Samples saved to {args.sample_path}.")





