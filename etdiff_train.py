import argparse
import time
# import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

import wandb
from helpers.utils import (create_id, is_file_not_on_disk, is_file_on_disk,
                           is_positive_float, is_positive_integer,
                           seed_everything, exists, reverse_normalize)
from models.ETDiff.gaussian_diffusion import GaussianDiffusion
from models.ETDiff.mixed_diffusion import MixedDiffusion
from models.ETDiff.et_diff import ETDiff
from models.ETDiff.blocks import NeuralCDE, RNN, EncoderDecoderRNN
from models.ETDiff.utils import TimeSeriesDataset
from models.denoising_diffusion_pytorch import Unet1D

# categorical_cols = [1, 3, 5, 7]       # for eICU
categorical_cols = [1, 3, 5, 7, 9, 11, 13]      # for MIMIC-IV

COLUMNS_DICT = {
    "mimiciv": {"numerical": [0, 2, 4, 6, 8], "categorical": [1, 3, 5, 7, 9, 10], "categorical_num_classes": [2, 2, 2, 2, 2, 2]},
    "mimiciii": {"numerical": [0, 2, 4, 6, 8, 10, 12], "categorical": [1, 3, 5, 7, 9, 11, 13, 14], "categorical_num_classes": [2, 2, 2, 2, 2, 2, 2, 2]},
    "eicu": {"numerical": [0, 2, 4, 6], "categorical": [1, 3, 5, 7, 8], "categorical_num_classes": [2, 2, 2, 2, 2]},
    "hirid": {"numerical": [0, 1, 2, 3, 4, 5, 6], "categorical": [7], "categorical_num_classes": [2]},
}


def parse_arguments():        
    prs = argparse.ArgumentParser(
        prog='etdiff_train.py',
        description='Train our model',
        epilog='Copyright (C) 2023',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    prs.add_argument(
        "--load_path", type=is_file_on_disk, default="data/eicu-extract/TRAIN_irregular_all_patients_0_1440_5.pt", # required=True,
        help="path to load data"
    )
    prs.add_argument(
        "--data_name", type=str, default="eicu",
        help="name of the folder to save data in datasets"
    )
    prs.add_argument(
        "--check_point_path", 
        # type=is_file_not_on_disk,     # DEBUG: remove this when done with debugging
        required=True,
        help="path to save model"
    )
    prs.add_argument(
        "--cut_length", type=int, default=272,
        help="how much length of the entire sequence to use, mainly for compatibility with tensor multiplications"
    )
    prs.add_argument(
        "--hidden", type=int, default=256,
        help="hidden dimension for RNN",
    )
    prs.add_argument(
        "--num_layers", type=int, default=3,
        help="number of layers for RNN",
    )
    prs.add_argument(
        "--model", type=str, default="lstm", choices=["lstm", "gru"],
        help="RNN model to use",
    )
    prs.add_argument(
        "--bidirectional", action="store_true", 
        help="whether to use bidirectional RNN",
    )
    prs.add_argument(
        "--batch_size", type=is_positive_integer, default=32, 
        help="batch size"
    )
    prs.add_argument(
        "--learning_rate", type=is_positive_float, default=8e-5, 
        help="learning rate"
    )
    prs.add_argument(
        "--embed_dim", type=int, default=64,
    )
    prs.add_argument(
        "--time_dim", type=int, default=256,
    )
    prs.add_argument(
        "--dropout", type=float, default=0,
        help="dropout rate"
    )
    prs.add_argument(
        "--num_steps", type=is_positive_integer, default=700000, 
        help="number of iterations to train diffusion model"
    )
    prs.add_argument(
        "--gradient_accumulate_every", type=is_positive_integer, default=2, 
        help="gradient accumulation steps"
    )
    prs.add_argument(
        "--ema_decay", type=is_positive_float, default=0.995, 
        help="ema decay rate"
    )
    prs.add_argument(
        "--timesteps", type=is_positive_integer, default=1000, 
        help="timesteps for diffusion"
    )
    prs.add_argument(
        "--auto_normalize", type=bool, default=True, 
        help="whether to normalize data to [-1, 1] in GaussianDiffusion()"
    )
    prs.add_argument(
        "--record_every", type=is_positive_integer, default=10, 
        help="record every n steps"
    )
    prs.add_argument("--wandb", action="store_true", 
        help="whether to use wandb"
    )
    prs.add_argument("--project_name", type=str, default="Tony-results",
        help="Project name for wandb"
    )
    prs.add_argument("--entity", type=str, default='gen-ehr',
        help="entity for wandb"
    )
    # for sampling from a trained model
    prs.add_argument("--eval", action="store_true",
        help="whether to evaluate model"
    )
    prs.add_argument("--eval_path", type=is_file_on_disk, required=False,
        help="EVAL: path to load model"
    )
    prs.add_argument(
        "--seed", type=int, default=2023,
    )
    prs.add_argument(
        "--positive_only", action="store_true", required=False,
        help="whether to only sample positive classes",
    )
    prs.add_argument(
        "--loss_lambda", type=float, default=0.8,
    )
    prs.add_argument(
        "--diff_type", type=str, default="mixed", choices=["mixed", "gaussian"],
    )
    prs.add_argument(
        "--dim", type=int, default=64,
    )
    args = prs.parse_args()
    return args

@torch.no_grad()
def collect_samples(model, num_samples, batch_size=20, min=None, max=None, positive_only: bool=False):
    if not positive_only:
        model.eval()
        iterations = num_samples // batch_size
        samples_list = []
        
        for _ in range(1, iterations+1):
            samples = model.sample(batch_size=batch_size)
            samples_list.append(samples)
            
        if min is not None and max is not None:
            samples = torch.cat(samples_list, dim = 0).squeeze().cpu().numpy()
            samples = reverse_normalize(samples, min, max)
        else:
            samples = torch.cat(samples_list, dim = 0).squeeze().cpu().numpy()
    
    else:
        print("*** COLLECT POSITIVE SAMPLES ***")
        model.eval()
        collected_samples = 0
        samples_list = []
        while collected_samples < num_samples:
            samples = model.sample(batch_size=batch_size)
            if torch.any(torch.round(samples[:, -1, :]) == 1):
                mask = torch.round(samples[:, -1, :]) == 1
                mask = mask.any(dim=1)
                samples_list.append(samples[mask])
                collected_samples += len(samples[mask])
            if collected_samples % 1000 == 0:
                print(f"Collected {collected_samples} samples")
            
        if min is not None and max is not None:
            samples = torch.cat(samples_list, dim = 0).squeeze().cpu().numpy()
            samples = reverse_normalize(samples, min, max)
        else:
            samples = torch.cat(samples_list, dim = 0).squeeze().cpu().numpy()
        
    return samples

def model_path_to_sample_path(model_path) -> str:
    sample_path = model_path.replace(".pt", "_samples.npy")
    sample_path = sample_path.replace("models", "samples")
    return sample_path

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    args = parse_arguments()
    seed_everything(args.seed)
    run_id = create_id()
    
    if args.wandb:
        wandb.init(project=args.project_name, entity=args.entity, name=f"ETDiff_{run_id}", config=args)
    else:
        wandb = None
    
    dataset = TimeSeriesDataset(
        categorical_cols = COLUMNS_DICT[args.data_name]["categorical"] if args.data_name in ["eicu", "mimiciv", "mimiciii", "hirid"] else None,             # indicate columns that are not time series values (missing indicators)
        data_name = args.data_name, 
        load_path = args.load_path,
        cut_length = args.cut_length,
    )
    if args.diff_type == "gaussian":
        print("=== Gaussian Diffusion ===")
        model = RNN(
            input_channels = dataset.channels,
            hidden_channels = args.hidden,
            output_channels = dataset.channels,
            layers = args.num_layers,
            model = args.model,
            dropout = args.dropout,
            bidirectional = args.bidirectional,
            self_condition = False,
            embed_dim = args.embed_dim,
            time_dim = args.time_dim,
        )
        diffusion = GaussianDiffusion(
            model = model,
            channels = dataset.channels,
            seq_length = dataset.seq_length,
            timesteps = args.timesteps,
            auto_normalize = args.auto_normalize,
        )
    elif args.diff_type == "mixed":
        print("=== Mixed Diffusion ===")
        model = RNN(
            input_channels = dataset.channels + len(dataset.categorical_cols),
            hidden_channels = (dataset.channels + len(dataset.categorical_cols)) * 4 if args.data_name in ["stock", "energy", "mimiciv", "mimiciii", "hirid"] else args.hidden,
            output_channels = dataset.channels + len(dataset.categorical_cols),
            layers = args.num_layers,
            model = args.model,
            dropout = args.dropout,
            bidirectional = args.bidirectional,
            self_condition = False,
            embed_dim = args.embed_dim,
            time_dim = args.time_dim,
        )
        # model = Unet1D(
        #     dim = args.dim,
        #     dim_mults = (1,2,4),
        #     channels = dataset.channels + len(dataset.categorical_cols),
        #     resnet_block_groups = 2,
        #     embed_dim = args.embed_dim,
        #     time_dim = args.time_dim,
        #     # attn_dim_head = args.attn_dim_head,
        #     # attn_heads = args.attn_heads,
        # )
        diffusion = MixedDiffusion(
            model = model,
            channels = dataset.channels + len(dataset.categorical_cols),
            seq_length = dataset.seq_length,
            timesteps = args.timesteps,
            auto_normalize = args.auto_normalize,
            numerical_features_indices = COLUMNS_DICT[args.data_name]["numerical"],
            categorical_features_indices = COLUMNS_DICT[args.data_name]["categorical"],
            categorical_num_classes = COLUMNS_DICT[args.data_name]["categorical_num_classes"],
            loss_lambda = args.loss_lambda,
        )
    
    etdiff = ETDiff(
        diffusion_model = diffusion,
        dataset = dataset,
        sample_every = args.record_every,
        train_batch_size = args.batch_size,
        diff_lr = args.learning_rate,
        diff_num_steps = args.num_steps,         # total training steps
        gradient_accumulate_every = args.gradient_accumulate_every,    # gradient accumulation steps
        ema_decay = args.ema_decay,
        amp = False,
        wandb = wandb,
        check_point_path = args.check_point_path,
        run_id = run_id,
    )
    
    params = count_params(model)
    print(f"Number of parameters: {params}")
    if args.wandb:
        wandb.log({"total_params": params})
    
    if args.eval == False:
        etdiff.train()
        sample_path = model_path_to_sample_path(args.check_point_path)          # save synthetic samples
        print(f"Creating synthetic samples...")
        samples = collect_samples(model = etdiff.ema.ema_model, num_samples = 20000, batch_size = 100, min = etdiff.sample_min, max = etdiff.sample_max)
        np.save(sample_path, samples)
        print(f"Saved synthetic samples to {sample_path}")
    else:
        print("Collecting...")
        etdiff.load(args.eval_path)
        save_path = model_path_to_sample_path(args.eval_path)
        samples = collect_samples(
            model = etdiff.ema.ema_model, 
            num_samples = 20000, batch_size = 100, 
            min = etdiff.sample_min, max = etdiff.sample_max,
            positive_only = False,
        )
        np.save(save_path, samples)
        print(f"Samples saved to {save_path}")