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
                           seed_everything)
from models.latent_diffusion.gaussian_diffusion import GaussianDiffusion
from models.latent_diffusion.generative_model import LatentDiffusion
from models.latent_diffusion.autoencoder import Encoder, Decoder
from models.latent_diffusion.unet import UNet
from models.latent_diffusion.utils import IrregularTSDataset


def parse_arguments():        
    prs = argparse.ArgumentParser(
        prog='ours_train.py',
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
        "--check_point_path", type=is_file_not_on_disk, required=True,
        help="path to save model"
    )
    prs.add_argument(
        "--e_hidden_channels", type=int, default=64,
        help="encoder hidden channels, hidden channel of Zt"
    )
    prs.add_argument(
        "--e_output_channels", type=int, default=64,
        help="encoder output channels (after appling FC layer), determines the size of latent tensor"
    )
    prs.add_argument(
        "--d_hidden_channels", type=int, default=64,
        help="decoder hidden channels, for GRU-ODE and GRU, need to be the same as d_gru_input_size"
    )
    prs.add_argument(
        "--d_gru_input_size", type=int, default=64,
        help="size of input to GRU, this will be changed by a layer before GRU on input data"
    )
    prs.add_argument(
        "--d_x_hidden", type=int, default=64,
        help="hidden state of the linear layer before changing size to gru input size"
    )
    prs.add_argument(
        "--d_delta_t", type=float, default=2.5,
        help="delta_t used for solving GRU-ODE"
    )
    prs.add_argument(
        "--d_num_layer", type=int, default=2,
        help="number of layers in GRU-ODE decoder"
    )
    prs.add_argument(
        "--d_last_activation", type=str, default="softplus",
        choices=["softplus", "sigmoid", "tanh", "identity"],
        help="last activation function of GRU-ODE decoder, choice depends on data type"
    )
    prs.add_argument(
        "--ae_num_steps", type=int, default=5,
        help="number of iterations to train autoencoder"
    )
    prs.add_argument(
        "--mults", nargs="+", type=int, #required=True,
        default = (1, 2, 4, 8),
        help="UNet dimension multipliers"
    )
    prs.add_argument(
        "--batch_size", type=is_positive_integer, default=32, 
        help="batch size"
    )
    prs.add_argument(
        "--dim", type=is_positive_integer, default=64, 
        help="UNet initial dimension"
    )
    prs.add_argument(
        "--learning_rate", type=is_positive_float, default=8e-5, 
        help="learning rate"
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
    prs.add_argument("--eval", type=str, default=None,
        # default='ema',
        help="EVAL: whether to use model to sample data or not, if yes, specify 'ema' or 'model'"
    )
    prs.add_argument("--eval_num_samples", type=is_positive_integer, default=20, 
        help="EVAL: number of samples to collect"
    )
    prs.add_argument("--eval_path", type=is_file_on_disk, required=False,
        help="EVAL: path to load model"
    )
    prs.add_argument("--eval_bsz", type=is_positive_integer, default=20, 
        help="EVAL: number of samples for each reverse diffusion process"
    )
    args = prs.parse_args()
    return args

def collect_samples(model, num_samples, batch_size=20, min=None, max=None):
    model.eval()
    iterations = num_samples // batch_size
    samples_list = []
    
    for i in range(1, iterations+1):
        with torch.no_grad():
            samples = model.sample(batch_size=batch_size)
        samples_list.append(samples)
        
    if min is not None and max is not None:
        samples = torch.cat(samples_list, dim = 0).squeeze().cpu()
        samples = (samples * (max - min) + min).numpy()
        corrected_samples = np.zeros((samples.shape[0], 5, samples.shape[2]))       # NOTE: the following code is specific to eICU dataset and features trained on 7.24
        for i in range(0, 7, 2):
            corrected_samples[:, int(i/2), :] = np.where(np.round(samples[:, i+1, :]) == 1, np.nan, samples[:, i, :])
        corrected_samples[:, 4, :] = np.round(samples[:, -1, :])
        samples = corrected_samples
    else:
        samples = torch.cat(samples_list, dim = 0).squeeze().cpu().numpy()
        
    return samples

def visualize_samples(samples):
    rows = samples.shape[0]
    cols = 5
    fig, axs = plt.subplots(rows, cols, figsize=(cols*11, rows*8))
    data_iterator = iter(samples)
    x = 5*np.arange(1, samples.shape[-1]+1) / 60
    plot_dict = {0: 'Heart Rate (Beats/Min)', 1: 'Respiration (Breaths/Min)', 2: 'SPO2 (%)', 3: 'Mean Arterial Pressure (mmHg)', 4: 'Mortality (1/0)'}
    
    for i in range(rows):
        timeseries = next(data_iterator)
        color_choice = iter(sns.color_palette())
        for j in range(cols):
            if np.all(np.isnan(timeseries[j])):
                next(color_choice)
                continue
            else:
                sns.lineplot(x=x, y=timeseries[j], ax=axs[i, j], linewidth=2, label=plot_dict[j], color=next(color_choice))            
            axs[i, j].set_xlabel("Time (Hours)", fontsize=30)
            axs[i, j].set_ylabel("Measurement Value", fontsize=30)
            axs[i, j].legend(fontsize=30)
            axs[i, j].tick_params(axis='both', which='major', labelsize=30)
    
    plt.tight_layout()
    plt.legend()
    plt.savefig('img/samples.png')

def model_path_to_sample_path(model_path) -> str:
    sample_path = model_path.replace(".pt", "_samples.npy")
    sample_path = sample_path.replace("models", "samples")
    return sample_path

def normalize_data(data: torch.Tensor, eps=0):
    min, max = data.amin(dim=(0, 2))[None, :, None], data.amax(dim=(0, 2))[None, :, None]
    data = (data - min + eps) / (max - min + 2*eps)
    assert torch.all((data >= 0) & (data <= 1)), 'failed to normalize data'
    return data.float(), min.float(), max.float()

def replace_nan_with_other(data: torch.Tensor, other: str='mean'):
    if other == 'mean':
        mean = data.nanmean(dim=(0, 2))[None, :, None]
        data = torch.where(torch.isnan(data), mean, data)
    elif other == 'zero':
        data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
    else:
        raise ValueError()
    assert torch.all(~torch.isnan(data)), 'failed to replace NaNs!'
    return data

if __name__ == "__main__":
    args = parse_arguments()
    seed_everything(2023)
    run_id = create_id()
    
    if args.wandb:
        wandb.init(project=args.project_name, entity=args.entity, name=f"latent_diffusion_{run_id}", config=args)
    else:
        wandb = None
    
    dataset = IrregularTSDataset(
        indicator_cols = None,             # indicate columns that are not time series values (missing indicators)
        data_name = args.data_name, 
        load_path = args.load_path,
    )
    
    encoder = Encoder(
        input_channels = dataset.channels + 1,          # plus one since time dimension is used to construct coeffs, this doesn't effect autoencoder since time generation is done through missing indicators
        hidden_channels = args.e_hidden_channels,
        output_channels = args.e_output_channels,
    )
    decoder = Decoder(
        input_size = args.e_output_channels,
        hidden_size = args.d_hidden_channels,
        output_size = dataset.channels,
        gru_input_size = args.d_gru_input_size,
        x_hidden = args.d_x_hidden,
        delta_t = args.d_delta_t,
        num_layer = args.d_num_layer,
        last_activation = args.d_last_activation,
    )
    
    unet = UNet(
        dim = args.dim,
        dim_mults = args.mults,
        channels = args.e_output_channels,
    )
    diffusion = GaussianDiffusion(
        model = unet,
        seq_length = dataset.seq_length,
        timesteps = args.timesteps,
        auto_normalize = args.auto_normalize,
    )
    
    trainer = LatentDiffusion(
        diffusion_model = diffusion,
        encoder = encoder,
        decoder = decoder,
        dataset = dataset,
        sample_every = args.record_every,
        autoencoder_num_steps = args.ae_num_steps,
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
    
    if args.eval is None:
        trainer.train()
    else:
        print("Collecting...")
        trainer.load(args.eval_path)
        
        if args.eval == 'ema':
            model = trainer.ema.ema_model
        elif args.eval == 'model':
            model = trainer.model
            
        save_path = model_path_to_sample_path(args.eval_path)
        samples = collect_samples(model = model, num_samples = args.eval_num_samples, batch_size = args.eval_bsz, min = min, max = max)
        # np.save(save_path, samples)
        # print(f"Samples saved to {save_path}")
        visualize_samples(samples)