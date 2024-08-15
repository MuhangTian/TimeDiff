#!/usr/bin/env ipython
# Utility functions that don't fit in other scripts
import argparse
import json
import numpy as np
import torch

def rgan_options_parser():
    """
    Define parser to parse options from command line, with defaults.
    Refer to this function for definitions of various variables.
    """
    parser = argparse.ArgumentParser(description='Train a GAN to generate sequential, real-valued data.')
    # meta-option
#     parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str, default='')
    # options pertaining to data
#     parser.add_argument('--data', help='what kind of data to train with?',
#             default='gp_rbf',
#             choices=['gp_rbf', 'sine', 'mnist', 'load',
#                 'resampled_eICU', 'eICU_task'])
#     parser.add_argument('--num_samples', type=int, help='how many training examples \
#                     to generate?', default=28*5*100)
#     parser.add_argument('--seq_length', type=int, default=30)
#     parser.add_argument('--num_signals', type=int, default=1)
#     parser.add_argument('--normalise', type=bool, default=False, help='normalise the \
        #     training/vali/test data (during split)?')
    parser.add_argument('--cond_dim', type=int, default=0, help='dimension of \
            *conditional* input')
    parser.add_argument('--max_val', type=int, default=1, help='assume conditional \
            codes come from [0, max_val)')
    parser.add_argument('--one_hot', type=bool, default=False, help='convert categorical \
            conditional information to one-hot encoding')
    parser.add_argument('--predict_labels', type=bool, default=False, help='instead \
            of conditioning with labels, require model to output them')
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--dump_path", type=str, required=True)
#     ### for gp_rbf
#     parser.add_argument('--scale', type=float, default=0.1)
#             ### for sin (should be using subparsers for this...)
#     parser.add_argument('--freq_low', type=float, default=1.0)
#     parser.add_argument('--freq_high', type=float, default=5.0)
#     parser.add_argument('--amplitude_low', type=float, default=0.1)
#     parser.add_argument('--amplitude_high', type=float, default=0.9)
#             ### for mnist
#     parser.add_argument('--multivariate_mnist', type=bool, default=False)
#     parser.add_argument('--full_mnist', type=bool, default=False)
#             ### for loading
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument("--sample_save_path", type=str, required=True)
            ### for eICU
#     parser.add_argument('--resample_rate_in_min', type=int, default=15)

    # hyperparameters of the model
    parser.add_argument('--hidden_units_g', type=int, default=100)
    parser.add_argument('--hidden_units_d', type=int, default=100)
    parser.add_argument('--kappa', type=float, help='weight between final output \
            and intermediate steps in discriminator cost (1 = all \
            intermediate', default=1)
    parser.add_argument('--latent_dim', type=int, default=5, help='dimensionality \
            of the latent/noise space')
    parser.add_argument('--batch_mean', type=bool, default=False, help='append the mean \
            of the batch to all variables for calculating discriminator loss')
    parser.add_argument('--learn_scale', type=bool, default=False, help='make the \
            "scale" parameter at the output of the generator learnable (else fixed \
            to 1')
    # options pertaining to training
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=28)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--D_rounds', type=int, default=5, help='number of rounds \
            of discriminator training')
    parser.add_argument('--G_rounds', type=int, default=1, help='number of rounds \
            of generator training')
    parser.add_argument('--use_time', type=bool, default=False, help='enforce \
            latent dimension 0 to correspond to time')
    parser.add_argument('--WGAN', type=bool, default=False)
    parser.add_argument('--WGAN_clip', type=bool, default=False)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--wrong_labels', type=bool, default=False, help='augment \
            discriminator loss with real examples with wrong (~shuffled, sort of) labels')
    # options pertaining to evaluation and exploration
    parser.add_argument('--identifier', type=str, default='test', help='identifier \
            string for output files')
    # options pertaining to differential privacy
    parser.add_argument('--dp', type=bool, default=False, help='train discriminator \
            with differentially private SGD?')
    parser.add_argument('--l2norm_bound', type=float, default=1e-5,
            help='bound on norm of individual gradients for DP training')
    parser.add_argument('--batches_per_lot', type=int, default=1,
            help='number of batches per lot (for DP)')
    parser.add_argument('--dp_sigma', type=float, default=1e-5,
            help='sigma for noise added (for DP)')

    return parser

def load_settings_from_file(settings):
    """
    Handle loading settings from a JSON file, filling in missing settings from
    the command line defaults, but otherwise overwriting them.
    """
    settings_path = './experiments/settings/' + settings['settings_file'] + '.txt'
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r'))
    # check for settings missing in file
    for key in settings.keys():
        if not key in settings_loaded:
            print(key, 'not found in loaded settings - adopting value from command line defaults: ', settings[key])
            # overwrite parsed/default settings with those read from file, allowing for
    # (potentially new) default settings not present in file
    settings.update(settings_loaded)
    return settings

def normalize_data(data, indicator_cols: list=None, eps=0):
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

def reverse_normalize(data, min, max) -> np.ndarray:
    return (data * (max - min) + min)          # to turn back into real scale

def replace_nan_with_mean(data: torch.Tensor):
    mean = data.nanmean(dim=(0, 2))[None, :, None]
    data = torch.where(torch.isnan(data), mean, data)
    assert torch.all(~torch.isnan(data)), 'failed to replace NaNs!'
    return data

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