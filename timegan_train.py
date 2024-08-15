'''
Benchmark our results with the following work:

2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar
Original Code Author: Jinsung Yoon (jsyoon0823@gmail.com)
'''
import argparse
import random
# imports
import sys

import joblib
import numpy as np
import tensorflow as tf

import wandb
from helpers.utils import is_file_on_disk, smart_load, smart_to_numpy
from models.time_gan.metrics.discriminative_score_metrics import \
    discriminative_score_metrics
from models.time_gan.tgan import tgan

tf.logging.set_verbosity(tf.logging.ERROR)

def parse_args():
    prs = argparse.ArgumentParser(
        prog='timegan_train.py',
        description='Train TimeGAN for benchmarking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    prs.add_argument(
        "--load_path",  type=is_file_on_disk,  required=True,
        help="path to load data"
    )
    prs.add_argument(
        "--cut_length", type=int, default=60,
        help="length of the time series"
    )
    prs.add_argument(
        "--iter", type=int, default=50000,
        help="Number of iterations"
    )
    prs.add_argument(
        "--hidden_dim", type=int, default=256,
        help="hidden dimension for RNN"
    )
    prs.add_argument(
        "--batch_size", type=int,
        default=128,
    )
    prs.add_argument(
        "--num_layers", type=int,
        default=3,
    )
    prs.add_argument(
        "--module_name", type=str, default="gru",
        help="module to use for TimeGAN, options are 'gru', 'lstm' or 'lstmLN'"
    )
    prs.add_argument(
        "--sub_iter", type=int, default=10,
        help="Number of iterations for discriminative score"
    )
    prs.add_argument(
        "--name", type=str, required=True,
        help="name to save the samples"
    )
    prs.add_argument(
        "--wandb", action="store_true", 
        help="whether to use wandb"
    )
    prs.add_argument(
        "--project_name", type=str, default="Tony-results",
        help="Project name for wandb"
    )
    prs.add_argument(
        "--entity", type=str, default='gen-ehr',
        help="entity for wandb"
    )
    prs.add_argument(
        "--seed", type=int, default=2023,
    )
    args = prs.parse_args()
    return args

def replace_nan_with_other_np(data: np.ndarray, other: str='mean'):
    if other == 'mean':
        mean = np.nanmean(data, axis=(0, 2))[None, :, None]
        data = np.where(np.isnan(data), mean, data)
    elif other == 'zero':
        data = np.where(np.isnan(data), np.zeros_like(data), data)
    else:
        raise ValueError()
    assert not np.isnan(data).any(), 'failed to replace NaNs!'
    return data

def set_seed(seed_value):
    tf.set_random_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    print(f"Seed set to {seed_value}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    
    if args.wandb:
        wandb.init(project=args.project_name, entity=args.entity, name=args.name, config=args)
    else:
        wandb = None
    
    # Experiments iterations
    dataX = np.transpose(replace_nan_with_other_np(smart_load(args.load_path)), (0, 2, 1))[:,:args.cut_length,:]
    Sub_Iteration = args.sub_iter
    seq_length = dataX.shape[1]

    print(f"Using GPU? {tf.test.is_gpu_available()}")
    print(f"Dimension of data: {np.asarray(dataX).shape}\nseq_length: {seq_length}")      # (#samples, #seq_length, #features)
    print(f"load_path: {args.load_path}")
    parameters = dict()
    parameters['hidden_dim'] = args.hidden_dim
    parameters['num_layers'] = args.num_layers
    parameters['iterations'] = args.iter
    parameters['batch_size'] = args.batch_size
    parameters['module_name'] = args.module_name   # Other options: 'lstm' or 'lstmLN'
    parameters['z_dim'] = len(dataX[0][0,:]) 
    
    # Output Initialization
    Discriminative_Score = list()

    # Synthetic Data Generation
    # dataX_hat = tgan(dataX, parameters, name=args.name, wandb=wandb, train=False)
    dataX_hat = tgan(dataX, parameters, name=args.name, wandb=wandb, train=True)
    np.save(f"results/samples/{args.name}.npy", dataX_hat)
    print(dataX_hat.shape)
    
    print('Finish Synthetic Data Generation!')