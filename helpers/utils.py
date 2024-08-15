import argparse
import logging
import os
import random
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# import yaml


def verbose_print(content: str, verbose: bool) -> None:
    if verbose:
        return print(content)
    else:
        return None

def unique(X):
    '''helper to get unique values'''
    if isinstance(X, np.ndarray) or isinstance(X, list):
        result = np.unique(X)
    elif isinstance(X, pd.Series):
        result = X.unique()
    else:
        raise ValueError(f"Unsupported type: {type(X)}")

    return result

def load_yaml(path: str) -> dict:
    with open(path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    return config

def load_object(path: str) -> object:
    """
    To load an object stored using joblib library
    """
    return joblib.load(path)

def assert_dataframes_equal(df1:pd.DataFrame, df2:pd.DataFrame) -> None:
    '''check whether two dataframes are exactly same'''
    pd.testing.assert_frame_equal(df1, df2)
    print("Both frames are EQUAL")

def plt_save_or_show(save_path:str, dpi:int=200, verbose: bool=True) -> None:
    '''save or show plt.plot() visualizations'''
    if save_path is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        if verbose:
            print(f"Figure saved as {save_path}")

def get_logger(log_file: str=None) -> logging:
    logging.basicConfig(
        filename = log_file, 
        format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)
    
    return logging

def seed_everything(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed_value}\n")

def create_id() -> int: return int(time.time())

def is_file_on_disk(file_name):
    if not os.path.isfile(file_name):
        raise argparse.ArgumentTypeError("the file %s does not exist!" % file_name)
    else:
        return file_name
    
def is_positive_integer(value):
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return parsed_value

def is_positive_float(value):
    parsed_value = float(value)
    if parsed_value <= 0.0:
        raise argparse.ArgumentTypeError("%s must be a positive value" % value)
    return parsed_value

def is_file_not_on_disk(file_name):
    if os.path.isfile(file_name):
        raise argparse.ArgumentTypeError("the file %s already exists on disk" % file_name)
    else:
        return file_name

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

def standardize(data):
    assert isinstance(data, np.ndarray), "must be numpy array!"
    return (data - np.mean(data)) / np.std(data)

def standardize_data(data, axis=(0,1), eps=1e-8):
    """standardize along the feature (channel) dimension""" 
    mean = data.mean(axis=axis, keepdims=True)
    std = data.std(axis=axis, keepdims=True)
    data = (data - mean) / (std + eps)
    return data

def normalize_data(data, axis=(0,1), eps=1e-8):
    """normalize along the feature (channel) dimension"""
    min = data.min(axis=axis, keepdims=True)
    max = data.max(axis=axis, keepdims=True)
    data = (data - min) / (max - min + eps)
    return data

def timeseries_median_window(data, window_size=10):
    assert isinstance(data, np.ndarray), "must be numpy array!"
    assert len(data.shape) == 3, "must be 3D array!"

    new_shape = (data.shape[0], data.shape[1], data.shape[2] // window_size)
    result = np.zeros(new_shape)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2] // window_size):
                result[i, j, k] = np.median(data[i, j, k*window_size : (k + 1) * window_size])
    
    return result

def flatten_time_dim_if_3d(data) -> np.ndarray:
    assert isinstance(data, np.ndarray), "must be numpy array!"
    if len(data.shape) == 3:
        data = np.transpose(data, (0, 2, 1)).reshape(len(data), -1)
    return data

def impute_nan_with_zero(data) -> np.ndarray:
    assert isinstance(data, np.ndarray), "must be numpy array!"
    return np.where(np.isnan(data), 0, data)

def impute_nan_with_mean(data) -> np.ndarray:
    assert isinstance(data, np.ndarray), "must be numpy array!"
    return np.where(np.isnan(data), np.nanmean(data), data)

def impute_nan_with_neg_one(data) -> np.ndarray:
    assert isinstance(data, np.ndarray), "must be numpy array!"
    return np.where(np.isnan(data), -1, data)

def get_label_from_2d(data) -> np.ndarray:
    assert len(data.shape) == 2, "data must be 2D!"
    unique = np.round(np.sum(data, axis=1) / data.shape[1])         # do this instead of np.unique(data, axis=1) because some models could return values that mismatch, thus cannot reduce to 1D
    return np.squeeze(unique)

def get_random_indices(data_size):
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    return indices
    
def reverse_normalize(data, min, max) -> np.ndarray:
    return (data * (max - min) + min)          # to turn back into real scale

def exists(x):
    return x is not None