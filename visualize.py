import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from helpers.utils import (create_id, flatten_time_dim_if_3d,
                           get_label_from_2d, get_random_indices,
                           impute_nan_with_neg_one, impute_nan_with_zero,
                           is_file_not_on_disk, is_file_on_disk,
                           is_positive_float, is_positive_integer,
                           normalize_data, seed_everything, smart_load,
                           smart_to_numpy, standardize, standardize_data,
                           timeseries_median_window)
from eval_samples import regular_transform_to_features

GLOBAL_DICT = {
    "mimiciv": {
        "plot_dict": {0: 'Heart Rate', 1: 'SBP', 2: 'DBP', 3: 'Resp. Rate', 4: 'SPO2'},
        "features": 5,
        "time": np.arange(1, 73, 1),
    },
    "eicu": {
        "plot_dict": {0: "Heart Rate", 1: "Resp. Rate", 2: "SPO2", 3: "MAP"},
        "features": 4,
        "time": np.arange(60, 1440, 5),
    },
    "mimiciii": {
        "plot_dict": {0: "Heart Rate", 1: "SBP", 2: "DBP", 3: "MBP", 4: "Resp. Rate", 5: "Temp.", 6: "SPO2"},
        "features": 7,
        "time": np.arange(0, 25, 1),
    },
    "hirid": {
        "plot_dict": {0: "Heart Rate", 1: "SBP", 2: "DBP", 3: "MAP", 4: "SPO2", 5: "ST Elevation", 6: "CVP"},
        "features": 7,
        "time": np.arange(2, 202, 2),
    },
}

COLUMNS_DICT = {
    "mimiciv": {"numerical": [0, 2, 4, 6, 8], "categorical": [1, 3, 5, 7, 9, 10], "categorical_num_classes": [2, 2, 2, 2, 2, 2]},
    "mimiciii": {"numerical": [0, 2, 4, 6, 8, 10, 12], "categorical": [1, 3, 5, 7, 9, 11, 13, 14], "categorical_num_classes": [2, 2, 2, 2, 2, 2, 2, 2]},
    "eicu": {"numerical": [0, 2, 4, 6], "categorical": [1, 3, 5, 7, 8], "categorical_num_classes": [2, 2, 2, 2, 2]},
}

def parse_arguments():        
    prs = argparse.ArgumentParser(
        prog='visualize.py',
        description='Visualizes the time series samples',
        epilog='Copyright (C) 2023',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    prs.add_argument("--save_path", type=str, required=True)
    prs.add_argument("--data_name", type=str, required=True)
    prs.add_argument("--data_path", type=str, required=True)
    args = prs.parse_args()
    return args

def reverse_to_nonan_indicator_2d(cum_nonan_indicator):
    nonan_indicator = np.concatenate([cum_nonan_indicator[:, :1], cum_nonan_indicator[:, 1:] - cum_nonan_indicator[:, :-1]], axis=1)
    return nonan_indicator

def plot_it(data_name, data, save_path="fig.png"):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    rows = 8
    cols = GLOBAL_DICT[data_name]["features"]
    fig, axs = plt.subplots(rows, cols, figsize=(cols*11, rows*8))
    data_iterator = iter(data)
    
    x = GLOBAL_DICT[data_name]["time"]  
    plot_dict = GLOBAL_DICT[data_name]["plot_dict"]

    for i in tqdm(range(rows), desc="plotting..."):
        timeseries = next(data_iterator)
        color_choice = iter(sns.color_palette())
        for j in range(cols):
            idx = j
            data = timeseries[idx]
            if np.all(np.isnan(data)):
                next(color_choice)
                continue
            else:
                sns.lineplot(x=x, y=data, ax=axs[i, j], label=plot_dict[idx], color=next(color_choice), linewidth=2)
                
            axs[i, j].set_xlabel("Time", fontsize=30)
            axs[i, j].set_ylabel("Measurement Value", fontsize=30)
            axs[i, j].set_xlim(x[0], x[-1])
            axs[i, j].legend(fontsize=50)
            axs[i, j].tick_params(axis='both', which='major', labelsize=30)

    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved at {save_path}")

if __name__ == "__main__":
    np.random.seed(2026)
    args = parse_arguments()
    data = smart_load(args.data_path)
    data = smart_to_numpy(data)
    data = regular_transform_to_features(data, args.data_name)
    print(data.shape)
    plot_it(args.data_name, data, args.save_path)