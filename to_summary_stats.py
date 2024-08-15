import argparse

import numpy as np
import pandas as pd
from scipy.stats import mode, skew
from tqdm import tqdm

from helpers.utils import smart_load, smart_to_numpy

CHANNELS_TO_FEATURES = {
    "eicu": {0: "heartrate", 1: "resprate", 2: "spo2", 3: "meanbp", 4: "hospital_expire_flag"},
    "mimiciv": {0: "heartrate", 1: "sbp", 2: "dbp", 3: "resprate", 4: "spo2", 5: "hospital_expire_flag"},
    "mimiciii": {0: "heartrate", 1: "sbp", 2: "dbp", 3: "mbp", 4: "resprate", 5: "temp", 6: "spo2", 7: "hospital_expire_flag"},
    "hirid": {0: "heartrate", 1: "sbp", 2: "dbp", 3: "mbp", 4: "spo2", 5: "st", 6: "cvt", 7: "hospital_expire_flag"},
}

def parse_args():
    prs = argparse.ArgumentParser(
        prog='to_summary_stats.py',
        description='Turn samples into tabular summary statistics form',
        epilog='Copyright (C) 2023',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    prs.add_argument("--sync_path", type=str, required=True)
    prs.add_argument("--train_path", type=str, required=True)
    prs.add_argument("--test_path", type=str, required=True)
    prs.add_argument("--sync_save_path", type=str, required=True)
    prs.add_argument("--train_save_path", type=str, required=True)
    prs.add_argument("--test_save_path", type=str, required=True)
    args = prs.parse_args()
    return args

def reverse_to_nonan_indicator_2d(cum_nonan_indicator):
    nonan_indicator = np.concatenate([cum_nonan_indicator[:, :1], cum_nonan_indicator[:, 1:] - cum_nonan_indicator[:, :-1]], axis=1)
    return nonan_indicator

def regular_transform_to_features(samples, data_name="eicu"):
    
    assert isinstance(samples, np.ndarray), "samples must be a numpy array"
    
    if data_name == "eicu":
        unmasked_samples = np.zeros((samples.shape[0], 5, samples.shape[2]))
        for i in range(0, 7, 2):
            nan_indicator = np.round(samples[:,i+1,:])
            data = np.where(nan_indicator == 1, np.nan, samples[:,i,:])
            unmasked_samples[:, int(i/2), :] = data
        unmasked_samples[:, 4, :] = np.round(samples[:, -1, :])         # label channel
        
    elif data_name == "mimiciii":
        unmasked_samples = np.zeros((samples.shape[0], 8, samples.shape[2]))
        for i in range(0, 14, 2):
            nan_indicator = np.round(samples[:,i+1,:])
            data = np.where(nan_indicator == 1, np.nan, samples[:,i,:])
            unmasked_samples[:, int(i/2), :] = data
        unmasked_samples[:, 7, :] = np.round(samples[:, -1, :])         # label channel
    
    elif data_name == "mimiciv":
        unmasked_samples = np.zeros((samples.shape[0], 6, samples.shape[2]))
        for i in range(0, 10, 2):
            nan_indicator = np.round(samples[:,i+1,:])
            data = np.where(nan_indicator == 1, np.nan, samples[:,i,:])
            unmasked_samples[:, int(i/2), :] = data
        unmasked_samples[:, 5, :] = np.round(samples[:, -1, :])         # label channel
    
    elif data_name == "hirid":
        unmasked_samples = samples          # no missing values for hirid so no need to do anything
    
    else:
        raise ValueError(f"Unknown data name: {data_name}")
        
    return unmasked_samples

def turn_timeseries_to_summary_stats_df(timeseries_sample, channels_to_features):
    """
    Turn timeseries dataframe to summary statistics dataframe.
    """
    result_df = {}
    for patient_idx in tqdm(range(timeseries_sample.shape[0]), desc="Turning timeseries to summary stats..."):
        for channel_idx in range(timeseries_sample.shape[1]):
            if channel_idx == timeseries_sample.shape[1] - 1:            # for label column
                try:
                    result_df['hospital_expire_flag'].append(np.unique(timeseries_sample[patient_idx, channel_idx, :])[0])
                except:
                    result_df['hospital_expire_flag'] = [np.unique(timeseries_sample[patient_idx, channel_idx, :])[0]]
                continue
            feature_name = channels_to_features[channel_idx]
            arr = timeseries_sample[patient_idx, channel_idx, :]
            arr = np.squeeze(arr[~np.isnan(arr)])
            if arr.size <= 1:
                first, min, max, range_, mean, std, median, mode_, kurtosis_, lower_quartile, upper_quartile, iqr, skewness = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            else:
                if arr.size == 1:
                    first = arr
                else:
                    first = arr[0]
                min = arr.min()
                max = arr.max()
                range_ = max - min
                mean = arr.mean()
                std = arr.std()
                mode_ = mode(arr, keepdims=True)[0][0]
                skewness = skew(arr)
            
            try:
                result_df[f"{feature_name}_first"].append(first)
                result_df[f"{feature_name}_min"].append(min)
                result_df[f"{feature_name}_max"].append(max)
                result_df[f"{feature_name}_range"].append(range_)
                result_df[f"{feature_name}_mean"].append(mean)
                result_df[f"{feature_name}_std"].append(std)
                result_df[f"{feature_name}_mode"].append(mode_)
                result_df[f"{feature_name}_skewness"].append(skewness)
            except:
                result_df[f"{feature_name}_first"] = [first]
                result_df[f"{feature_name}_min"] = [min]
                result_df[f"{feature_name}_max"] = [max]
                result_df[f"{feature_name}_range"] = [range_]
                result_df[f"{feature_name}_mean"] = [mean]
                result_df[f"{feature_name}_std"] = [std]
                result_df[f"{feature_name}_mode"] = [mode_]
                result_df[f"{feature_name}_skewness"] = [skewness]
            
    result_df = pd.DataFrame(result_df)
    return result_df

def get_data_name(path):
    if "mimiciv" in path or "mimic-iv" in path:
        return "mimiciv"
    elif "mimiciii" in path or "mimic-iii" in path:
        return "mimiciii"
    elif "eicu" in path:
        return "eicu"
    elif "hirid" in path:
        return "hirid"

if __name__ == "__main__":
    args = parse_args()
    args.data_name = get_data_name(args.train_path)
    sync_data, train_data, test_data = smart_to_numpy(smart_load(args.sync_path)), smart_to_numpy(smart_load(args.train_path)), smart_to_numpy(smart_load(args.test_path))
    sync_data, train_data, test_data = regular_transform_to_features(sync_data, args.data_name), regular_transform_to_features(train_data, args.data_name)[:, :, :sync_data.shape[-1]], regular_transform_to_features(test_data, args.data_name)[:, :, :sync_data.shape[-1]]
    print(sync_data.shape, train_data.shape, test_data.shape)
    
    channels_to_features = CHANNELS_TO_FEATURES[args.data_name]
    
    sync_df = turn_timeseries_to_summary_stats_df(sync_data, channels_to_features)
    train_df = turn_timeseries_to_summary_stats_df(train_data, channels_to_features)
    test_df = turn_timeseries_to_summary_stats_df(test_data, channels_to_features)
    print(f"Sync df shape: {sync_df.shape}, train df shape: {train_df.shape}, test df shape: {test_df.shape}")
    print(f"Class Ratio | Train: {train_df['hospital_expire_flag'].sum() / train_df.shape[0]}, Test: {test_df['hospital_expire_flag'].sum() / test_df.shape[0]}, Sync: {sync_df['hospital_expire_flag'].sum() / sync_df.shape[0]}")
    
    print("** Synthetic Data **")
    print(sync_df.describe())
    
    print("** Train Data **")
    print(train_df.describe())
    
    sync_df.to_csv(args.sync_save_path, index=False)
    train_df.to_csv(args.train_save_path, index=False)
    test_df.to_csv(args.test_save_path, index=False)