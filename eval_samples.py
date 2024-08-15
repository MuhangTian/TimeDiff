import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

from evaluate.nn_classifier import FCN, RNN, nnTimeSeriesClassifier
from evaluate.privacy import MIR, NNAA
from evaluate.utility import (TRTR, TRTS, TSRTR, TSTR, TSTR_TRTR,
                              DiscriminativeScore, PredictiveScore,
                              TrajectoryVisualizer, TSRTR_Imbalance, tSNE,
                              tSNE_Analysis)
from helpers.utils import (create_id, flatten_time_dim_if_3d,
                           get_label_from_2d, get_random_indices,
                           impute_nan_with_neg_one, impute_nan_with_zero,
                           is_file_not_on_disk, is_file_on_disk,
                           is_positive_float, is_positive_integer,
                           normalize_data, seed_everything, smart_load,
                           smart_to_numpy, standardize, standardize_data,
                           timeseries_median_window)

COLUMNS_DICT = {
    "mimiciv": {"numerical": [0, 2, 4, 6, 8], "categorical": [1, 3, 5, 7, 9, 10], "categorical_num_classes": [2, 2, 2, 2, 2, 2]},
    "mimiciii": {"numerical": [0, 2, 4, 6, 8, 10, 12], "categorical": [1, 3, 5, 7, 9, 11, 13, 14], "categorical_num_classes": [2, 2, 2, 2, 2, 2, 2, 2]},
    "eicu": {"numerical": [0, 2, 4, 6], "categorical": [1, 3, 5, 7, 8], "categorical_num_classes": [2, 2, 2, 2, 2]},
}


def parse_arguments():        
    prs = argparse.ArgumentParser(
        prog='eval_samples.py',
        description='Evaluate synthetic samples using various metrics',
        epilog='Copyright (C) 2023',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    prs.add_argument(
        "--data_name", type=str, required=True,
        help="name of the dataset",
    )
    prs.add_argument(
        "--t_sne_num", type=int, default=2000,
        help="number of samples to use for t-sne",
    )
    prs.add_argument(
        "--perplexity", type=int, default=30,
        help="perplexity for t-sne",
    )
    prs.add_argument(
        "--predictive_iter", type=int, default=5000,
        help="number of iterations for predictive score",
    )
    prs.add_argument(
        "--discriminative_iter", type=int, default=2000,
        help="number of iterations for discriminative score",
    )
    prs.add_argument(
        "--discri_pred_scaler", type=str, default="standardize",
        help="how to preprocess data to train discriminator and predictor",
    )
    prs.add_argument(
        "--sync_path", type=str, required=True,
        help="path to load synthetic data"
    )
    prs.add_argument(
        "--train_path", type=is_file_on_disk, required=True,
        help="path to load real training data"
    )
    prs.add_argument(
        "--test_path", type=is_file_on_disk, required=True,
        help="path to load real test data"
    )
    prs.add_argument(
        "--metric", type=str, required=True,
        help="metric to evaluate synthetic data"
    ),
    prs.add_argument(
        "--project_name", type=str, default="Tony-results",
        help="Project name for wandb"
    )
    prs.add_argument(
        "--entity", type=str, default='gen-ehr',
        help="entity for wandb"
    )
    prs.add_argument(
        "--runs", type=int, default=10,
        help="whether to evaluate over 10 runs"
    )
    prs.add_argument(
        "--img_name", type=str, default="t-sne.png",
        help="name of the image to save",
    )
    prs.add_argument(
        "--seed", type=is_positive_integer, default=2023,
    )
    prs.add_argument(
        "--scoring_metric", type=str, choices=["auprc", "auc"], default="auc",
    )
    prs.add_argument(
        "--mir_threshold", type=is_positive_float, default=0.1,
    )
    prs.add_argument(
        "--privacy_with_real", action="store_true",
        help="whether to use real data to evaluate privacy",
    )
    prs.add_argument(
        "--mir_model_name", type=str, default=None,
        help="model name for MIR score",
    )
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
        
    elif data_name == "eicu_ONEHOT":
        unmasked_samples = np.zeros((samples.shape[0], 5, samples.shape[2]))
        numerical_idx = 0
        for i in range(len(COLUMNS_DICT["eicu"]["numerical"]), samples.shape[1] - 2, 2):
            nan_indicator_onehot = torch.tensor(samples[:, [i, i+1], :])
            nan_indicator_onehot = F.softmax(nan_indicator_onehot, dim=1)
            nan_indicator = torch.argmax(nan_indicator_onehot, dim=1)
            nan_indicator = nan_indicator.numpy()
            data = np.where(nan_indicator == 1, np.nan, samples[:, numerical_idx, :])
            unmasked_samples[:, numerical_idx, :] = data
            numerical_idx += 1
        mortality_indicator_onehot = samples[:, [-2, -1], :]
        mortality_indicator_onehot = F.softmax(torch.tensor(mortality_indicator_onehot), dim=1)
        mortality_indicator = torch.argmax(mortality_indicator_onehot, dim=1).numpy()
        unmasked_samples[:, 4, :] = mortality_indicator
    
    elif data_name == "mimiciii_ONEHOT":
        unmasked_samples = np.zeros((samples.shape[0], 8, samples.shape[2]))
        numerical_idx = 0
        for i in range(len(COLUMNS_DICT["mimiciii"]["numerical"]), samples.shape[1] - 2, 2):
            nan_indicator_onehot = torch.tensor(samples[:, [i, i+1], :])
            nan_indicator_onehot = F.softmax(nan_indicator_onehot, dim=1)
            nan_indicator = torch.argmax(nan_indicator_onehot, dim=1)
            nan_indicator = nan_indicator.numpy()
            data = np.where(nan_indicator == 1, np.nan, samples[:, numerical_idx, :])
            unmasked_samples[:, numerical_idx, :] = data
            numerical_idx += 1
        mortality_indicator_onehot = samples[:, [-2, -1], :]
        mortality_indicator_onehot = F.softmax(torch.tensor(mortality_indicator_onehot), dim=1)
        mortality_indicator = torch.argmax(mortality_indicator_onehot, dim=1).numpy()
        unmasked_samples[:, 7, :] = mortality_indicator
    
    elif data_name == "mimiciv_ONEHOT":
        unmasked_samples = np.zeros((samples.shape[0], 6, samples.shape[2]))
        numerical_idx = 0
        for i in range(len(COLUMNS_DICT["mimiciv"]["numerical"]), samples.shape[1] - 2, 2):
            nan_indicator_onehot = torch.tensor(samples[:, [i, i+1], :])
            nan_indicator_onehot = F.softmax(nan_indicator_onehot, dim=1)
            nan_indicator = torch.argmax(nan_indicator_onehot, dim=1)
            nan_indicator = nan_indicator.numpy()
            data = np.where(nan_indicator == 1, np.nan, samples[:, numerical_idx, :])
            unmasked_samples[:, numerical_idx, :] = data
            numerical_idx += 1
        mortality_indicator_onehot = samples[:, [-2, -1], :]
        mortality_indicator_onehot = F.softmax(torch.tensor(mortality_indicator_onehot), dim=1)
        mortality_indicator = torch.argmax(mortality_indicator_onehot, dim=1).numpy()
        unmasked_samples[:, 5, :] = mortality_indicator
    
    
    else:
        raise ValueError(f"Unknown data name: {data_name}")
        
    return unmasked_samples
    

if __name__ == "__main__":
    args = parse_arguments()
    seed_everything(args.seed)
    print(f"*** METRIC: {args.metric} ***")
    
    if args.metric in ["discriminative", "predictive", "discriminative-real", "predictive-real"]:
        import tensorflow as tf
        tf.set_random_seed(args.seed)
        print(f"Set tensorflow seed to {args.seed}")
    
    print(f"Synthetic Sample Path: {args.sync_path}")
    sync_data, train_data, test_data = smart_load(args.sync_path), smart_load(args.train_path), smart_load(args.test_path)
    
    if args.sync_path.split('.')[-1] != "csv":
        if args.data_name in ["eicu", "mimiciii", "mimiciv", "hirid", "eicu_ONEHOT", "mimiciii_ONEHOT", "mimiciv_ONEHOT"]:
            sync_data, train_data, test_data = smart_to_numpy(sync_data), smart_to_numpy(train_data), smart_to_numpy(test_data)

            if "halo" in args.sync_path: # do not apply regular transform for HALO model, since it requires reverse discretization in a separate file
                train_data, test_data = regular_transform_to_features(train_data, args.data_name)[:, :, :sync_data.shape[-1]], regular_transform_to_features(test_data, args.data_name)[:, :, :sync_data.shape[-1]]
            else:
                sync_data, train_data, test_data = regular_transform_to_features(sync_data, args.data_name), regular_transform_to_features(train_data, args.data_name)[:, :, :sync_data.shape[-1]], regular_transform_to_features(test_data, args.data_name)[:, :, :sync_data.shape[-1]]
                
            X_sync, X_train, X_eval = sync_data[:, :-1, :], train_data[:, :-1, :], test_data[:, :-1, :]
            y_sync, y_train, y_eval = get_label_from_2d(sync_data[:, -1, :]), get_label_from_2d(train_data[:, -1, :]), get_label_from_2d(test_data[:, -1, :])
            print(f"Class imbalance stats:\nSynthetic: {100*np.mean(y_sync):.2f}%\nReal Train: {100*np.mean(y_train):.3f}%\nReal Test: {100*np.mean(y_eval):.3f}%")
        else:           # NOTE: other regular time series data doesn't have labels, nor have missingness
            X_sync, X_train, X_eval = smart_to_numpy(sync_data), smart_to_numpy(train_data), smart_to_numpy(test_data)
            if X_sync.shape[-1] < X_train.shape[-1] or X_sync.shape[-1] < X_eval.shape[-1]:
                # adjust shape since some synthetic samples could be shorter than real samples
                X_train, X_eval = X_train[:,:,:X_sync.shape[-1]], X_eval[:,:,:X_sync.shape[-1]]
        csv_flag = False
    else:
        csv_flag = True
    
    if args.metric in ["discriminative", "predictive", "discriminative-real", "predictive-real", "nnaa", "mir"]:
        iterations = args.runs
    else:
        iterations = 1
    
    score_arr, AA_test_arr, AA_train_arr = [], [], []
    for i in range(iterations):
        if args.metric == "nnaa":
            sync_data_tmp, train_data_tmp, test_data_tmp = impute_nan_with_neg_one(sync_data), impute_nan_with_neg_one(train_data), impute_nan_with_neg_one(test_data)
            sync_data_tmp, train_data_tmp, test_data_tmp = standardize_data(sync_data_tmp, axis=(0,2)), standardize_data(train_data_tmp, axis=(0,2)), standardize_data(test_data_tmp, axis=(0,2))
            sync_data_tmp, train_data_tmp, test_data_tmp = flatten_time_dim_if_3d(sync_data_tmp), flatten_time_dim_if_3d(train_data_tmp), flatten_time_dim_if_3d(test_data_tmp)
            
            score = NNAA(synthetic_data = sync_data_tmp if not args.privacy_with_real else train_data_tmp, train_data = train_data_tmp, eval_data = test_data_tmp)
            score, AA_test, AA_train = score()
            print(f"{args.metric} score: {score}, AA_test: {AA_test}, AA_train: {AA_train}")
            AA_test_arr.append(AA_test)
            AA_train_arr.append(AA_train)
            
        elif args.metric == "mir":
            
            if args.mir_model_name is not None:
                print(f"Model: {args.mir_model_name}")
                
            sync_data_tmp, train_data_tmp, test_data_tmp = impute_nan_with_neg_one(sync_data), impute_nan_with_neg_one(train_data), impute_nan_with_neg_one(test_data)
            sync_data_tmp, train_data_tmp, test_data_tmp = normalize_data(sync_data_tmp, axis=(0,2)), normalize_data(train_data_tmp, axis=(0,2)), normalize_data(test_data_tmp, axis=(0,2))
            sync_data_tmp, train_data_tmp, test_data_tmp = flatten_time_dim_if_3d(sync_data_tmp), flatten_time_dim_if_3d(train_data_tmp), flatten_time_dim_if_3d(test_data_tmp)
            print(sync_data_tmp.shape, train_data_tmp.shape, test_data_tmp.shape)
            score = MIR(synthetic_data = sync_data_tmp, train_data = train_data_tmp, eval_data = test_data_tmp, threshold = args.mir_threshold, privacy_with_real = args.privacy_with_real)
            score = score()
            print(f"{args.metric} score: {score}")
        
        elif args.metric == "trajectory":
            
            if args.data_name == "eicu":
                var_dict = {0: "heartrate", 1: "resprate", 2: "spo2", 3: "map"}
            elif args.data_name == "mimiciii":
                var_dict = {0: "heartrate", 1: "sbp", 2: "dbp", 3: "mbp", 4: "resprate", 5: "temp", 6: "spo2"}
            elif args.data_name == "mimiciv":
                var_dict = {0: "heartrate", 1: "sbp", 2: "dbp", 3: "resprate", 4: "spo2"}
            elif args.data_name == "hirid":
                var_dict = {0: "heartrate", 1: "sbp", 2: "dbp", 3: "map", 4: "spo2", 5: "st", 6: "cvp"}
                
            vis = TrajectoryVisualizer(
                sync_data = X_sync, real_data = X_eval,
                img_name = args.img_name, var_name_dict = var_dict,
                sample_size = 5000,
            )
            vis()
        
        elif args.metric == "nn_tstr/trtr":
            X_sync, y_sync, X_eval, y_eval = torch.tensor(impute_nan_with_neg_one(X_sync)), torch.tensor(y_sync), torch.tensor(impute_nan_with_neg_one(X_eval)).float(), torch.tensor(y_eval)
            X_train, y_train = torch.tensor(impute_nan_with_neg_one(X_train)), torch.tensor(y_train)
            
            sample_size = min(X_sync.shape[0], X_train.shape[0])            # ensure same sample size
            train_indices = np.random.choice(len(X_train), sample_size, replace=False)
            sync_indices = np.random.choice(len(X_sync), sample_size, replace=False)
            
            X_sync, X_train, y_sync, y_train = X_sync[sync_indices], X_train[train_indices], y_sync[sync_indices], y_train[train_indices]
            assert len(X_sync) == len(X_train)
            
            tstr_auc, trtr_auc = [], []
            train_num_steps = 5000
            bsz = 128
            
            for i in tqdm(range(1, 11), desc="TSTR..."):
                clf = nnTimeSeriesClassifier(model=RNN, X_train=X_sync, y_train=y_sync, bsz=bsz, train_num_steps=train_num_steps, wandb=None)
                clf.train()
                clf.model.to("cpu")         # move to cpu since X_eval could be huge
                clf.device = torch.device("cpu")
                y_proba = clf.predict_proba(X_eval.to("cpu"))
                auc = roc_auc_score(y_eval, y_proba)
                tstr_auc.append(auc)
            
            for i in tqdm(range(1, 11), desc="TRTR..."):
                clf = nnTimeSeriesClassifier(model=RNN, X_train=X_train, y_train=y_train, bsz=bsz, train_num_steps=train_num_steps, wandb=None)
                clf.train()
                clf.model.to("cpu")
                clf.device = torch.device("cpu")
                y_proba = clf.predict_proba(X_eval.to("cpu"))
                auc = roc_auc_score(y_eval, y_proba)
                trtr_auc.append(auc)
            
            print(f"===== COMPLETE =====\nTSTR AUC: {np.mean(tstr_auc):.3f} $\pm$ {np.std(tstr_auc):.3f}\nTRTR AUC: {np.mean(trtr_auc):.3f} $\pm$ {np.std(trtr_auc):.3f}")
            break
            
        elif args.metric == "tstr/trtr":
            if csv_flag:
                X_train_tmp, X_sync_tmp, X_eval_tmp = train_data.drop("hospital_expire_flag", axis=1), sync_data.drop("hospital_expire_flag", axis=1), test_data.drop("hospital_expire_flag", axis=1)
                y_train, y_sync, y_eval = train_data["hospital_expire_flag"], sync_data["hospital_expire_flag"], test_data["hospital_expire_flag"]
                imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=-1)
                X_train_tmp = imputer.fit_transform(X_train_tmp)
                X_sync_tmp, X_eval_tmp = imputer.transform(X_sync_tmp), imputer.transform(X_eval_tmp)
            else:
                X_train_tmp, X_sync_tmp, X_eval_tmp = flatten_time_dim_if_3d(X_train), flatten_time_dim_if_3d(X_sync), flatten_time_dim_if_3d(X_eval)
                X_train_tmp, X_sync_tmp, X_eval_tmp = impute_nan_with_neg_one(X_train_tmp), impute_nan_with_neg_one(X_sync_tmp), impute_nan_with_neg_one(X_eval_tmp)
            
            visual = TSTR_TRTR(
                X_sync = X_sync_tmp, y_sync = y_sync,
                X_train = X_train_tmp, y_train = y_train,
                X_eval = X_eval_tmp, y_eval = y_eval,
                reps = 10,
                img_name = args.img_name,
                metric = args.scoring_metric,
                legend = (args.data_name == "mimiciv"),
            )
            visual()
            break
        
        elif args.metric == "tsrtr":
            if csv_flag:
                X_train_tmp, X_sync_tmp, X_eval_tmp = train_data.drop("hospital_expire_flag", axis=1), sync_data.drop("hospital_expire_flag", axis=1), test_data.drop("hospital_expire_flag", axis=1)
                y_train, y_sync, y_eval = train_data["hospital_expire_flag"], sync_data["hospital_expire_flag"], test_data["hospital_expire_flag"]
                imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=-1)
                X_train_tmp = imputer.fit_transform(X_train_tmp)
                X_sync_tmp, X_eval_tmp = imputer.transform(X_sync_tmp), imputer.transform(X_eval_tmp)
            else:
                X_train_tmp, X_sync_tmp, X_eval_tmp = flatten_time_dim_if_3d(X_train), flatten_time_dim_if_3d(X_sync), flatten_time_dim_if_3d(X_eval)
                X_train_tmp, X_sync_tmp, X_eval_tmp = impute_nan_with_neg_one(X_train_tmp), impute_nan_with_neg_one(X_sync_tmp), impute_nan_with_neg_one(X_eval_tmp)
            
            visual = TSRTR(
                X_sync = X_sync_tmp, y_sync = y_sync,
                X_train = X_train_tmp, y_train = y_train,
                X_eval = X_eval_tmp, y_eval = y_eval,
                reps = 10,
                img_name = args.img_name,
                metric = args.scoring_metric,
                synthetic_percentage = [0.1, 0.3, 0.5, 0.7, 0.9],
                # synthetic_percentage = [0.1, 0.2 , 0.3, 0.4, 0.5],
                # legend = (args.data_name == "mimiciv"),
                legend = True,
                # real_train_size = len(X_train) if len(X_train) < 20000 else 20000, 
                real_train_size = 2000,
            )
            visual()
            break
        
        elif args.metric == "tsrtr_imbalance":
            X_train_tmp, X_sync_tmp, X_eval_tmp = flatten_time_dim_if_3d(X_train), flatten_time_dim_if_3d(X_sync), flatten_time_dim_if_3d(X_eval)
            X_train_tmp, X_sync_tmp, X_eval_tmp = impute_nan_with_neg_one(X_train_tmp), impute_nan_with_neg_one(X_sync_tmp), impute_nan_with_neg_one(X_eval_tmp)
            
            visual = TSRTR_Imbalance(
                X_sync = X_sync_tmp, y_sync = y_sync,
                X_train = X_train_tmp, y_train = y_train,
                X_eval = X_eval_tmp, y_eval = y_eval,
                reps = 10,
                img_name = args.img_name,
                imbalance_ratio = [0.3, 0.35, 0.4, 0.45, 0.5],
                real_sample_size = 2000,
                metric = args.scoring_metric,
            )
            visual()
            break
            
        elif args.metric == "t-sne":
            X_train, X_sync, X_eval = smart_to_numpy(X_train), smart_to_numpy(X_sync), smart_to_numpy(X_eval)
            X_train, X_sync, X_eval = impute_nan_with_neg_one(X_train), impute_nan_with_neg_one(X_sync), impute_nan_with_neg_one(X_eval)
            
            X_train, X_sync, X_eval = standardize_data(X_train, axis=(0,2)), standardize_data(X_sync, axis=(0,2)), standardize_data(X_eval, axis=(0,2))
            X_train, X_sync, X_eval = flatten_time_dim_if_3d(X_train), flatten_time_dim_if_3d(X_sync), flatten_time_dim_if_3d(X_eval)
            
            print(f"Dimensions of X_train: {X_train.shape}, X_sync: {X_sync.shape}, X_eval: {X_eval.shape}")
            
            visual = tSNE(
                X_train = X_train, X_sync = X_sync, X_eval = X_eval,
                perplexity = args.perplexity, n_iter = 300, learning_rate = 'auto', 
                sample_num=args.t_sne_num, n_components=2, save=True, img_name = args.img_name,
            )
            visual()
            break
        
        elif args.metric == "t-sne-tg":
            X_train, X_sync, X_eval = smart_to_numpy(X_train), smart_to_numpy(X_sync), smart_to_numpy(X_eval)
            X_train, X_sync, X_eval = impute_nan_with_neg_one(X_train), impute_nan_with_neg_one(X_sync), impute_nan_with_neg_one(X_eval)
            X_train, X_sync, X_eval = standardize_data(X_train), standardize_data(X_sync), standardize_data(X_eval)
            X_train, X_sync, X_eval = np.transpose(X_train, (0,2,1)), np.transpose(X_sync, (0,2,1)), np.transpose(X_eval, (0,2,1))
            print(X_train.shape, X_sync.shape, X_eval.shape)
            tSNE_Analysis(
                original_test=X_eval, synthetic=X_sync, original_train=X_train, 
                img_name = args.img_name, sample_num=args.t_sne_num
            )
            break
        
        elif args.metric == "discriminative":
            score = DiscriminativeScore(
                original_data = np.transpose(impute_nan_with_neg_one(X_eval), (0, 2, 1)), 
                synthetic_data = np.transpose(impute_nan_with_neg_one(X_sync), (0, 2, 1)),
                iterations = args.discriminative_iter,
                scaler = args.discri_pred_scaler,
                train_min = np.min(np.transpose(impute_nan_with_neg_one(X_train), (0, 2, 1)), axis=(0,1), keepdims=True),
                train_max = np.max(np.transpose(impute_nan_with_neg_one(X_train), (0, 2, 1)), axis=(0,1), keepdims=True),
            )           # exclude hospital mortality flag
            score = score()
            print(f"Discriminative Score: {score}")
        
        elif args.metric == "predictive":
            score = PredictiveScore(
                original_data = np.transpose(impute_nan_with_neg_one(X_eval), (0, 2, 1)), 
                synthetic_data = np.transpose(impute_nan_with_neg_one(X_sync), (0, 2, 1)),
                iterations = args.predictive_iter,
                scaler = args.discri_pred_scaler,
                train_min = np.min(np.transpose(impute_nan_with_neg_one(X_train), (0, 2, 1)), axis=(0,1), keepdims=True),
                train_max = np.max(np.transpose(impute_nan_with_neg_one(X_train), (0, 2, 1)), axis=(0,1), keepdims=True),
            )
            score = score()
            print(f"Predictive Score: {score}")
        
        elif args.metric == "discriminative-real":
            score = DiscriminativeScore(
                original_data = np.transpose(impute_nan_with_neg_one(X_eval), (0, 2, 1)), 
                synthetic_data = np.transpose(impute_nan_with_neg_one(X_train), (0, 2, 1)),
                iterations = args.discriminative_iter,
                scaler = args.discri_pred_scaler,
                train_min = np.min(np.transpose(impute_nan_with_neg_one(X_train), (0, 2, 1)), axis=(0,1), keepdims=True),
                train_max = np.max(np.transpose(impute_nan_with_neg_one(X_train), (0, 2, 1)), axis=(0,1), keepdims=True),
            )           # exclude hospital mortality flag
            score = score()
            print(f"Discriminative Score: {score}")
        
        elif args.metric == "predictive-real":
            score = PredictiveScore(
                original_data = np.transpose(impute_nan_with_neg_one(X_eval), (0, 2, 1)), 
                synthetic_data = np.transpose(impute_nan_with_neg_one(X_train), (0, 2, 1)),
                iterations = args.predictive_iter,
                scaler = args.discri_pred_scaler,
                train_min = np.min(np.transpose(impute_nan_with_neg_one(X_train), (0, 2, 1)), axis=(0,1), keepdims=True),
                train_max = np.max(np.transpose(impute_nan_with_neg_one(X_train), (0, 2, 1)), axis=(0,1), keepdims=True),
            )
            score = score()
            print(f"Predictive Score: {score}")
            
        else:
            raise ValueError(f"Unsupported metric: {args.metric}")
        
        score_arr.append(score)
    
    if "predictive" in args.metric or "discriminative" in args.metric or args.metric == "nnaa" or args.metric == "mir":
        if args.metric in ["predictive", "discriminative", "mir"]:
            print(f"Average score: {np.mean(score_arr):.3f} $\pm$ {np.std(score_arr):.3f}")
        elif args.metric == "nnaa":
            print(f"Average score: {np.mean(score_arr):.4f} $\pm$ {np.std(score_arr):.4f}\nAA_train: {np.mean(AA_train_arr):.4f} $\pm$ {np.std(AA_train_arr):.4f}\nAA_test: {np.mean(AA_test_arr):.4f} $\pm$ {np.std(AA_test_arr):.4f}")
            
    