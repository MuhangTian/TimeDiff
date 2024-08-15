'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

PCA and TSNE analysis between Original data and Synthetic data
Inputs: 
  - dataX: original data
  - dataX_hat: synthetic data
  
Outputs:
  - PCA Analysis Results
  - t-SNE Analysis Results

'''
#%% Necessary Packages

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch

#%% PCA Analysis
    
def PCA_Analysis (dataX, dataX_hat, name):
  
    # Analysis Data Size
    Sample_No = 1000
    
    # Data Preprocessing
    for i in range(Sample_No):
        if (i == 0):
            arrayX = np.reshape(np.mean(np.asarray(dataX[0]),1), [1,len(dataX[0][:,0])])
            arrayX_hat = np.reshape(np.mean(np.asarray(dataX_hat[0]),1), [1,len(dataX[0][:,0])])
        else:
            arrayX = np.concatenate((arrayX, np.reshape(np.mean(np.asarray(dataX[i]),1), [1,len(dataX[0][:,0])])))
            arrayX_hat = np.concatenate((arrayX_hat, np.reshape(np.mean(np.asarray(dataX_hat[i]),1), [1,len(dataX[0][:,0])])))
    
    # Parameters        
    No = len(arrayX[:,0])
    colors = ["red" for i in range(No)] +  ["blue" for i in range(No)]    
    
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(arrayX)
    pca_results = pca.transform(arrayX)
    pca_hat_results = pca.transform(arrayX_hat)
        
    # Plotting
    f, ax = plt.subplots(1)
    
    plt.scatter(pca_results[:,0], pca_results[:,1], c = colors[:No], alpha = 0.2, label = "Original")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], c = colors[No:], alpha = 0.2, label = "Synthetic")

    ax.legend()
    
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.show()
    
    
#%% TSNE Analysis
    
def tSNE_Analysis (dataX, dataX_hat):
  
    # Analysis Data Size
    Sample_No = 1000
  
    # Preprocess
    for i in range(Sample_No):
        if (i == 0):
            arrayX = np.reshape(np.mean(np.asarray(dataX[0]),1), [1,len(dataX[0][:,0])])
            arrayX_hat = np.reshape(np.mean(np.asarray(dataX_hat[0]),1), [1,len(dataX[0][:,0])])
        else:
            arrayX = np.concatenate((arrayX, np.reshape(np.mean(np.asarray(dataX[i]),1), [1,len(dataX[0][:,0])])))
            arrayX_hat = np.concatenate((arrayX_hat, np.reshape(np.mean(np.asarray(dataX_hat[i]),1), [1,len(dataX[0][:,0])])))
     
    # Do t-SNE Analysis together       
    final_arrayX = np.concatenate((arrayX, arrayX_hat), axis = 0)
    
    # Parameters
    No = len(arrayX[:,0])
    colors = ["red" for i in range(No)] +  ["blue" for i in range(No)]    
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(final_arrayX)
    
    # Plotting
    f, ax = plt.subplots(1)
    
    plt.scatter(tsne_results[:No,0], tsne_results[:No,1], c = colors[:No], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[No:,0], tsne_results[No:,1], c = colors[No:], alpha = 0.2, label = "Synthetic")

    ax.legend()
    
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.savefig("t-sne.png", dpi=600)
    print("Saved as t-sne.png")
    # plt.show()

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

def regular_transform_to_features(samples):
    """for regular data type in eICU, transform to features"""
    assert isinstance(samples, np.ndarray), "samples must be a numpy array"
    unmasked_samples = np.zeros((samples.shape[0], 5, samples.shape[2]))
    for i in range(0, 7, 2):
        nonan_indicator = np.round(reverse_to_nonan_indicator_2d(samples[:,i+1,:]))
        data = np.where(nonan_indicator == 1, samples[:,i,:], np.nan)
        unmasked_samples[:, int(i/2), :] = data
    unmasked_samples[:, 4, :] = np.round(samples[:, -1, :])         # label channel
    return unmasked_samples

def reverse_to_nonan_indicator_2d(cum_nonan_indicator):        # Reverse cumulative sum using PyTorch
    nonan_indicator = np.concatenate([cum_nonan_indicator[:, :1], cum_nonan_indicator[:, 1:] - cum_nonan_indicator[:, :-1]], axis=1)
    return nonan_indicator

def flatten_time_dim_if_3d(data) -> np.ndarray:
    assert isinstance(data, np.ndarray), "must be numpy array!"
    if len(data.shape) == 3:
        data = np.transpose(data, (0, 2, 1)).reshape(len(data), -1)
    return data

def impute_nan_with_neg_one(data) -> np.ndarray:
    assert isinstance(data, np.ndarray), "must be numpy array!"
    return np.where(np.isnan(data), -1, data)

if __name__ == "__main__":
    sync_path = "results/samples/etdiff_regular_24hrs_256_2_rms_1_samples.npy"
    train_path = "data/eicu-extract/TRAIN_regular_all_patients_60_1440_276.pt"
    test_path = "data/eicu-extract/TEST_regular_all_patients_60_1440_276.pt"
    sync_data, train_data, test_data = smart_load(sync_path), smart_load(train_path), smart_load(test_path)
    sync_data, train_data, test_data = smart_to_numpy(sync_data), smart_to_numpy(train_data), smart_to_numpy(test_data)
    X_sync, X_train, X_eval = regular_transform_to_features(sync_data)[:, :-1, :], regular_transform_to_features(train_data)[:, :-1, :sync_data.shape[-1]], regular_transform_to_features(test_data)[:, :-1, :sync_data.shape[-1]]
    # X_train, X_sync, X_eval = flatten_time_dim_if_3d(X_train), flatten_time_dim_if_3d(X_sync), flatten_time_dim_if_3d(X_eval)
    X_train, X_sync, X_eval = impute_nan_with_neg_one(X_train), impute_nan_with_neg_one(X_sync), impute_nan_with_neg_one(X_eval)
    X_train, X_sync, X_eval = np.transpose(X_train, (0,2,1)), np.transpose(X_sync, (0,2,1)), np.transpose(X_eval, (0,2,1))
    print(X_train.shape, X_sync.shape, X_eval.shape)
    tSNE_Analysis(X_eval, X_sync)