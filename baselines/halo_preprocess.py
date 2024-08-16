"""
To preprocess continuous time series for hierarchical autoregressive language model (HALO)
Performs the following steps:
    1. Use quantiles for each feature to discretize continuous values
    2. Convert continuous data into one-hot, binary encoding (if feature is missing, set missing index to 1)
    3. Add start and end tokens
    4. Save the processed data
"""
import torch
import numpy as np
from tqdm import tqdm
import argparse

COLUMNS_DICT = {
    "mimiciv": {"numerical": [0, 2, 4, 6, 8], "categorical": [1, 3, 5, 7, 9, 10], "categorical_num_classes": [2, 2, 2, 2, 2, 2]},
    "mimiciii": {"numerical": [0, 2, 4, 6, 8, 10, 12], "categorical": [1, 3, 5, 7, 9, 11, 13, 14], "categorical_num_classes": [2, 2, 2, 2, 2, 2, 2, 2]},
    "eicu": {"numerical": [0, 2, 4, 6], "categorical": [1, 3, 5, 7, 8], "categorical_num_classes": [2, 2, 2, 2, 2]},
    "hirid": {"numerical": [0, 1, 2, 3, 4, 5, 6], "categorical": [7], "categorical_num_classes": [2]},
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess continuous time series for HALO')
    parser.add_argument('--data', type=str, default="mimiciv_train.pt", help='Path to the data file')
    parser.add_argument('--dataset', type=str, default="mimiciv", help='Dataset name (mimiciv, mimiciii, eicu, hirid)')
    return parser.parse_args()

def assert_zeros(data: np.ndarray):
    assert np.all(data == 0), "Data is not zeroed out"

class HALOPreprocessor:
    def __init__(self, data: np.ndarray, args: argparse.Namespace=None):
        assert isinstance(data, np.ndarray), "data must be a numpy array"

        self.args = args
        self.data = data        # assumes data is of dimension (num_samples, num_features, seq_length)
        self.continuous_vocab_size = 101*self.data.shape[1]+2 # 101 * num_features + 2, +2 for end and start tokens
        self.label_vocab_size = 1

        if self.is_stock_energy:
            self.prep_data = np.zeros((
                len(data),      # sample size
                self.data.shape[2] + 2,  # +2 because of start and end vector
                self.continuous_vocab_size,
            ))
            self.prep_mask = np.zeros((
                len(data),
                self.data.shape[2] + 2,
                1,
            ))
        elif self.is_eicu_mimic_hirid:
            self.prep_data = np.zeros((
                len(data),      # sample size
                self.data.shape[2] + 3,   # +3 because of start, label, and end vector
                self.continuous_vocab_size + 1,    # +1 again for label token (we only have one label for mortality)
            ))
            self.prep_mask = np.zeros((
                len(data),
                self.data.shape[2] + 3,
                1,
            ))
        else:
            raise ValueError(f"Dataset '{self.args.dataset}' is not supported")
        
        self.find_bins_per_feature()
    
    @property
    def is_stock_energy(self):
        return self.args.dataset in ["stocks", "energy"]
    
    @property
    def is_eicu_mimic_hirid(self):
        return self.args.dataset in ["eicu", "mimiciv", "mimiciii", "hirid"]

    def find_bins_per_feature(self):
        """
        Find the bins for each feature using quantiles
        """
        self.bins = []
        for i in range(self.data.shape[1]):
            nonnan_data = self.data[:, i, :][~np.isnan(self.data[:, i, :])]
            quantiles = np.quantile(nonnan_data, np.linspace(0, 1, 101))
            self.bins.append([(quantiles[i], quantiles[i+1]) for i in range(100)])
    
    def find_index_per_feature_value(self, feature_idx: int, value: float):
        """
        Find the index for the value of a feature
        """
        for i in range(100):
            if value >= self.bins[feature_idx][i][0] and value <= self.bins[feature_idx][i][1]:
                return i
        raise ValueError(f"Value {value} not in bins for feature {feature_idx}")

    def identify_mortality(self, data_array: np.ndarray):
        if np.all(data_array.astype(int) == 0):
            return 0
        elif np.all(data_array.astype(int) == 1):
            return 1
        else:
            raise ValueError("Mortality label must be binary")
    
    def preprocess(self):
        """
        Preprocess continuous data into one-hot, binary encoding
        """
        if self.is_stock_energy:
            self.prep_data[:, 0, -2] = 1       # start token for all samples
            for i in tqdm(range(len(self.data))):

                self.prep_data[i, -1, -1] = 1 # Set the final sequence to have the end token

                for j in range(self.data.shape[2]):
                    self.prep_mask[i, j+1] = 1

                    for k in range(self.data.shape[1]):
                        if np.isnan(self.data[i, k, j]):
                            missing_idx = -1 + 101*(k+1)
                            assert_zeros(self.prep_data[i, j+1, missing_idx])
                            self.prep_data[i, j+1, missing_idx] = 1
                        else:
                            feature_idx = self.find_index_per_feature_value(feature_idx=k, value=self.data[i, k, j])
                            assert_zeros(self.prep_data[i, j+1, feature_idx + 101*k])
                            self.prep_data[i, j+1, feature_idx + 101*k] = 1

            self.prep_mask = self.prep_mask[:,1:,:]

        elif self.is_eicu_mimic_hirid:
            self.prep_data[:, 0, -2] = 1        # start token for all patients
            for i in tqdm(range(len(self.data))):

                mortality_flag = self.identify_mortality(self.data[i, -1, :])
                self.prep_data[i, 1, -3] = mortality_flag       # label vector, 1 for mortality, 0 for survival, -3 is the index for label token
                self.prep_data[i, -1, -1] = 1 # Set the final visit to have the end token

                for j in range(self.data.shape[2]):
                    self.prep_mask[i, j+2] = 1

                    for k in range(self.data.shape[1] - 1):     # minus one because the last channel is the label
                        if np.isnan(self.data[i, k, j]):        # use onehot representation for feature value
                            missing_idx = -1 + 101*(k+1)
                            assert_zeros(self.prep_data[i, j+2, missing_idx])
                            self.prep_data[i, j+2, missing_idx] = 1
                        else:
                            feature_idx = self.find_index_per_feature_value(feature_idx=k, value=self.data[i, k, j])
                            assert_zeros(self.prep_data[i, j+2, feature_idx + 101*k])
                            self.prep_data[i, j+2, feature_idx + 101*k] = 1
            
            self.prep_mask[:,1] = 1 # Set the mask to cover the labels
            self.prep_mask = self.prep_mask[:,1:,:]

        else:
            raise ValueError(f"Dataset '{self.args.dataset}' is not supported")
    
        return self.prep_data, self.prep_mask

if __name__ == "__main__":
    args = parse_arguments()

    if args.dataset in ["stocks", "energy"]:
        data = torch.load(args.data).numpy()
    elif args.dataset in ["mimiciv", "mimiciii", "eicu", "hirid"]:
        features = [*COLUMNS_DICT[args.dataset]["numerical"], -1]
        data = torch.load(args.data)[:, features, :].numpy()
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported")
    
    preprocessor = HALOPreprocessor(data, args)
    prep_data, prep_mask = preprocessor.preprocess()
    print(f"Preprocessed data shape: {prep_data.shape}\nPreprocessed mask shape: {prep_mask.shape}")
    prep_data, prep_mask = torch.tensor(prep_data).to(torch.float32), torch.tensor(prep_mask).to(torch.float32)
    torch.save(prep_data, f"data/halo/{args.dataset}_halo_processed.pt")
    torch.save(prep_mask, f"data/halo/{args.dataset}_halo_mask.pt")

    print(f"Data saved as 'data/halo/{args.dataset}_halo_processed.pt' and 'data/halo/{args.dataset}_halo_mask.pt'")