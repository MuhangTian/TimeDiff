from typing import *

import numpy as np
from tqdm import tqdm


class NNAA:
    def __init__(self, synthetic_data, train_data, eval_data, batch_size: int=1000, n_draw: int=None) -> None:
        assert all(len(data.shape) == 2 for data in (synthetic_data, train_data, eval_data) ), "Data must be 2D."
        if eval_data.shape[1] > synthetic_data.shape[1]:
            eval_data, train_data = eval_data[:, :synthetic_data.shape[1]], train_data[:, :synthetic_data.shape[1]]
        assert synthetic_data.shape[1] == train_data.shape[1] == eval_data.shape[1], "Data must have the same number of features."
        assert not any(np.isnan(data).any() for data in (synthetic_data, train_data, eval_data)), "NaN detected in data!"
        self.synthetic_data = synthetic_data
        self.train_data = train_data
        self.eval_data = eval_data
            
        self.n_test = len(self.eval_data)
        self.n_train = len(self.train_data)
        self.batch_size = batch_size
        if self.n_train == self.n_test:
            self.n_draw = 1
        else:
            if n_draw is None:
                self.n_draw = np.ceil(self.n_train / self.n_test).astype(int)
            else:
                self.n_draw = n_draw
        self.n_row, self.n_col = self.synthetic_data.shape
        self.steps = np.ceil(self.n_test / self.batch_size).astype(int)
    
    def min_distance_diff(self, data1, data2) -> float:
        a = np.sum(data2 ** 2, axis=1).reshape(data2.shape[0], 1)
        b = np.sum(data1.T ** 2, axis=0)
        square_sum = a + b                          # a^2 + b^2
        two_ab = np.dot(data2, data1.T) * 2         # 2ab
        distance_matrix = square_sum - two_ab
        return np.min(distance_matrix, axis=0)
    
    def min_distance_same(self, data1, data2) -> float:
        a = np.sum(data2 ** 2, axis=1).reshape(data2.shape[0], 1)
        b = np.sum(data1.T ** 2, axis=0)
        square_sum = a + b
        two_ab = np.dot(data2, data1.T) * 2
        distance_matrix = square_sum - two_ab
        n_col = distance_matrix.shape[1]
        min_distance = np.zeros(n_col)
        for i in range(n_col):              # append all that is not zero (equal to itself)
            sorted_column = np.sort(distance_matrix[:, i])
            min_distance[i] = sorted_column[1]
        return min_distance
    
    def __call__(self) -> float:
        # training dataset
        distance_train_TS = np.zeros(self.n_test)
        distance_train_TT = np.zeros(self.n_test)
        distance_train_ST = np.zeros(self.n_test)
        distance_train_SS = np.zeros(self.n_test)
        AA_train = 0
        
        for ii in tqdm(range(self.n_draw), desc="Calculating AA_train..."):
            train_sample = np.random.permutation(self.train_data)[:self.n_test]
            fake_sample = np.random.permutation(self.synthetic_data)[:self.n_test]
            for i in range(self.steps):
                distance_train_TS[i * self.batch_size: (i + 1) * self.batch_size] = self.min_distance_diff(train_sample[i * self.batch_size: (i + 1) * self.batch_size], fake_sample)
                distance_train_ST[i * self.batch_size: (i + 1) * self.batch_size] = self.min_distance_diff(fake_sample[i * self.batch_size: (i + 1) * self.batch_size], train_sample)
                distance_train_TT[i * self.batch_size: (i + 1) * self.batch_size] = self.min_distance_same(train_sample[i * self.batch_size: (i + 1) * self.batch_size], train_sample)
                distance_train_SS[i * self.batch_size: (i + 1) * self.batch_size] = self.min_distance_same(fake_sample[i * self.batch_size: (i + 1) * self.batch_size], fake_sample)
                assert not any(np.isnan(data).any() for data in (distance_train_SS, distance_train_ST, distance_train_TS, distance_train_TT)), "NaN detected in data!"
            AA_train += (np.sum(distance_train_TS > distance_train_TT) + np.sum(distance_train_ST > distance_train_SS)) / self.n_test / 2

        AA_train /= self.n_draw

        # test dataset
        distance_test_TS = np.zeros(self.n_test)
        distance_test_TT = np.zeros(self.n_test)
        distance_test_ST = np.zeros(self.n_test)
        distance_test_SS = np.zeros(self.n_test)
        AA_test = 0
        
        for ii in tqdm(range(self.n_draw), desc="Calculating AA_test..."):
            fake_sample = np.random.permutation(self.synthetic_data)[:self.n_test]
            for i in range(self.steps):
                distance_test_TS[i * self.batch_size: (i + 1) * self.batch_size] = self.min_distance_diff(self.eval_data[i * self.batch_size: (i + 1) * self.batch_size], fake_sample)
                distance_test_ST[i * self.batch_size: (i + 1) * self.batch_size] = self.min_distance_diff(fake_sample[i * self.batch_size: (i + 1) * self.batch_size], self.eval_data)
                distance_test_TT[i * self.batch_size: (i + 1) * self.batch_size] = self.min_distance_same(self.eval_data[i * self.batch_size: (i + 1) * self.batch_size], self.eval_data)
                distance_test_SS[i * self.batch_size: (i + 1) * self.batch_size] = self.min_distance_same(fake_sample[i * self.batch_size: (i + 1) * self.batch_size], fake_sample)
                assert not any(np.isnan(data).any() for data in (distance_test_SS, distance_test_ST, distance_test_TS, distance_test_TT)), "NaN detected in data!"
            AA_test += (np.sum(distance_test_TS > distance_test_TT) + np.sum(distance_test_ST > distance_test_SS)) / self.n_test / 2

        AA_test /= self.n_draw

        privacy_loss = AA_test - AA_train
        return np.abs(privacy_loss), AA_test, AA_train


class MIR:
    def __init__(self, synthetic_data, train_data, eval_data, threshold, batch_size: int=1000, privacy_with_real: bool=False) -> None:
        self.eval_data = eval_data
        self.train_data = self.sample(train_data)
        self.batch_size = batch_size
        self.synthetic_data = self.sample(synthetic_data)
        self.threshold = threshold
        self.privacy_with_real = privacy_with_real
    
    def sample(self, X, size=None):
        if size is None:
            indices = np.random.choice(len(X), len(self.eval_data), replace=False)
        else:
            indices = np.random.choice(len(X), size, replace=False)
        return X[indices]
    
    def normalize_dist(self, dist):
        return (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
    
    def find_replicant(self, real, fake):
        a = np.sum(fake ** 2, axis=1).reshape(fake.shape[0], 1) + np.sum(real.T ** 2, axis=0)
        b = np.dot(fake, real.T) * 2
        distance_matrix = a - b
        return np.min(distance_matrix, axis=0)
    
    def __call__(self) -> float:
        distance_train = np.zeros(len(self.train_data))
        distance_test = np.zeros(len(self.eval_data))
        
        if not self.privacy_with_real:
            steps = np.ceil(len(self.train_data) / self.batch_size)
            for i in range(int(steps)):
                distance_train[i * self.batch_size:(i + 1) * self.batch_size] = self.find_replicant(self.train_data[i * self.batch_size:(i + 1) * self.batch_size], self.synthetic_data)
            distance_train = self.normalize_dist(distance_train)

        steps = np.ceil(len(self.eval_data) / self.batch_size)
        for i in range(int(steps)):
            distance_test[i * self.batch_size:(i + 1) * self.batch_size] = self.find_replicant(self.eval_data[i * self.batch_size:(i + 1) * self.batch_size], self.synthetic_data)
            
        distance_test = self.normalize_dist(distance_test)

        n_tp = np.sum(distance_train <= self.threshold)  # true positive counts
        n_fn = len(self.train_data) - n_tp
        n_fp = np.sum(distance_test <= self.threshold)  # false positive counts
        f1 = n_tp / (n_tp + (n_fp + n_fn) / 2)  # F1 score
        return f1