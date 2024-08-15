import random
import warnings
from copy import deepcopy
from typing import *
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, auc, brier_score_loss,
                             mean_absolute_error, precision_recall_curve,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

# import evaluate.utils as utils

def tSNE_Analysis (original_train, original_test, synthetic, sample_num, img_name="t-sne-ts.png"):
    """from TimeGAN paper"""
    # Analysis Data Size
    dataX, dataX_hat = original_train, synthetic
    Sample_No = sample_num
  
    # Preprocess
    for i in range(Sample_No):
        if (i == 0):
            arrayX = np.reshape(np.mean(np.asarray(dataX[0]),1), [1,len(dataX[0][:,0])])
            arrayX_hat = np.reshape(np.mean(np.asarray(dataX_hat[0]),1), [1,len(dataX[0][:,0])])
            arrayX_test = np.reshape(np.mean(np.asarray(original_test[0]),1), [1,len(dataX[0][:,0])])
        else:
            arrayX = np.concatenate((arrayX, np.reshape(np.mean(np.asarray(dataX[i]),1), [1,len(dataX[0][:,0])])))
            arrayX_hat = np.concatenate((arrayX_hat, np.reshape(np.mean(np.asarray(dataX_hat[i]),1), [1,len(dataX[0][:,0])])))
            arrayX_test = np.concatenate((arrayX_test, np.reshape(np.mean(np.asarray(original_test[i]),1), [1,len(dataX[0][:,0])])))
     
    # Do t-SNE Analysis together       
    final_arrayX = np.concatenate((arrayX, arrayX_test, arrayX_hat), axis = 0)
    
    # Parameters
    No = len(arrayX[:,0])
    colors = ["red" for i in range(No)] + ["orange" for i in range(No)] + ["blue" for i in range(No)]
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(final_arrayX)
    
    # Plotting
    f, ax = plt.subplots(1)
    
    plt.scatter(tsne_results[:No,0], tsne_results[:No,1], c = colors[:No], alpha = 0.2, label = "Original Training")
    plt.scatter(tsne_results[No:2*No,0], tsne_results[No:2*No,1], c = colors[No:2*No], alpha = 0.2, label = "Original Testing")
    plt.scatter(tsne_results[2*No:,0], tsne_results[2*No:,1], c = colors[2*No:], alpha = 0.2, label = "Synthetic")

    ax.legend()
    
    # plt.title('t-SNE plot')
    # plt.xlabel('x-tsne')
    # plt.ylabel('y_tsne')
    plt.savefig(img_name, dpi=600)
    print(f"Saved as {img_name}")


def adjust_prob(y_prob) -> np.ndarray:
    try:
        if len(y_prob.shape) == 2:        # for some scikit-learn models where probas is 2D
            if y_prob.shape[1] == 1:
                return y_prob
            else:
                return y_prob[:, 1]
    except AttributeError:
        pass
    return y_prob

def get_metric(y_true, y_prob, metric) -> float:
    if metric == 'auc':
        metric = roc_auc_score(y_true, y_prob)
    elif metric == 'auprc':
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        metric = auc(recall, precision)
    elif metric == 'brier':
        metric = brier_score_loss(y_true, y_prob)
    return metric

def standardize_data(data, eps=1e-8):
    """standardize along the feature (channel) dimension""" 
    mean = data.mean(axis=(0, 1), keepdims=True)
    std = data.std(axis=(0, 1), keepdims=True)
    data = (data - mean) / (std + eps)
    return data

def normalize_data(data, min=None, max=None, eps=1e-8):
    """normalize along the feature (channel) dimension"""
    if min is None and max is None:
        min = data.min(axis=(0, 1), keepdims=True)
        max = data.max(axis=(0, 1), keepdims=True)
    else:
        print("Using given min and max values for normalization...")
    data = (data - min) / (max - min + eps)
    # assert (data >= 0).all() and (data <= 1).all(), "data not normalized!"
    return data
    

class BaseUtility:
    def __init__(self, clf,  X_sync: pd.DataFrame, y_sync: pd.Series, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series) -> None:
        self.clf = clf
        self.X_sync = X_sync
        self.y_sync = y_sync
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval

class TSTR(BaseUtility):
    """Train on synthetic, test on real"""
    def __init__(self, clf, X_sync: pd.DataFrame, y_sync: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series, metric = 'auc') -> None:
        super().__init__(clf, X_sync, y_sync, None, None, X_eval, y_eval)
        self.metric = metric
    
    def __call__(self) -> float:
        self.clf.fit(self.X_sync, self.y_sync)
        y_prob = adjust_prob(self.clf.predict_proba(self.X_eval))
        return get_metric(self.y_eval, y_prob, self.metric)

class TRTS(BaseUtility):
    """Train on real, test on synthetic"""
    def __init__(self, clf, X_sync: pd.DataFrame, y_sync: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series, metric = 'auc') -> None:
        super().__init__(clf, X_sync, y_sync, None, None, X_eval, y_eval)
        self.metric = metric
    
    def __call__(self) -> float:
        self.clf.fit(self.X_eval, self.y_eval)
        y_prob = adjust_prob(self.clf.predict_proba(self.X_sync) )
        return get_metric(self.y_sync, y_prob, self.metric)

class TRTR(BaseUtility):
    """Train on real, test on real"""
    def __init__(self, clf, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series, metric = 'auc') -> None:
        super().__init__(clf, None, None, X_train, y_train, X_eval, y_eval)
        self.metric = metric
    
    def __call__(self) -> float:
        self.clf.fit(self.X_train, self.y_train)
        y_prob = adjust_prob(self.clf.predict_proba(self.X_eval))
        return get_metric(self.y_eval, y_prob, self.metric)

class TSTR_TRTR(BaseUtility):
    def __init__(self, X_train, y_train, X_eval, y_eval, X_sync, y_sync, reps, img_name, metric = "auc", legend: bool = True):
        super().__init__(None, X_sync, y_sync, X_train, y_train, X_eval, y_eval)
        self.metric = metric
        self.sample_size = len(X_eval)
        print(f"Sampling at sample size: {self.sample_size}")
        self.reps = reps
        self.img_name = img_name
        self.legend = legend
    
    def preprocess(self, train_data, test_data):
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        return train_data, test_data
    
    def get_score(self, clf, X_train, y_train, X_test, y_test):
        assert len(np.unique(y_train)) == 2 and len(np.unique(y_test)) == 2, "Only binary classification is supported!"

        X_train, X_test = self.preprocess(X_train, X_test)
        clf.fit(X_train, y_train)
        y_prob = adjust_prob(clf.predict_proba(X_test))
        return get_metric(y_test, y_prob, self.metric)

    def sample(self, X, y):
        assert len(X) == len(y)
        
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=self.sample_size / len(X))
        for train_index, sample_index in stratified_split.split(X, y):
            X_sampled, y_sampled = X[sample_index], y[sample_index]
        
        return X_sampled, y_sampled
        
    def __call__(self):
        xgboost = XGBClassifier()
        rf = RandomForestClassifier()
        ada = AdaBoostClassifier()
        log_l1 = LogisticRegression(penalty='l1', solver='liblinear')
        log_l2 = LogisticRegression(penalty='l2', solver='liblinear')
        
        tstr_means, tstr_stds, trtr_means, trtr_stds = [], [], [], []
        for clf in [xgboost, rf, ada, log_l1, log_l2]:
            print(f"\n*** Model: {clf.__class__.__name__} ***")
            tstr_arr, trtr_arr = [], []
            for _ in range(self.reps):
                try:
                    X_sync, y_sync = self.sample(self.X_sync, self.y_sync)
                except:     # for the case where y only has one class
                    tstr_score = 0

                X_train, y_train = self.sample(self.X_train, self.y_train)
                # X_eval, y_eval = self.sample(self.X_eval, self.y_eval)    # no need to sample since the size is the same
                
                independent_clf = deepcopy(clf)     # make a copy for trtr evaluation
                try:
                    tstr_score = self.get_score(clf, X_sync, y_sync, self.X_eval, self.y_eval)
                except:     # for the case where y only has one class
                    tstr_score = 0
                trtr_score = self.get_score(independent_clf, X_train, y_train, self.X_eval, self.y_eval)
                print(f"TSTR: {tstr_score:.3f}, TRTR: {trtr_score:.3f}")
                
                tstr_arr.append(tstr_score)
                trtr_arr.append(trtr_score)
            
            tstr_mean, tstr_std = np.mean(tstr_arr), np.std(tstr_arr)
            trtr_mean, trtr_std = np.mean(trtr_arr), np.std(trtr_arr)
            
            tstr_means.append(tstr_mean)
            tstr_stds.append(tstr_std)
            trtr_means.append(trtr_mean)
            trtr_stds.append(trtr_std)
        
        model_names = ['XGB', 'RF', 'AB', 'LR L1', 'LR L2']
        x = np.arange(len(model_names))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        palette = sns.color_palette()
        sns.set_theme(style="ticks")
        ax.bar(x - width/2, tstr_means, width, label='TSTR', yerr=tstr_stds, capsize=5, color=palette[0])
        ax.bar(x + width/2, trtr_means, width, label='TRTR', yerr=trtr_stds, capsize=5, color=palette[1])

        ax.set_ylabel(self.metric.upper(), fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=18)
        ax.tick_params(axis='y', labelsize=13)
        if self.legend:
            ax.legend(loc="lower right", fontsize="large")
        sns.despine()

        fig.tight_layout()

        if self.img_name.split(".")[-1] == "pdf":
            plt.savefig(self.img_name, format="pdf", dpi=500)
        else:
            plt.savefig(self.img_name, dpi=500)
        print(f"Saved as {self.img_name}")


class TSRTR(TSTR_TRTR):
    def __init__(
        self, X_train, y_train, X_eval, y_eval, X_sync, y_sync, 
        reps, img_name, synthetic_percentage, metric = "auc",
        real_train_size: int = 2000,
        legend: bool = True,
        ):
        super().__init__(X_train, y_train, X_eval, y_eval, X_sync, y_sync, reps, img_name, metric, legend)
        self.synthetic_percentage = synthetic_percentage
        self.real_train_size = real_train_size
    
    def get_size(self, percent):
        return int( (percent * self.real_train_size) / (1 - percent) )
    
    def sample(self, X, y, size=None):
        assert len(X) == len(y)
        if size is None:
            indices = np.random.choice(len(X), self.sample_size, replace=False)
        else:
            indices = np.random.choice(len(X), size, replace=False)
        return X[indices], y[indices]
    
    def __call__(self):
        xgboost = XGBClassifier()
        rf = RandomForestClassifier()
        ada = AdaBoostClassifier()
        log_l1 = LogisticRegression(penalty='l1', solver='liblinear')
        log_l1.name = "Logistic Regression (L1)"
        log_l2 = LogisticRegression(penalty='l2', solver='liblinear')
        log_l2.name = "Logistic Regression (L2)"

        mean_dict, std_dict = {}, {}
        
        for clf in [xgboost, rf, ada, log_l1, log_l2]:
            print(f"\n*** Model: {clf.__class__.__name__} ***")
            for percent in self.synthetic_percentage:
                tsrtr_arr = []
                for _ in range(self.reps):                    
                    synthetic_size = self.get_size(percent)
                    
                    X_sync, y_sync = self.sample(self.X_sync, self.y_sync, synthetic_size)
                    X_train, y_train = self.sample(self.X_train, self.y_train, self.real_train_size)
                    X_eval, y_eval = self.sample(self.X_eval, self.y_eval)
                    
                    X_train = np.concatenate((X_train, X_sync))
                    y_train = np.concatenate((y_train, y_sync))
                    
                    tsrtr_score = self.get_score(clf, X_train, y_train, X_eval, y_eval)
                    tsrtr_arr.append(tsrtr_score)
                tsrtr_mean, tsrtr_std = np.mean(tsrtr_arr), np.std(tsrtr_arr)
                print(f"PERCENT = {percent} | TSRTR: {tsrtr_mean:.3f} ± {tsrtr_std:.3f}")
                
                clf_name = getattr(clf, "name", clf.__class__.__name__)
                
                try:
                    mean_dict[clf_name].append(tsrtr_mean)
                    std_dict[clf_name].append(tsrtr_std)
                except:
                    mean_dict[clf_name] = [tsrtr_mean]
                    std_dict[clf_name] = [tsrtr_std]
        
        sns.set_theme(style="ticks")
        palette = sns.color_palette("tab10")
        percentages = self.synthetic_percentage

        plt.figure()

        model_names = ['XGB', 'RF', 'AB', 'LR L1', 'LR L2']
        for i, (clf_name, means) in enumerate(mean_dict.items()):
            stds = std_dict[clf_name]
            plt.errorbar(percentages, means, yerr=stds, color=palette[i], marker='o', label=model_names[i], linestyle='-', alpha=0.7)

        plt.xlabel('Synthetic Percentage', fontsize=18)
        plt.ylabel(self.metric.upper(), fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        if self.legend:
            plt.legend(loc="lower right", fontsize="large")
        sns.despine()
        plt.tight_layout()
        
        if self.img_name.split(".")[-1] == "pdf":
            plt.savefig(self.img_name, format="pdf", dpi=500)
        else:
            plt.savefig(self.img_name, dpi=500)
        print(f"Saved as {self.img_name}")


class TSRTR_Imbalance(TSTR_TRTR):
    def __init__(
        self, X_train, y_train, X_eval, y_eval, X_sync, y_sync, 
        reps, img_name, imbalance_ratio, real_sample_size, metric = "auc",
        ):
        super().__init__(X_train, y_train, X_eval, y_eval, X_sync, y_sync, reps, img_name, metric)
        self.imbalance_ratio = imbalance_ratio
        self.real_sample_size = real_sample_size
    
    def sample(self, X, y, size=None):
        assert len(X) == len(y)
        if size is None:
            indices = np.random.choice(len(X), self.sample_size, replace=False)
        else:
            indices = np.random.choice(len(X), size, replace=False)
        return X[indices], y[indices]
        
    def get_class_ratio(self, y):
        return len(y[y==1]) / len(y)

    def get_size(self, y, desired_ratio):
        return int((len(y[y == 1]) - desired_ratio * len(y)) / (desired_ratio - 1))
    
    def __call__(self):
        xgboost = XGBClassifier()
        rf = RandomForestClassifier()
        ada = AdaBoostClassifier()
        log_l1 = LogisticRegression(penalty='l1', solver='liblinear')
        log_l1.name = "Logistic Regression (L1)"
        log_l2 = LogisticRegression(penalty='l2', solver='liblinear')
        log_l2.name = "Logistic Regression (L2)"

        mean_dict, std_dict = {}, {}
        
        for clf in [xgboost, rf, ada, log_l1, log_l2]:
            print(f"\n*** Model: {clf.__class__.__name__} ***")
            for ratio in self.imbalance_ratio:
                tsrtr_arr = []
                for _ in range(self.reps):
                    X_train, y_train = self.sample(self.X_train, self.y_train, self.real_sample_size)    
                    current_ratio = self.get_class_ratio(y_train)             
                    desired_size = self.get_size(y_train, ratio)
                    
                    X_sync, y_sync = self.sample(self.X_sync, self.y_sync, desired_size)
                    current_ratio = self.get_class_ratio(y_sync)
                    X_eval, y_eval = self.sample(self.X_eval, self.y_eval)
                    
                    X_train = np.concatenate((X_train, X_sync))
                    y_train = np.concatenate((y_train, y_sync))
                    
                    current_ratio = self.get_class_ratio(y_train)

                    tsrtr_score = self.get_score(clf, X_train, y_train, X_eval, y_eval)
                    tsrtr_arr.append(tsrtr_score)
                tsrtr_mean, tsrtr_std = np.mean(tsrtr_arr), np.std(tsrtr_arr)
                print(f"RATIO = {ratio} : {current_ratio:.1f} | TSRTR: {tsrtr_mean:.3f} ± {tsrtr_std:.3f}")
                
                clf_name = getattr(clf, "name", clf.__class__.__name__)
                
                try:
                    mean_dict[clf_name].append(tsrtr_mean)
                    std_dict[clf_name].append(tsrtr_std)
                except:
                    mean_dict[clf_name] = [tsrtr_mean]
                    std_dict[clf_name] = [tsrtr_std]
        
        sns.set_theme(style="ticks")
        palette = sns.color_palette()
        percentages = self.imbalance_ratio

        plt.figure()

        model_names = ['XGB', 'RF', 'AB', 'LR L1', 'LR L2']
        for i, (clf_name, means) in enumerate(mean_dict.items()):
            stds = std_dict[clf_name]
            plt.errorbar(percentages, means, yerr=stds, color=palette[i], marker='o', label=model_names[i], linestyle='-')

        plt.xlabel('Class Ratio', fontsize=15)
        plt.ylabel(self.metric.upper(), fontsize=15)
        plt.legend()
        sns.despine()
        plt.tight_layout()
        
        if self.img_name.split(".")[-1] == "pdf":
            plt.savefig(self.img_name, format="pdf", dpi=500)
        else:
            plt.savefig(self.img_name, dpi=500)
        print(f"Saved as {self.img_name}")


class tSNE(BaseUtility):
    """
    Visualize t-SNE dimension reduction plot for qualitative evaluation.
    See https://distill.pub/2016/misread-tsne/ for usage of t-SNE and effects of hyper-parameters.
    """
    def __init__(self, X_train, X_sync, X_eval, sample_num=1000, perplexity=30, n_iter=5000, learning_rate='auto', n_jobs=-1, n_components=2, save=False, img_name="t-sne.png") -> None:
        self.sample_num = sample_num
        X_train, X_sync, X_eval = self.sample(X_train, X_sync, X_eval)
        super().__init__(None, X_sync, None, X_train, None, X_eval, None)
        if perplexity < 5 or perplexity > 50:
            warnings.warn(f"It is recommended that perplexity should be between 5 and 50, but got {perplexity}.")
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.lr = learning_rate
        self.n_jobs = n_jobs
        assert n_components in [2, 3], "n_components should be 2 or 3."
        self.n_components = n_components
        self.save = save
        self.img_name = img_name
    
    def sample(self, X_train, X_sync, X_eval):
        train_indices = np.random.choice(len(X_train), self.sample_num, replace=False)
        sync_indices = np.random.choice(len(X_sync), self.sample_num, replace=False)
        eval_indices = np.random.choice(len(X_eval), self.sample_num, replace=False)
        return X_train[train_indices], X_sync[sync_indices], X_eval[eval_indices]
    
    def __call__(self) -> None:
        clf = TSNE(n_components=self.n_components, random_state=2023, perplexity=self.perplexity, n_iter=self.n_iter, learning_rate=self.lr, n_jobs=self.n_jobs, verbose=1)
        whole_data = np.concatenate((self.X_train, self.X_sync, self.X_eval), axis=0).astype(np.float32)
        print(f"Iterations: {self.n_iter}\nPerplexity: {self.perplexity}\nLearning rate: {self.lr}\nVisualizing with t-SNE with {self.sample_num} samples for each dataset...\n")

        # DEBUG:
        unique_data = np.unique(whole_data)
        print(f"Unique values: {set(type(e) for e in unique_data)}")

        X_embedded = clf.fit_transform(whole_data)
        min, max = np.percentile(X_embedded, 1), np.percentile(X_embedded, 99)       # for setting the axes limits
        
        alpha = 0.3
        s = 25
        
        sns.set_style("white")
        colors = ["red" for _ in range(self.sample_num)] +  ["blue" for _ in range(self.sample_num)] + ["orange" for _ in range(self.sample_num)]
        if self.n_components == 2:
            plt.figure(figsize=(8, 6))
            # f, ax = plt.subplots(1)
            plt.scatter(
                X_embedded[:self.sample_num, 0], 
                X_embedded[:self.sample_num, 1], 
                c=colors[:self.sample_num], 
                label="Real Training", alpha=alpha,
                s=s,
            )
            plt.scatter(
                X_embedded[2*self.sample_num: 3*self.sample_num, 0], 
                X_embedded[2*self.sample_num: 3*self.sample_num, 1],
                c=colors[2*self.sample_num: 3*self.sample_num], 
                label="Real Testing", alpha=alpha, 
                s=s,
            )
            plt.scatter(
                X_embedded[self.sample_num: 2*self.sample_num, 0], 
                X_embedded[self.sample_num: 2*self.sample_num, 1], 
                c=colors[self.sample_num: 2*self.sample_num], 
                label="Synthetic", alpha=alpha, 
                s=s,
            )
            # plt.xlabel("X", fontsize=16)
            # plt.ylabel("Y", fontsize=16)
            # plt.title("2D t-SNE Embedding", weight='bold', fontsize=18)
            # plt.legend(fontsize=16)
            
        elif self.n_components == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(
                X_embedded[:self.sample_num, 0], 
                X_embedded[:self.sample_num, 1],
                X_embedded[:self.sample_num, 2], 
                c=colors[:self.sample_num], 
                label="Real Training", alpha=alpha, s=s
            )
            ax.scatter(
                X_embedded[2*self.sample_num: 3*self.sample_num, 0], 
                X_embedded[2*self.sample_num: 3*self.sample_num, 1],
                X_embedded[2*self.sample_num: 3*self.sample_num, 2],
                c=colors[2*self.sample_num: 3*self.sample_num], 
                label="Real Testing", alpha=alpha, s=s
            )
            ax.scatter(
                X_embedded[self.sample_num: 2*self.sample_num, 0], 
                X_embedded[self.sample_num: 2*self.sample_num, 1], 
                X_embedded[self.sample_num: 2*self.sample_num, 2], 
                c=colors[self.sample_num: 2*self.sample_num], 
                label="Synthetic", alpha=alpha, s=s
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            # ax.set_title("3D t-SNE Embedding")
            # ax.legend()
        plt.xlim(min-2, max+2)
        plt.ylim(min-2, max+2)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        if self.save:
            if self.img_name.split(".")[-1] == "pdf":
                plt.savefig(self.img_name, format="pdf")
            else:
                plt.savefig(self.img_name, dpi=500)
            print(f"Saved as {self.img_name}")
        else:
            plt.show()
            

class TrajectoryVisualizer:
    def __init__(self, sync_data, real_data, img_name, var_name_dict, fig_size=(10, 5), sample_size=2000):
        assert isinstance(sync_data, np.ndarray) and isinstance(real_data, np.ndarray), "data must be numpy array!"
        self.sync_data = sync_data
        self.real_data = real_data
        self.sample_size = sample_size if min(len(sync_data), len(real_data)) > sample_size else min(len(sync_data), len(real_data))
        self.fig_size = fig_size
        self.img_name = img_name
        self.var_name_dict = var_name_dict
    
    def sample(self):
        sync_indices = np.random.choice(len(self.sync_data), self.sample_size, replace=False)
        real_indices = np.random.choice(len(self.real_data), self.sample_size, replace=False)
        self.sync_data = self.sync_data[sync_indices]
        self.real_data = self.real_data[real_indices]
    
    def __call__(self):
        self.sample()
        sns.set_theme(style="ticks")
        
        sync_mean = np.nanmean(self.sync_data, axis=0)
        sync_std = np.nanstd(self.sync_data, axis=0)

        real_mean = np.nanmean(self.real_data, axis=0)
        real_std = np.nanstd(self.real_data, axis=0)

        time_steps = self.sync_data.shape[2]
        
        for variable_idx in range(self.sync_data.shape[1]):
            df = pd.DataFrame({
                'Time': np.tile(np.arange(time_steps), 2),
                'Value': np.concatenate((sync_mean[variable_idx], real_mean[variable_idx])),
                'Type': ['Synthetic'] * time_steps + ['Real'] * time_steps
            })

            plt.figure(figsize=self.fig_size)
            sns.lineplot(data=df, x='Time', y='Value', hue='Type')

            plt.fill_between(np.arange(time_steps),
                             sync_mean[variable_idx] - sync_std[variable_idx],
                             sync_mean[variable_idx] + sync_std[variable_idx], color='blue', alpha=0.2)
            plt.fill_between(np.arange(time_steps),
                             real_mean[variable_idx] - real_std[variable_idx],
                             real_mean[variable_idx] + real_std[variable_idx], color='orange', alpha=0.2)

            plt.xlabel('Time', fontsize=16)
            plt.ylabel('Value', fontsize=16)
            plt.legend(fontsize=16)
            sns.despine()
            
            if hasattr(self, 'img_name') and hasattr(self, 'var_name_dict'):
                file_name = f"{self.img_name}_{self.var_name_dict[variable_idx]}"
                plt.savefig(f"{file_name}.pdf", format="pdf")
                print(f"Saved as {file_name}.pdf")


class DiscriminativeScore:
    """
    Calculate discriminative score for synthetic data. Used implementation from previous studies for sake of consistency., **This needs tensorflow 1.X**
    
    Reference:
    ----------
        Yoon, Jinsung, Daniel Jarrett, and Mihaela Van der Schaar. "Time-series generative adversarial networks." Advances in neural information processing systems 32 (2019).
    """
    def __init__(self, original_data, synthetic_data, iterations=5000, scaler="standardize", train_min=None, train_max=None):
        assert scaler in ["standardize", "normalize"], "must be one of these!"
        self.scaler = scaler
        self.original_data = original_data
        self.synthetic_data = synthetic_data
        self.iterations = iterations
        self.train_min = train_min                    # used for consistency with previous approaches, which evaluates only on normalized data (didn't reverse the process)
        self.train_max = train_max
    
    def sample(self):
        num = len(self.original_data) if len(self.original_data) < len(self.synthetic_data) else len(self.synthetic_data)
        original_indices = np.arange(len(self.original_data))
        synthetic_indices = np.arange(len(self.synthetic_data))
        np.random.shuffle(original_indices)
        np.random.shuffle(synthetic_indices)
        if self.scaler == "standardize":
            print("Standardizing...")
            self.original_data, self.synthetic_data = standardize_data(self.original_data[original_indices[:num]]), standardize_data(self.synthetic_data[synthetic_indices[:num]])
        elif self.scaler == "normalize":
            print("Normalizing...")
            self.original_data, self.synthetic_data = normalize_data(self.original_data[original_indices[:num]]), normalize_data(self.synthetic_data[synthetic_indices[:num]], min=self.train_min, max=self.train_max)
        
    def __call__(self):
        self.sample()
        import tensorflow as tf
        tf.reset_default_graph()
        dataX, dataX_hat = self.original_data, self.synthetic_data
        # Basic Parameters
        No = len(dataX)
        data_dim = len(dataX[0][0,:])

        dataT = list()
        Max_Seq_Len = 0
        for i in range(No):
            Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
            dataT.append(len(dataX[i][:,0]))
        
        hidden_dim = max(int(data_dim/2),1)
        iterations = self.iterations
        batch_size = 128
        
        X = tf.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name = "myinput_x")
        X_hat = tf.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name = "myinput_x_hat")

        T = tf.placeholder(tf.int32, [None], name = "myinput_t")
        T_hat = tf.placeholder(tf.int32, [None], name = "myinput_t_hat")
        
        def discriminator (X, T):
        
            with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE) as vs:
                d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'cd_cell')
                # d_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim, activation=tf.nn.tanh, name='cd_cell')
                d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, X, dtype=tf.float32, sequence_length = T)
                Y_hat = tf.contrib.layers.fully_connected(d_last_states, 1, activation_fn=None) 
                Y_hat_Final = tf.nn.sigmoid(Y_hat)          # binary classification, sigmoid is fine
                d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
        
            return Y_hat, Y_hat_Final, d_vars
        
        def train_test_divide (dataX, dataX_hat, dataT):
        
            No = len(dataX)
            idx = np.random.permutation(No)
            train_idx = idx[:int(No*0.8)]
            test_idx = idx[int(No*0.8):]
            
            trainX = [dataX[i] for i in train_idx]
            trainX_hat = [dataX_hat[i] for i in train_idx]
                
            testX = [dataX[i] for i in test_idx]
            testX_hat = [dataX_hat[i] for i in test_idx]
            
            trainT = [dataT[i] for i in train_idx]
            testT = [dataT[i] for i in test_idx]
        
            return trainX, trainX_hat, testX, testX_hat, trainT, testT
        
        Y_real, Y_pred_real, d_vars = discriminator(X, T)
        Y_fake, Y_pred_fake, _ = discriminator(X_hat, T_hat)
            
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_real, labels = tf.ones_like(Y_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_fake, labels = tf.zeros_like(Y_fake)))
        D_loss = D_loss_real + D_loss_fake
        
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        trainX, trainX_hat, testX, testX_hat, trainT, testT = train_test_divide (dataX, dataX_hat, dataT)
        
        for itt in tqdm(range(iterations), desc="Training Discriminator..."):
            idx = np.random.permutation(len(trainX))
            train_idx = idx[:batch_size]     
                
            X_mb = list(trainX[i] for i in train_idx)
            T_mb = list(trainT[i] for i in train_idx)

            idx = np.random.permutation(len(trainX_hat))
            train_idx = idx[:batch_size]     
                
            X_hat_mb = list(trainX_hat[i] for i in train_idx)
            T_hat_mb = list(trainT[i] for i in train_idx)

            _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})            
                    
        Y_pred_real_curr, Y_pred_fake_curr = sess.run([Y_pred_real, Y_pred_fake], feed_dict={X: testX, T: testT, X_hat: testX_hat, T_hat: testT})
        
        Y_pred_final = np.squeeze(np.concatenate((Y_pred_real_curr, Y_pred_fake_curr), axis = 0))
        Y_label_final = np.concatenate((np.ones([len(Y_pred_real_curr),]), np.zeros([len(Y_pred_real_curr),])), axis = 0)
    
        Acc = accuracy_score(Y_label_final, Y_pred_final>0.5)  
        Disc_Score = np.abs(0.5-Acc)
        
        return Disc_Score 


class PredictiveScore:
    """
    Calculate predictive score for evaluation. 
    Primarily used the approach in TimeGAN paper but modified it to predict all values instead of one.
    See details in GT-GAN paper supplementary material (they did the same thing for predictive metric as well).
    
    Reference:
    ----------
        Yoon, Jinsung, Daniel Jarrett, and Mihaela Van der Schaar. "Time-series generative adversarial networks." Advances in neural information processing systems 32 (2019).
        Jeon, Jinsung, et al. "GT-GAN: General Purpose Time Series Synthesis with Generative Adversarial Networks." Advances in Neural Information Processing Systems 35 (2022): 36999-37010.
    """
    def __init__(self, original_data, synthetic_data, iterations=5000, scaler="standardize", train_min=None, train_max=None):
        assert scaler in ["standardize", "normalize"], "must be one of these!"
        self.scaler = scaler
        self.original_data = original_data
        self.synthetic_data = synthetic_data
        self.iterations = iterations
        self.train_min = train_min          # used for consistency with previous approaches, which evaluates only on normalized data (didn't reverse the process)
        self.train_max = train_max
    
    def sample(self):
        num = len(self.original_data) if len(self.original_data) < len(self.synthetic_data) else len(self.synthetic_data)
        original_indices = np.arange(len(self.original_data))
        synthetic_indices = np.arange(len(self.synthetic_data))
        np.random.shuffle(original_indices)
        np.random.shuffle(synthetic_indices)
        if self.scaler == "standardize":
            self.original_data, self.synthetic_data = standardize_data(self.original_data[original_indices[:num]]), standardize_data(self.synthetic_data[synthetic_indices[:num]])
        elif self.scaler == "normalize":        # for consistency with previous approaches (TimeGAN, GT-GAN)
            self.original_data, self.synthetic_data = normalize_data(self.original_data[original_indices[:num]]), normalize_data(self.synthetic_data[synthetic_indices[:num]], min=self.train_min, max=self.train_max)
    
    def __call__(self):
        import tensorflow as tf
        self.sample()
        dataX, dataX_hat = self.original_data, self.synthetic_data
        tf.reset_default_graph()

        No = len(dataX)
        data_dim = len(dataX[0][0,:])
        
        dataT = list()
        Max_Seq_Len = 0
        for i in range(No):
            Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
            dataT.append(len(dataX[i][:,0]))
        
        hidden_dim = max(int(data_dim/2),1)
        iterations = self.iterations
        batch_size = 128

        X = tf.placeholder(tf.float32, [None, Max_Seq_Len-1, data_dim], name = "myinput_x")
        T = tf.placeholder(tf.int32, [None], name = "myinput_t")    
        Y = tf.placeholder(tf.float32, [None, Max_Seq_Len-1, data_dim], name = "myinput_y")
        
        def predictor (X, T):
        
            with tf.variable_scope("predictor", reuse = tf.AUTO_REUSE) as vs:
                d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
                d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, X, dtype=tf.float32, sequence_length = T)
                # Y_hat = tf.contrib.layers.fully_connected(d_outputs, data_dim, activation_fn=None) 
                # Y_hat_Final = tf.nn.sigmoid(Y_hat)   
                # NOTE: use linear activation instead of sigmoid since standardization is applied rather than normalization
                Y_hat_Final = tf.contrib.layers.fully_connected(d_outputs, data_dim, activation_fn=None) 
                d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
        
            return Y_hat_Final, d_vars
        
        Y_pred, d_vars = predictor(X, T)
        D_loss = tf.losses.absolute_difference(Y, Y_pred)
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        for itt in tqdm(range(iterations), desc="Training Predictor..."):
            idx = np.random.permutation(len(dataX_hat))
            train_idx = idx[:batch_size]     
                
            X_mb = list(dataX_hat[i][:-1] for i in train_idx)
            T_mb = list(dataT[i]-1 for i in train_idx)
            Y_mb = list(np.reshape(dataX_hat[i][1:],[len(dataX_hat[i][1:]),data_dim]) for i in train_idx)        

            _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})            

        idx = np.random.permutation(len(dataX_hat))
        train_idx = idx[:No]     
        
        X_mb = list(dataX[i][:-1,:(data_dim)] for i in train_idx)
        T_mb = list(dataT[i]-1 for i in train_idx)
        Y_mb = list(np.reshape(dataX[i][1:], [len(dataX[i][1:]),data_dim]) for i in train_idx)
        
        pred_Y_curr = sess.run(Y_pred, feed_dict={X: X_mb, T: T_mb})
        
        MAE_Temp = 0
        for i in range(No):
            MAE_Temp = MAE_Temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
        
        MAE = MAE_Temp / No
        return MAE
    
    # def __call__(self):
    #     # NOTE: this version uses the same predictor objective as TimeGAN, i.e., only predicts the next one single value rather than all the values
    #     # NOTE: we do this for sake of consistency and fair comparison.
    #
    #     # Initialization on the Graph
    #     import tensorflow as tf
    #     self.sample()
        
    #     def extract_time(data):
    #         time = list()
    #         max_seq_len = 0
    #         for i in range(len(data)):
    #             max_seq_len = max(max_seq_len, len(data[i][:,0]))
    #             time.append(len(data[i][:,0]))
                
    #         return time, max_seq_len
        
    #     ori_data, generated_data = self.original_data, self.synthetic_data
    #     tf.reset_default_graph()

    #     # Basic Parameters
    #     no, seq_len, dim = np.asarray(ori_data).shape
            
    #     # Set maximum sequence length and each sequence length
    #     ori_time, ori_max_seq_len = extract_time(ori_data)
    #     generated_time, generated_max_seq_len = extract_time(ori_data)
    #     max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
            
    #     ## Builde a post-hoc RNN predictive network 
    #     # Network parameters
    #     hidden_dim = int(dim/2)
    #     iterations = 5000
    #     batch_size = 128
            
    #     # Input place holders
    #     X = tf.placeholder(tf.float32, [None, max_seq_len-1, dim-1], name = "myinput_x")
    #     T = tf.placeholder(tf.int32, [None], name = "myinput_t")    
    #     Y = tf.placeholder(tf.float32, [None, max_seq_len-1, 1], name = "myinput_y")
            
    #     # Predictor function
    #     def predictor (x, t):
    #         """Simple predictor function.
            
    #         Args:
    #         - x: time-series data
    #         - t: time information
            
    #         Returns:
    #         - y_hat: prediction
    #         - p_vars: predictor variables
    #         """
    #         with tf.variable_scope("predictor", reuse = tf.AUTO_REUSE) as vs:
    #             p_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')
    #             p_outputs, p_last_states = tf.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length = t)
    #             y_hat_logit = tf.contrib.layers.fully_connected(p_outputs, 1, activation_fn=None) 
    #             y_hat = tf.nn.sigmoid(y_hat_logit)
    #             p_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            
    #         return y_hat, p_vars
            
    #     y_pred, p_vars = predictor(X, T)
    #     # Loss for the predictor
    #     p_loss = tf.losses.absolute_difference(Y, y_pred)
    #     # optimizer
    #     p_solver = tf.train.AdamOptimizer().minimize(p_loss, var_list = p_vars)
                
    #     ## Training    
    #     # Session start
    #     sess = tf.Session()
    #     sess.run(tf.global_variables_initializer())
            
    #     # Training using Synthetic dataset
    #     for itt in tqdm(range(iterations), desc="Training Predictor..."):
                
    #         # Set mini-batch
    #         idx = np.random.permutation(len(generated_data))
    #         train_idx = idx[:batch_size]     
                    
    #         X_mb = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)
    #         T_mb = list(generated_time[i]-1 for i in train_idx)
    #         Y_mb = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx)        
                
    #         # Train predictor
    #         _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})        
            
    #     ## Test the trained model on the original data
    #     idx = np.random.permutation(len(ori_data))
    #     train_idx = idx[:no]
            
    #     X_mb = list(ori_data[i][:-1,:(dim-1)] for i in train_idx)
    #     T_mb = list(ori_time[i]-1 for i in train_idx)
    #     Y_mb = list(np.reshape(ori_data[i][1:,(dim-1)], [len(ori_data[i][1:,(dim-1)]),1]) for i in train_idx)
            
    #     # Prediction
    #     pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})
            
    #     # Compute the performance in terms of MAE
    #     MAE_temp = 0
    #     for i in range(no):
    #         MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
            
    #     predictive_score = MAE_temp / no
            
    #     return predictive_score
    