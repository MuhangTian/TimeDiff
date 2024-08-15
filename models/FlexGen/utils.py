import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.sparse as sp 
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import random
from itertools import chain

DATA_SOURCE = "mimic"

def plot_sub(real_prob, fake_prob, feature, ax, name, cc, rmse):
    df = pd.DataFrame({'real': real_prob,  'fake': fake_prob, "feature": feature})
    sns.scatterplot(ax=ax, data=df, x='real', y='fake', hue="feature", s=10, alpha=0.8, edgecolor='none', legend=None, palette='Paired_r')
    sns.lineplot(ax=ax, x=[0, 1], y=[0, 1], linewidth=0.5, color="darkgrey")
    ax.set_title(name, fontsize=11)
    ax.set(xlabel="Bernoulli success probability of real data")
    ax.set(ylabel="Bernoulli success probability of synthetic data")
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.text(0.75, 0.05, 'CC='+str(cc), fontsize=9)
    ax.text(0.75, 0.05, 'RMSE='+str(rmse), fontsize=9)


def cal_cc(real_prob, fake_prob):
    return float("{:.4f}".format(np.corrcoef(real_prob, fake_prob)[0, 1]))

def cal_rmse(real_prob, fake_prob):
    return float("{:.4f}".format(sqrt(mean_squared_error(real_prob, fake_prob)))) 
    



def Pamona_geodesic_distances(X, num_neighbors, mode="distance", metric="minkowski"):

    assert (mode in ["connectivity", "distance"]), "Norm argument has to be either one of 'connectivity', or 'distance'. "
    if mode=="connectivity":
        include_self=True
    else:
        include_self=False
    knn = kneighbors_graph(X, num_neighbors, n_jobs=-1, mode=mode, metric=metric, include_self=include_self)
    connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
    dist = sp.csgraph.dijkstra(knn, directed=False)
    connected_element = []

    ## for local connectively
    if connected_components is not 1:
        inf_matrix = []
        
        for i in range(len(X)):
            inf_matrix.append(list(chain.from_iterable(np.argwhere(np.isinf(dist[i])))))

        for i in range(len(X)):
            if i==0:
                connected_element.append([0])
            else:
                for j in range(len(connected_element)+1):
                    if j == len(connected_element):
                        connected_element.append([])
                        connected_element[j].append(i)
                        break
                    if inf_matrix[i] == inf_matrix[connected_element[j][0]]:
                        connected_element[j].append(i)
                        break

        components_dist = []
        x_index = []
        y_index = []
        components_dist.append(np.inf)
        x_index.append(-1)
        y_index.append(-1)
        for i in range(connected_components):
            for j in range(i):    
                for num1 in connected_element[i]:
                    for num2 in connected_element[j]:
                        if np.linalg.norm(X[num1]-X[num2])<components_dist[len(components_dist)-1]:
                            components_dist[len(components_dist)-1]=np.linalg.norm(X[num1]-X[num2])
                            x_index[len(x_index)-1] = num1
                            y_index[len(y_index)-1] = num2
                components_dist.append(np.inf)
                x_index.append(-1)
                y_index.append(-1)

        components_dist = components_dist[:-1]
        x_index = x_index[:-1]
        y_index = y_index[:-1]

        sort_index = np.argsort(components_dist)
        components_dist = np.array(components_dist)[sort_index]
        x_index = np.array(x_index)[sort_index]
        y_index = np.array(y_index)[sort_index]

        for i in range(len(x_index)):
            knn = knn.todense()
            knn = np.array(knn)
            knn[x_index[i]][y_index[i]] = components_dist[i]
            knn[y_index[i]][x_index[i]] = components_dist[i]
            knn = sp.csr_matrix(knn)
            connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
            dist = sp.csgraph.dijkstra(knn, directed=False)
            if connected_components == 1:
                break

    return dist/dist.max()

def get_spatial_distance_matrix(data, metric="euclidean"):
    Cdata= sp.spatial.distance.cdist(data,data,metric=metric)
    return Cdata/Cdata.max()

def unit_normalize(data, norm="l2", bySample=True):
    """
    From SCOT code: https://github.com/rsinghlab/SCOT
    Default norm used is l2-norm. Other options: "l1", and "max"
    If bySample==True, then we independently normalize each sample. If bySample==False, then we independently normalize each feature
    """
    assert (norm in ["l1","l2","max"]), "Norm argument has to be either one of 'max', 'l1', or 'l2'."

    if bySample==True:
        axis=1
    else:
        axis=0

    return normalize(data, norm=norm, axis=axis)
    

def zscore_standardize(data):
    """
    From SCOT code: https://github.com/rsinghlab/SCOT
    """
    scaler=StandardScaler()
    scaledData=scaler.fit_transform(data)
    return scaledData


def init_random_seed(manual_seed):
    seed = None
    if manual_seed is None:
        seed = random.randint(1,10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)


def Interval_Estimation(Gc, interval_num=20):
    Gc = np.array(Gc)
    Gc_last_col = Gc[0:-1,-1]
    Gc_max = np.max(Gc_last_col)
    Gc_min = np.min(Gc_last_col)
    Gc_interval = Gc_max - Gc_min

    row = np.shape(Gc)[0]-1
    col = np.shape(Gc)[1]-1
    count = np.zeros(interval_num)

    interval_value = []
    for i in range(interval_num+1):
        interval_value.append(Gc_min+(1/interval_num)*i*Gc_interval)

    for i in range(row):
        for j in range(interval_num):
            if Gc[i][col] >= interval_value[j] and Gc[i][col] < interval_value[j+1]:
                count[j] += 1
            if Gc[i][col] == interval_value[j+1]:
                count[interval_num-1] += 1

    print('count', count)

    fig = plt.figure(figsize=(10, 6.5))

    a = list(range(interval_num))
    a = list(map(str,a))
    font_label = {
             'weight': 'normal',
             'size': 25,
         }

    plt.plot(a,count,'k')

    for i in range(interval_num):
        plt.plot(a[i], count[i], 's', color='k')

    plt.xticks(fontproperties = 'Arial', size = 18)
    plt.yticks(fontproperties = 'Arial', size = 18)
    plt.xlabel('interval', font_label)
    plt.ylabel('number', font_label)

    plt.show()

# print("evaluate VAE")
# dataset_train_object = MIMICDATASET(x_path='m_train.csv',x_s_path='ms_train.csv',\
#                                     y_path='my_train.csv', train=True, transform=False)
# vae_tmp = torch.load('vae_tmp.pt').to('cuda')
# vae_tmp.eval()
# x = dataset_train_object.xt.to('cuda')
# x_recon,mu,logvar = vae_tmp(x)

# vae_sta = torch.load('vae_sta.pt')
# vae_sta.eval()
    
# print("evaluate static")

# discrete_x_real = pd.read_csv('ms_train.csv',index_col=[0,1,2], header=[0,1])
# discrete_x_real.columns = discrete_x_real.columns.droplevel()
# discrete_x_real = pd.get_dummies(discrete_x_real, columns = ['diagnosis'])
# discrete_x_real = pd.get_dummies(discrete_x_real, columns = ['ethnicity'])
# discrete_x_real = pd.get_dummies(discrete_x_real, columns = ['admission_type'])
# discrete_x_fake = pd.read_csv('synthetic_mimic/ldm_static.csv', index_col=0)
# discrete_x_fake[discrete_x_fake<0.5] = 0
# discrete_x_fake[discrete_x_fake>=0.5] = 1
# real_prob = np.mean(discrete_x_real.values, axis=0)
# fake_prob = np.mean(discrete_x_fake.values, axis=0)
# feature = np.concatenate([([i]* discrete_x_real.shape[1] ) for i in list(range(discrete_x_real.shape[-1])) ], axis=0)
# cc_value = cal_cc(real_prob, fake_prob)
# rmse_value = cal_rmse(real_prob, fake_prob)
# sns.set_style("whitegrid", {'grid.linestyle': ' '})
# fig, ax = plt.subplots(figsize=(4.2, 3.8))
# plot_sub(real_prob, fake_prob, discrete_x_real.columns, ax, name="TabDDPM", cc=cc_value, rmse=rmse_value)
# fig.show()

# print("evaluate temporal")

# tmp_x_real = pd.read_csv('m_train.csv', index_col=[0,1,2], header=[0,1,2])
# tmp_x_fake = pd.read_csv('synthetic_mimic/ldm_tmp.csv', index_col=0, header=[0,1,2])
# tmp_x_fake[tmp_x_fake.loc[:, tmp_x_fake.columns.get_level_values(1)=='mask']<0.5]=0
# tmp_x_fake[tmp_x_fake.loc[:, tmp_x_fake.columns.get_level_values(1)=='mask']>=0.5]=1
# real_prob = np.mean(tmp_x_real.values, axis=0)
# fake_prob = np.mean(tmp_x_fake.values, axis=0)
# cc_value = cal_cc(real_prob, fake_prob)
# rmse_value = cal_rmse(real_prob, fake_prob)
# sns.set_style("whitegrid", {'grid.linestyle': ' '})
# fig, ax = plt.subplots(figsize=(4.2, 3.8))
# plot_sub(real_prob, fake_prob, tmp_x_real.columns.get_level_values(0), ax, name="TabDDPM", cc=cc_value, rmse=rmse_value)
# fig.show()