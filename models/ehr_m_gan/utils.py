import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def renormlizer(data, max_val, min_val):
    data = data * max_val
    data = data + min_val
    return data

def ones_target(size, min, max):
    return tf.random_uniform([size, 1], min, max, dtype=tf.float32)

def zeros_target(size, min, max):
    return tf.random_uniform([size, 1], min, max, dtype=tf.float32)

def target_flip(size, min, max, ratio):
    ori = np.random.uniform(low=min, high=max, size=[size, 1])
    index = np.random.choice(size, int(size * ratio), replace=False)
    ori[index] = 1 - ori[index]
    return tf.convert_to_tensor(ori, dtype=tf.float32)

def np_sigmoid(x):  
    z = np.exp(-x)
    sig = 1/(1 + z)
    return sig

def np_rounding(prob):
    y = np.round(prob)
    return y