# -*- coding: utf-8 -*-
"""
@author: Huhaowen0130
"""

import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans

def get_stds(centroids, x_train, y_train, k):
    data_size = x_train.shape[0]
    
    stds = np.zeros(k)
    classes = np.zeros(k)
    for i in range(data_size):
        stds[int(y_train[i])] += np.linalg.norm(centroids[int(y_train[i])] - x_train[i])
        classes[int(y_train[i])] += 1        
    for j in range(k):
        stds[j] = stds[j] / classes[j]
        
    return stds

def rbf_units(x_train, k):
    kmeans_model = KMeans(k).fit(x_train)

    centroids = np.array(kmeans_model.cluster_centers_)    
    stds = get_stds(centroids, x_train, kmeans_model.labels_, k)
    
    return centroids, stds

def rbf(centroids, stds, x_data):
    k = len(centroids)
    input_size = x_data.shape[0]
    
    betta = np.zeros(k)    
    output = np.zeros([input_size, k])    
    for i in range(k):
        betta[i] = 1 / (2 * np.power(stds[i], 2))
        for j in range(input_size):            
            output[j][i] = np.exp(-betta[i] * np.power(np.linalg.norm(centroids[i] - x_data[j]), 2))
            
    return output