# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:08:18 2019

@author: MB87511
"""

import numpy as np
from random import random
import pandas as pd


class K_means(object):
    
    def __init__(self, n_clusters = 3, max_iter = 300):
            self.n_clusters = n_clusters 
            self.max_iter = max_iter
                
    def fit(self, data):
        
        self.centroids = data[np.random.permutation(data.shape[0])[:self.n_clusters]]
        for i in range(self.max_iter):
            self.labels = [self.closest_cluster(self.centroids, point) for point in data]

            old_centroids = np.array(self.centroids)
            for k in range(self.n_clusters):
                idxs = [idx for idx, label in enumerate(self.labels) if label == k]
                points_in_cluster = data[idxs]
                self.centroids[k] = points_in_cluster.sum(axis=0) / len(points_in_cluster)

            self.sum_of_errors = sum(((self.centroids[label] - point)**2).sum() for point, label in zip(X, self.labels))
            
            if(np.array_equal(self.centroids, old_centroids)):
                return self
            
        return self
    
    def euclid_dist(self, a, b):
        return np.sqrt(((a-b)**2).sum())
    
    def closest_cluster(self, centroids, points):
        return np.argmin([self.euclid_dist(points, centroid) for centroid in centroids])
    
    def predict(self, data):
        return self.labels
    
    def fit_predict(self, data):
        return self.fit(data).predict(data)
    
    def get_centroids(self):
        return self.centroids
    
    def get_sum_of_errors(self):
        return self.sum_of_errors


#### Method that returns best number of clusters for the data
def get_best_k(errors_sum_list):
    diffs = []
    for i in range(1,len(errors_sum_list)):
        diff = (errors_sum_list[i-1]-errors_sum_list[i])/errors_sum_list[i-1]
        diffs.append(diff)
    threshold = 0.2 
    best_k = [idx+1 for idx, diff in enumerate(diffs) if diff <= threshold][0]
    pd.DataFrame(errors_sum_list, index = range(1,10)).plot(title = 'Elbow Rule to determine best number of clusters, below threshold: ' +str(threshold))
    return best_k 

### Method to return labels of clusters for each datapoint and cluster's centroids for best_k kmean model
def best_kMeans_results(X, errors_sum_list, kmean_clusters):
    best_k = get_best_k(errors_sum_list)
    return best_k, pd.DataFrame(X).join(pd.Series(kmean_clusters['#clusters:'+str(best_k)].predict(X)).rename('Labels')), kmean_clusters['#clusters:'+str(best_k)].get_centroids()
         

def run_k_means(X):
    #### Generate different kmeans models and listing each sum of squared errors <<< MAIN >>> 
    errors_sum_list = []
    kmean_clusters = {}
    for i in range(1,10):
        model = K_means(n_clusters=i)
        model.fit(X)
        kmean_clusters['#clusters:'+str(i)] = model
        error = model.get_sum_of_errors()
        errors_sum_list.append(error)
    
    best_number_clusters, labels, centroids = best_kMeans_results(X, errors_sum_list, kmean_clusters)
    
    print('Best_number of clusters', best_number_clusters)
    print
    print('Labels for datapoints')
    print(labels)
    print 
    print('Centroids of clusters')
    print(centroids)

### Test case for any X-array
### Init 100-lenght random array of points
X= np.array([[random()*100, random()*100] for x in range(100)])
run_k_means(X)
