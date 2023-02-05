# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import itertools
# from sklearn.datasets import make_blobs

def euclidean(point, data):
    euc = np.sqrt(np.sum((point-data)**2, axis = 1))
    
    return euc

# def euclidean(point, data): # both point and data are arrays
#     l=[]
#     for row in data:
#         s = (point[0]-row[0])**2 + (point[1]-row[1])**2
#         euc = np.sqrt(s)
#         l.append(euc)
#     return np.array(l)

def plot_res(X_train,centroids, alpha , marker):
    plt.xlim(0, 100)
    plt.ylim(0,100)
    colors = itertools.cycle(["lime", "fuchsia", "red"])
    markers = itertools.cycle(["+","x","d"])
    x = []
    y = []
    for el in X_train:
        x.append(el[0])
        y.append(el[1])
    plt.scatter(x,y, c = 'k', alpha = 0.5)
    for el in centroids:
        plt.scatter(el[0], el[1],c = next(colors), marker = next(markers), alpha = alpha)
        
def plot_resV3(X_train,centroids, alpha , marker, nb_obs = 100, nb_clusters=2):
    plt.xlim(0, 200)
    plt.ylim(0,200)
    colors = itertools.cycle(["r", "g", "m"])
    markers = itertools.cycle(["+","x","1"])
    colorsc = ["tab:blue","tab:orange","tab:brown"]
    i = 0
    x = []
    y = []
    for el in X_train:
        i+=1
        x.append(el[0])
        y.append(el[1])
    for j in range(1, nb_clusters+1):
        plt.scatter(x[(j-1)*nb_obs:j*nb_obs], y[(j-1)*nb_obs:j*nb_obs] , c = colorsc[j-1])
    for el in centroids:
        plt.scatter(el[0], el[1],c = next(colors), marker = next(markers), alpha = alpha)
        
# def plot_resV2(data, alpha = 0.8, raw = True ):
#     plt.xlim(0, 100)
#     plt.ylim(0, 100)
#     colors = itertools.cycle(["r", "g", "b"])
#     markers = itertools.cycle(["+","x","d"])
#     x = []
#     y = []
#     for el in data:
#         x.append(el[0])
#         y.append(el[1])
#     if raw: # print the full data
#         plt.scatter(x,y, c = 'k', alpha = 0.8)    
#     else:
#         plt.scatter(el[0], el[1],c = next(colors), alpha = alpha, marker = next(markers))

class KMeans:
    def __init__(self, n_clusters, max_iter = 400):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.evolution = [] # to store the evolution of the centroids
        
    def fit(self, X_train, verbose = False, p = False):
        # randomly select the centroid start points according to a uniform distribution
        min_, max_ = np.min(X_train, axis = 0), np.max(X_train, axis = 0)
        self.centroids = [np.random.uniform(min_, max_) for _ in range(self.n_clusters)]
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            self.evolution.append(self.centroids)
            classified_points = [[] for _ in range(self.n_clusters)]
            if verbose: print("Iteration",iteration)
            for x in X_train:
                # print('X:',x)
                # print('Centroids:',np.array(self.centroids))
                dist = euclidean(x, self.centroids)
                # print('Distances',dist)
                centroid_id = np.argmin(dist)
                # print('Centroid ID',centroid_id)
                classified_points[centroid_id].append(x)
                
            # updated centroids
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis = 0) for cluster in classified_points]
                
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i] # don't update
            if verbose: print("Previous centroids:",prev_centroids)
            if verbose: print("New centroids:", self.centroids)
            if verbose: print("==============================")
            iteration +=1
        print("Algorithm converged after",iteration,"iterations.")
        alphas = itertools.cycle(list(np.arange(0.1,1,1/iteration)))
        if p: # plots
            # plot_resV2(X_train)
            for c in self.evolution:
                plot_resV3(X_train, c, alpha = next(alphas), marker = "x", nb_clusters=self.n_clusters)  
                # plot_resV2(c, alpha = next(alphas), raw = False)
        
            
    def evaluate(self, X):
        centroids = []
        centroid_ids = []
        for x in X:
            dist = euclidean(x, self.centroids)
            centroid_id = np.argmin(dist)
            centroids.append(self.centroids[centroid_id])
            centroid_ids.append(centroid_id)
            
        return centroids, centroid_ids
    
    def to__solution(self):
        l = []
        for el in self.centroids:
            l.append(el[0])
            l.append(el[1])
        return np.array(l)
    
# point = [1,2]
# data = [[1,2],[3,4],[5,6],[6,7]]
# point = np.array(point)
# data = np.array(data)
# # print(euclidean(point, data))
# # print(eucl(point, data))

# X_train, true_labels = make_blobs(n_samples=10, centers=2)
# # print(X_train, true_labels, type(X_train))
# # print(euclidean(point, X_train))
# model = KMeans(2)
# model.fit(X_train)
# print('Centroids:', model.centroids)
# print('Evaluation:',model.evaluate(X_train[0:2]))
# print('Distance 0:',euclidean(X_train[0],model.centroids))
# print('Distance 1:',euclidean(X_train[1],model.centroids))
# print('True labels:',true_labels[0],true_labels[1])
