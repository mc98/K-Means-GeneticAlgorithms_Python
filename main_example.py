# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:13:09 2022

@author: micho
"""

import generate_data_lib as gd
import kmeans as km

data_c = gd.generate_data(3)
raw_data = data_c.raw_data()       
# transform
X_train, true_labels = data_c.raw_to_standard(raw_data)
# plot
data_c.plot_input(raw_data)

model = km.KMeans(3)
model.fit(X_train)
print('Centroids:', model.centroids)
print('Evaluation:',model.evaluate(X_train[0:2]))
print('Distance 0:',km.euclidean(X_train[0],model.centroids))
print('Distance 1:',km.euclidean(X_train[1],model.centroids))
print('True labels:',true_labels[0],true_labels[1])