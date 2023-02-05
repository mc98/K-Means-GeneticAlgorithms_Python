# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:12:35 2022

@author: micho
"""

# libraries
import numpy
import random
import matplotlib.pyplot

class GenerateData:
    '''
    A class to generate and plot data in the the form of clusters for K-means clustering technique.
    '''
    
    def __init__(self,k):
        '''
        A is a primary function

        Parameters
        ----------
        k = int
            Number of clusters
        '''
        self.num_of_clusters = k

    @staticmethod
    def data_generation(num_of_samples, 
                        X_Cooridinate_Start, 
                        X_Cooridinate_End, 
                        Y_Cooridinate_Start, 
                        Y_Cooridinate_End):
        
        cluster_X = numpy.random.random(size=(num_of_samples))
        cluster_X = cluster_X * (X_Cooridinate_End - X_Cooridinate_Start) + X_Cooridinate_Start
        cluster_Y = numpy.random.random(size=(num_of_samples))
        cluster_Y = cluster_Y * (Y_Cooridinate_End - Y_Cooridinate_Start) + Y_Cooridinate_Start
        return cluster_X, cluster_Y

    
    def raw_data(self):
        data = []
        k = self.num_of_clusters
        if k == 2:
            c1 = numpy.array([self.data_generation(100,0,25,2,50)]).T
            c2 = numpy.array([self.data_generation(100,40,80,30,100)]).T
            data = numpy.concatenate((c1, c2), axis=0)

        elif k == 3:
            c1 = numpy.array([self.data_generation(100,0,25,2,50)]).T
            c2 = numpy.array([self.data_generation(100,40,80,30,100)]).T
            c3 = numpy.array([self.data_generation(100,50,90,80,150)]).T
            data = numpy.concatenate((c1, c2, c3), axis=0)

        else:
            print("Out of Scope")   

        return data

        
    def plot_input(self,input_data):
        '''
        This function plots the input data on a 2-D plane.

        Parameters
        -------------
        input_data = Dict
                    Data generated from @raw_data() function

        Returns
        -------------
        return: 2-D plot
        '''
        for i in range(len(input_data)):
            coordinate_list = input_data[f'cluster_{i}']
            matplotlib.pyplot.scatter(coordinate_list[0],
                                    coordinate_list[1])

        return matplotlib.pyplot.show()
    
    """  def raw_to_standard(self, raw_data):
        i = 0
        true_labels = []
        l = []
        for k in raw_data:
            for j in range(len(raw_data[k][0])):
                l.append([raw_data[k][0][j], raw_data[k][1][j]])
                true_labels.append(i)
            i+=1       
        true_labels = numpy.array(true_labels)
        l = numpy.array(l)
        return l, true_labels """
    
