#Import Libraries
from generate_data_class import GenerateData
import genetic_algo_gs as gag
import kmeans
import numpy
import itertools

#Set input values
kNo = 3 #Number of clusters
GA_gen = 500 #Number of generations for GA

#Data generation
gy = GenerateData(kNo)
rawData = gy.raw_data()

#########################################################################
#DO NOT BOTHER TO CHANGE ANYTHING HERE
'''
kmean_data() converts the data for kmeans algorithm
'''
def kmean_data(data):
    l = []
    for el in data:
        l2 = []
        l2.append(el[0][0])
        l2.append(el[1][0])
        l.append(numpy.array(l2))

    return numpy.array(l)

###########################################################################
#To perform k-Means algorithm UNCOMMENT below code
'''
model = kmeans.KMeans(kNo)
data1 = kmean_data(rawData)
model.fit(data1,False,True)
ml = model.to__solution()
print(f'Fitness value of K-Means output is: {gag.fitness_func(numpy.array([ml]))[0]:0.8f}')
'''
#To print fitness plot for k-Means UNCOMMENT below code
'''
ko = model.evolution
lp = []
kmean_fitness_list = []
ti = []
for el in ko: 
    yo = []   
    for subel in el:
        yo.append(subel[0])
        yo.append(subel[1])
    ti.append(numpy.array(yo))
ti = numpy.array(ti)

io = gag.fitness_func(ti)
gag.display_plot(io.tolist())
'''
#######################################################################################
#To perform Genetic Algorithm UNCOMMENT below code
'''
data_for_GA = []
for ty in model.evolution[-1]:
    data_for_GA.append(ty[0])
    data_for_GA.append(ty[1])
print(data_for_GA)

initial_centroids = gag.init_cluster_center(kNo,data_for_GA)
fit_func = gag.fitness_func(initial_centroids)

_, plot_result, _, _,centroidList = gag.GeneticAlgorithm(gag.fitness_func,kNo,GA_gen,dataCen=data_for_GA)
'''

#To plot GA results
#Fitness plot
'''
gag.display_plot(plot_result)
'''
#Centroid Evolution plot
'''
alphas = itertools.cycle(list(numpy.arange(0.1,1,1/GA_gen)))
for e in centroidList:
    kmeans.plot_resV3(data1,e,alpha = next(alphas),marker="x",nb_clusters=kNo)
'''