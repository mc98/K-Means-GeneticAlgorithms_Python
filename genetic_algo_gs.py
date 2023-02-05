#Import Libraries
import numpy
import matplotlib.pyplot as plt
from generate_data_class import GenerateData
import kmeans

#Set Constant Variables
num_clusters = 2

#Artificial Data Generation
gy = GenerateData(num_clusters)
data = gy.raw_data()


def cluster_data(solution):
    '''
    Input: 
    1. solution is the initial coordinates of centriods. For example,
    if num_of_clusters = 2, then
    solution = [C1-x, C1-y, C2-x, C2-y]
    2. data is the whole data, all the samples

    '''
    global num_clusters, data
    feature_vector_length = data.shape[1]
    cluster_centers = []
    all_clusters_dists = []
    clusters = []
    clusters_sum_dist = []

    for clust_idx in range(num_clusters):
        cluster_centers.append(solution[feature_vector_length*clust_idx:feature_vector_length*(clust_idx+1)])
        cluster_center_dists = kmeans.euclidean(data, cluster_centers[clust_idx])
        all_clusters_dists.append(numpy.array(cluster_center_dists))

    cluster_centers = numpy.array(cluster_centers)
    all_clusters_dists = numpy.array(all_clusters_dists)

    cluster_indices = numpy.argmin(all_clusters_dists, axis=0)
    for clust_idx in range(num_clusters):
        clusters.append(numpy.where(cluster_indices == clust_idx)[0])
        if len(clusters[clust_idx]) == 0:
            clusters_sum_dist.append(0)
        else:
            clusters_sum_dist.append(numpy.sum(all_clusters_dists[clust_idx, clusters[clust_idx]]))

    clusters_sum_dist = numpy.array(clusters_sum_dist)

    return cluster_centers, all_clusters_dists, clusters, clusters_sum_dist


def fitness_func(solution):
    '''
    fitness_func() is created and calls the cluster_data() function and 
    calculates the sum of distances in all clusters
    '''
    fit_list = []
    m,_ = solution.shape
    for t in range(m):
        _, _, _, clusters_sum_dist = cluster_data(solution[t])
        fitness = 1.0 / (numpy.sum(clusters_sum_dist) + 0.00000001)
        fit_list.append(fitness)

    return numpy.array(fit_list)

#GENERATE Initial coordinates for cluster center
def init_cluster_center(num_clusters,dataCent):
    io = []
    rc = [1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09]
    if num_clusters == 3:
        cluster1_x1 = numpy.array(dataCent)
    elif num_clusters == 2:
        cluster1_x1 = numpy.array(dataCent)
    else:
        print("Out of Scope")

    for y in range(len(rc)):
        ui = rc[y]*cluster1_x1
        io.append(ui)
    return numpy.array(io)

#Print Result
def get_results(generation,population,fitness):
    best = [fitness.max()]
    index = numpy.where(numpy.isclose(fitness, best))
    population = numpy.array(population)
    print(f'Generation #{generation}   |fitness: {max(fitness):0.7f} |Centroid = {population[index[0]][0]}')
    return population[index[0]][0]

#Plot Fitness Values over generations
def display_plot(best):
    plt.plot(best, color='c')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid()
    plt.show()

#Steps in Genetic Algorithm
#1. Selection
def selection(pop,sample_size, fitness):
    m,n = pop.shape
    new_pop = pop.copy()
       
    for i in range(m):
        rand_id = numpy.random.choice(m, size=max(1, int(sample_size*m)), replace=False)
        max_id = rand_id[fitness[rand_id].argmax()]
        new_pop[i] = pop[max_id].copy()
    
    return new_pop

#2. Crossover
def crossover(pop, pc):
    m,n = pop.shape
    new_pop = pop.copy()
    
    for i in range(0, m-1, 2):
        if numpy.random.uniform(0, 1) < pc:
            pos = numpy.random.randint(0, n-1)
            new_pop[i, pos+1:] = pop[i+1, pos+1:].copy()
            new_pop[i+1, pos+1:] = pop[i, pos+1:].copy()
            
    return new_pop

#3. Mutation
def mutation(pop, pm):
    m,n = pop.shape
    new_pop = pop.copy()
    mutation_prob = (numpy.random.uniform(0, 1, size=(m,n)) < pm).astype(int)
    return (mutation_prob + new_pop)


def GeneticAlgorithm(func,num_clusters,gen,dataCen,ps=0.2,pc=1.0,pm=0.1,random_state=1234):    
    numpy.random.seed(random_state)
    pop = init_cluster_center(num_clusters,dataCen)
    fitness = func(pop)
    best = [fitness.max()] 

    print('=' * 68)
    list_best_centroids = []
    ly0 = []
    res0 = get_results(-1,pop,fitness)
    ly0.append(numpy.array(res0[0:2]))
    ly0.append(numpy.array(res0[2:4]))
    if num_clusters == 3:
        ly0.append(numpy.array(res0[4:6]))
    list_best_centroids.append(ly0)

    i = 0
    while i < gen:
        pop = selection(pop, ps, fitness)
        pop = crossover(pop, pc)
        pop = mutation(pop, pm)
        fitness = func(pop)
        best.append(fitness.max())
        res1 = get_results(i,pop,fitness)
        ly0 = []
        ly0.append(numpy.array(res1[0:2]))
        ly0.append(numpy.array(res1[2:4]))
        if num_clusters == 3:
            ly0.append(numpy.array(res1[4:6]))
        list_best_centroids.append(ly0)
        i += 1
    
    print('=' * 68)
    print(f'Maximum fitness is: {max(best):0.5f} at Generation: {best.index(max(best))}')
        
    return fitness, best, i, pop, list_best_centroids