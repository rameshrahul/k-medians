#import taichi as ti
#import taichi.math as tm
import matplotlib.pyplot as plt
import numpy as np
from KMediansSolver import *
from KMediansInstance import *

import time


def visualize(vertices, assignments):
    unique_clusters = np.unique(assignments)
    colors = plt.get_cmap('tab10', len(unique_clusters))

    plt.figure()

    for i, cluster in enumerate(unique_clusters):
        cluster_points = vertices[assignments == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}", color=colors(i))
    
    plt.title("clusters of vertices")
    plt.legend()
    plt.show()


# Fixing random state for reproducibility
#np.random.seed(123123) #19680801

n = 30
data_range = 100
random_points = np.random.rand(n, 2)*data_range


instance = EuclideanInstance(random_points)
k = 5

print("running IP")
start = time.time()
ipsolver = IntegerProgram(instance, k)
obj, medians, assignments = ipsolver.solve()
print("IP opt: {}".format(obj))
print("IP runtime: {}".format(time.time() - start))

print("running brute force")
bruteforce = BruteForceSolver(instance, k)
start = time.time()
obj, medians, assignments = bruteforce.solve()
print("Brute Force opt: {}".format(obj))
print("brute force runtime: {}".format(time.time() - start))

print(obj)
print(medians)
print(assignments)


visualize(random_points, assignments)
#ax = fig.add_subplot(projection='3d')

fig = plt.figure()
ax = fig.add_subplot()
x, y = random_points.T
ax.scatter(x, y, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')

#plt.show()