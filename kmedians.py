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

# print("running brute force")
# bruteforce = BruteForceSolver(instance, k)
# start = time.time()
# obj, medians, assignments = bruteforce.solve()
# print("Brute Force opt: {}".format(obj))
# print("brute force runtime: {}".format(time.time() - start))

print("running local search greedy")
start = time.time()
lsgreedy = LocalSearchSolver(instance, k, epsilon=10, greedy=True)
obj, medians, assignments, num_iter = lsgreedy.solve()
print("LS greedy opt: {}".format(obj))
print("LS greedy runtime: {}".format(time.time() - start))
print("LS greedy num_iter: {}".format(num_iter))

print("running local search best")
start = time.time()
lsbest = LocalSearchSolver(instance, k, epsilon=10, greedy=False)
obj, medians, assignments, num_iter = lsbest.solve()
print("LS best opt: {}".format(obj))
print("LS best runtime: {}".format(time.time() - start))
print("LS best num_iter: {}".format(num_iter))

print("running local search best k-swap")
start = time.time()
lsbest = LocalSearchSolver(instance, k, swap_limit=k-1, epsilon=10, greedy=False)
obj, medians, assignments, num_iter = lsbest.solve()
print("LS best opt: {}".format(obj))
print("LS best runtime: {}".format(time.time() - start))
print("LS best num_iter: {}".format(num_iter))

#
# print(obj)
# print(medians)
# print(assignments)


visualize(random_points, assignments)
#ax = fig.add_subplot(projection='3d')

fig = plt.figure()
ax = fig.add_subplot()
x, y = random_points.T
ax.scatter(x, y, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')

#plt.show()