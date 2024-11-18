import taichi as ti
import taichi.math as tm
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer



class KMediansInstance(ABC):
    def __init__(self, vertices):
        self.vertices = vertices
        
    @abstractmethod
    def metric(self, v1, v2):
        return 0
    
    def obj(self, clusters):
        vertices = self.vertices
        metric_matrix = np.zeros((len(vertices), len(clusters)))
        for x in range(len(vertices)):
            metric_matrix[x] = [self.metric(vertices[x], vertices[cluster]) for cluster in clusters]

        assignments = np.argmin(metric_matrix, axis=1)
        min_obj = sum(np.min(metric_matrix, axis=1))
        return min_obj, assignments
    
    def brute_force(self, k):
        best_medians = []
        best_assignments = []
        best_obj = float('inf')
        vertices = self.vertices
        for combo in combinations(range(len(vertices)), k):
            total_dist, assignments = self.obj(combo)
            if total_dist < best_obj:
                best_obj = total_dist
                best_medians = combo
                best_assignments = assignments
        return best_obj, best_medians, best_assignments
    
    def solve_lp_approx(self, k):
        return 0
    
    def solve_local_search(self, k):
        return 0
    
class EuclidianSpace(KMediansInstance):
    def __init__(self, vertices):
        super().__init__(vertices)
    
    def metric(self, v1, v2):
        return np.linalg.norm(v1 - v2)


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

n = 20
data_range = 100
random_points = np.random.rand(n, 2)*data_range


instance = EuclidianSpace(random_points)
k = 3
obj, medians, assignments = instance.brute_force(3)

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