import taichi as ti
import taichi.math as tm
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer



class KMediansInstance(ABC):
    def __init__(self, vertices):
        self.vertices = vertices
        
    @abstractmethod
    def metric(v1, v2):
        return 0
    
    def solve_ip(self, k):
        return 0
    
    def solve_lp_approx(self, k):
        return 0
    
    def solve_local_search(self, k):
        return 0
    
class EuclidianSpace(KMediansInstance):
    def __init__(self, vertices):
        super().__init__(vertices)
    
    def metric(v1, v2):
        return np.linalg.norm(v1 - v2)



# Fixing random state for reproducibility
np.random.seed(19680801)

random_points = np.random.rand(100, 3)*100

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x, y, z = random_points.T
ax.scatter(x, y, z, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()