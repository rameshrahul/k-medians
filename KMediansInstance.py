from abc import ABC, abstractmethod
import numpy as np


class KMediansInstance(ABC):
    def __init__(self, vertices):
        self.vertices = vertices

    @abstractmethod
    def metric(self, v1, v2):
        return 0

    def build_distance_matrix(self):
        """Construct a distance matrix."""
        n = len(self.vertices)
        return [
            [self.metric(self.vertices[i], self.vertices[j]) for j in range(n)]
            for i in range(n)
        ]


class EuclideanInstance(KMediansInstance):
    def metric(self, v1, v2):
        return np.linalg.norm(v1 - v2)
