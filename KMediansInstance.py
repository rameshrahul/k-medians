from abc import ABC, abstractmethod
import numpy as np


class KMediansInstance(ABC):
    def __init__(self, vertices):
        self.vertices = vertices
        self.initial_medians = None

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


class TightInstance:

    def __init__(self):
        self.k = 0
        self.vertices = []
        self.distance_matrix = None
        self.initial_medians = []

    def initialize(self, k):
        if k % 2 == 0:
            raise ValueError("k must be odd.")

        self.k = k
        self._construct_instance()

    def build_distance_matrix(self):
        return self.distance_matrix

    def _construct_instance(self):
        k = self.k
        num_structures = (k - 1) // 2
        num_chains = (k + 1) // 2

        # Base structure
        structures = []
        for _ in range(num_structures):
            base_vertices = [len(self.vertices) + i for i in range(5)]
            v1, v2, v3, v4, v5 = base_vertices
            self.initial_medians.append(v1)
            self.vertices.extend(base_vertices)

            # Define distances
            structures.append({
                (v1, v2): 2, (v1, v3): 2, (v2, v4): 0, (v3, v5): 0
            })

        # Chain structure
        v = len(self.vertices)
        self.vertices.append(v)

        chains = []
        for i in range(num_chains):
            w = len(self.vertices)
            y = len(self.vertices) + 1

            self.initial_medians.append(y)

            self.vertices.extend([w, y])
            chains.append({
                (v, w): 1, (w, y): 1
            })

        # Create distance matrix
        num_vertices = len(self.vertices)
        self.distance_matrix = np.full((num_vertices, num_vertices), 100000)
        np.fill_diagonal(self.distance_matrix, 0)

        # Fill in base structures
        for structure in structures:
            for (i, j), dist in structure.items():
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist

        # Fill in chain structures
        for chain in chains:
            for (i, j), dist in chain.items():
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist

        self.vertices = np.asarray(self.vertices)

