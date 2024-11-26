from abc import ABC, abstractmethod
import numpy as np
from itertools import combinations

from gurobipy import *


class KMediansSolver(ABC):
    def __init__(self, instance, k):
        self.instance = instance
        self.k = k
        self.distance_matrix = instance.build_distance_matrix()

    @abstractmethod
    def solve(self):
        return 0

    def obj(self, clusters):
        vertices = self.instance.vertices
        metric_matrix = np.zeros((len(vertices), len(clusters)))
        for x in range(len(vertices)):
            metric_matrix[x] = [self.distance_matrix[x][cluster] for cluster in clusters]

        assignments = np.argmin(metric_matrix, axis=1)
        min_obj = sum(np.min(metric_matrix, axis=1))
        return min_obj, assignments


class BruteForceSolver(KMediansSolver):
    def solve(self):
        k = self.k
        best_medians = []
        best_assignments = []
        best_obj = float('inf')
        vertices = self.instance.vertices
        for combo in combinations(range(len(vertices)), k):
            total_dist, assignments = self.obj(combo)
            if total_dist < best_obj:
                best_obj = total_dist
                best_medians = combo
                best_assignments = assignments
        return best_obj, best_medians, best_assignments


class IntegerProgram(KMediansSolver):
    def solve(self):
        vertices = self.instance.vertices
        n = len(vertices)  # Number of vertices

        distances = self.distance_matrix

        IPmod = Model("k-medians")

        x = IPmod.addVars(n, n, vtype=GRB.BINARY, name="x")
        y = IPmod.addVars(n, vtype=GRB.BINARY, name="y")

        IPmod.setObjective(quicksum(distances[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

        IPmod.addConstrs(
            (quicksum(x[i, j] for j in range(n)) >= 1 for i in range(n)),
            name="vertex"
        )

        IPmod.addConstrs(
            (x[i, j] <= y[j] for i in range(n) for j in range(n)),
            name="ij"
        )

        IPmod.addConstr(
            quicksum(y[j] for j in range(n)) == self.k,
            name="k"
        )

        IPmod.optimize()

        if IPmod.status == GRB.OPTIMAL:
            optimal_objective = IPmod.objVal
            selected_medians = [j for j in range(n) if y[j].x > 0.5]
            min_obj, assignments = self.obj(selected_medians)
            return optimal_objective, selected_medians, assignments
        else:
            return "Could not find optimal"
