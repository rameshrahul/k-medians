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


class LocalSearchSolver(KMediansSolver):
    def __init__(self, instance, k, swap_limit=1, epsilon=1e-6, greedy=True):
        super().__init__(instance, k)
        self.swap_limit = swap_limit
        self.epsilon = epsilon
        self.greedy = greedy

    def solve(self):
        vertices = self.instance.vertices
        current_medians = list(np.random.choice(vertices.shape[0], self.k, replace=False))
        current_objective, current_assignments = self.obj(current_medians)

        improved = True
        num_iter = 0
        while improved:
            improved = False
            best_swap_out = None
            best_swap_in = None
            best_objective = current_objective

            non_medians = [i for i in range(len(vertices)) if i not in current_medians]

            for swap_out_set in combinations(current_medians, min(self.swap_limit, len(current_medians))):
                for swap_in_set in combinations(non_medians, len(swap_out_set)):
                    # Evaluate candidate solution
                    candidate_medians = current_medians[:]
                    for swap_out in swap_out_set:
                        candidate_medians.remove(swap_out)
                    candidate_medians.extend(swap_in_set)

                    candidate_objective, _ = self.obj(candidate_medians)

                    improvement = current_objective - candidate_objective
                    if improvement > self.epsilon:
                        if self.greedy:
                            # Greedy mode: Take the first improving swap
                            current_medians = candidate_medians
                            current_objective, current_assignments = self.obj(current_medians)
                            improved = True
                            break
                        else:
                            # Best mode: Track the best swap
                            if candidate_objective < best_objective:
                                best_swap_out = swap_out_set
                                best_swap_in = swap_in_set
                                best_objective = candidate_objective
                if improved and self.greedy:
                    break

            # Execute the best swap if no greedy improvement was found
            if not self.greedy and best_swap_out and best_swap_in:
                for swap_out in best_swap_out:
                    current_medians.remove(swap_out)
                current_medians.extend(best_swap_in)
                current_objective, current_assignments = self.obj(current_medians)
                improved = True
            num_iter += 1

        # Return the results
        return current_objective, current_medians, current_assignments, num_iter

