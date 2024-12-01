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


class IntegerProgramSolver(KMediansSolver):
    def solve(self):
        vertices = self.instance.vertices
        n = len(vertices)  # Number of vertices

        distances = self.distance_matrix

        with Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with Model("k-medians", env=env) as IPmod:
                # IPmod = Model("k-medians")

                x = IPmod.addVars(n, n, vtype=GRB.BINARY, name="x")
                y = IPmod.addVars(n, vtype=GRB.BINARY, name="y")

                IPmod.setObjective(quicksum(distances[i][j] * x[i, j] for i in range(n) for j in range(n)),
                                   GRB.MINIMIZE)

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
        if self.instance.initial_medians is None:
            current_medians = list(np.random.choice(vertices.shape[0], self.k, replace=False))
        else:
            current_medians = self.instance.initial_medians
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


class PrimalDualSolver(KMediansSolver):
    def __init__(self, instance, k, tolerance=1e-5, max_iterations=100):
        super().__init__(instance, k)
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def neighbors_i(self, i, v):
        U = range(len(self.instance.vertices))
        return [j for j in U if v[j] >= self.distance_matrix[i][j]]

    def neighbors_j(self, j, v):
        U = range(len(self.instance.vertices))
        return [i for i in U if v[j] >= self.distance_matrix[i][j]]

    def solve_with_lambda(self, lam):
        vertices = range(len(self.instance.vertices))
        num_vertices = len(vertices)
        v = np.zeros(num_vertices)  # Dual variable v_j initialized to 0 for all j
        w = np.zeros((num_vertices, num_vertices))  # Dual variable w_ij initialized to 0

        S = set(vertices)  # S starts as all vertices
        T = set()  # T starts as an empty set

        # Step 2: Primal-Dual Update
        while S:
            # Calculate the smallest increment to grow dual variables
            increment = np.inf

            for j in S:
                for i in vertices:
                    if i not in T:  # Tight dual inequality (v_j >= c_ij - w_ij)
                        if np.sum(w[i, :]) < lam:
                            remaining = lam - np.sum(w[i, :])
                            max_increment = remaining / len(S)  # Distribute increment uniformly across active variables
                            increment = min(increment, max_increment)
                for i in T:  # Neighbor condition (v_j >= c_ij)
                    increment = min(increment, self.distance_matrix[i][j] - v[j])

            # Increment dual variables
            for j in S:
                v[j] += increment
                for i in self.neighbors_j(j, v):
                    w[i][j] += increment

            # Check stopping conditions
            neighbors_added = set()
            for j in S:
                for i in T:
                    if v[j] >= self.distance_matrix[i][j]:  # Neighbor condition
                        neighbors_added.add(j)
                for i in vertices:
                    if i not in T and v[j] >= self.distance_matrix[i][j]:  # Tight dual inequality
                        T.add(i)
                        neighbors_added.update(self.neighbors_i(i, v))

            S.difference_update(neighbors_added)

        # Step 3: Initialize V
        V = set()

        # Step 4: Process T
        while T:
            i = T.pop()  # Pick any vertex in T
            V.add(i)

            # Remove all vertices from T that share neighbors with i
            neighbors_to_remove = set()
            for h in T:
                if any(w[i][j] > 0 and w[h][j] > 0 for j in vertices):
                    neighbors_to_remove.add(h)
            T.difference_update(neighbors_to_remove)

        return list(V)

    def solve(self):
        lambda_left = 0
        lambda_right = np.sum(self.distance_matrix)  # Upper bound for lambda
        best_medians = None

        iteration = 0
        while iteration < self.max_iterations:
            print("iterating")
            iteration += 1
            # Midpoint lambda
            lam = (lambda_left + lambda_right) / 2

            # Compute medians using the current lambda
            medians = self.solve_with_lambda(lam)
            print(lam, len(medians))

            # Check the size of the medians set
            if len(medians) == self.k:
                best_medians = medians
                break  # Exact solution found
            elif len(medians) > self.k:
                # Too many medians; increase lambda
                lambda_left = lam
            else:
                # Too few medians; decrease lambda
                lambda_right = lam

            # Check convergence
            if abs(lambda_right - lambda_left) < self.tolerance:
                break

        # Final lambda and medians
        total_dist, assignments = self.obj(best_medians)
        return total_dist, best_medians, assignments, iteration
