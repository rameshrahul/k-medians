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


def evaluate_k_medians_solvers(instances, solvers, ks, local_search_params):
    results = []

    for i, instance in enumerate(instances):
        print(f"Evaluating instance {i + 1}/{len(instances)}...")

        for k in ks:
            print(f"  Testing with k = {k}...")
            if isinstance(instance, TightInstance):
                instance = TightInstance()
                instance.initialize(k)

            # Calculate the optimal solution using the integer programming solver
            ip_solver = solvers["IntegerProgramSolver"](instance, k)
            start_time = time.time()
            optimal_value, _, _ = ip_solver.solve()
            ip_runtime = time.time() - start_time

            print(f"\tOptimal solution found (IP solver): {optimal_value} in {ip_runtime:.2f} seconds.")

            # Evaluate all solvers
            for solver_name, solver_class in solvers.items():
                if solver_name == "IntegerProgramSolver":
                    continue  # Skip re-running the integer program solver

                if solver_name == "LocalSearchSolver":
                    # Test LocalSearchSolver with different parameters
                    for params in local_search_params:
                        print(f"\tTesting {solver_name} with params {params}...")
                        solver = solver_class(instance, k, **params)

                        start_time = time.time()
                        solution_value, _, _, num_iter = solver.solve()
                        runtime = time.time() - start_time
                        approximation_ratio = solution_value / optimal_value

                        results.append({
                            "instance": i,
                            "solver": solver_name,
                            "k": k,
                            "params": params,
                            "runtime": runtime,
                            "approximation_ratio": approximation_ratio,
                            "num_iterations": num_iter
                        })

                        print(f"\tRuntime: {runtime:.2f}s | Approximation Ratio: {approximation_ratio:.4f} | "
                              f"Iterations: {num_iter}")
                elif solver_name == "PrimalDualSolver":
                    print(f"\tTesting {solver_name}...")
                    solver = solver_class(instance, k)

                    start_time = time.time()
                    solution_value, _, _, num_iter = solver.solve()
                    runtime = time.time() - start_time
                    approximation_ratio = solution_value / optimal_value

                    results.append({
                        "instance": i,
                        "solver": solver_name,
                        "k": k,
                        "params": None,
                        "num_ier": num_iter,
                        "runtime": runtime,
                        "approximation_ratio": approximation_ratio
                    })

                    print(f"\tRuntime: {runtime:.2f}s | Approximation Ratio: {approximation_ratio:.4f}| "
                          f"Iterations: {num_iter}")
                else:
                    # Test other solvers without additional parameters
                    print(f"\tTesting {solver_name}...")
                    solver = solver_class(instance, k)

                    start_time = time.time()
                    solution_value, _, _ = solver.solve()
                    runtime = time.time() - start_time
                    approximation_ratio = solution_value / optimal_value

                    results.append({
                        "instance": i,
                        "solver": solver_name,
                        "k": k,
                        "params": None,
                        "runtime": runtime,
                        "approximation_ratio": approximation_ratio
                    })

                    print(f"\tRuntime: {runtime:.2f}s | Approximation Ratio: {approximation_ratio:.4f}")

    return results


# Fixing random state for reproducibility
# np.random.seed(123123) #19680801

n = 100
data_range = 100
random_points = np.random.rand(n, 2) * data_range

ks = [3, 13, 55]
#instances = [EuclideanInstance(random_points)]
instances = [TightInstance()]


# Define solvers
solvers = {"IntegerProgramSolver": IntegerProgramSolver, "LocalSearchSolver": LocalSearchSolver,
           }

# Define LocalSearchSolver parameters to test
local_search_params = [
    {"epsilon": 10, "greedy": True, "swap_limit": 1},
    {"epsilon": 20, "greedy": True, "swap_limit": 1},
    {"epsilon": 100, "greedy": True, "swap_limit": 1},
    {"epsilon": 10, "greedy": False, "swap_limit": 1},
    {"epsilon": 20, "greedy": False, "swap_limit": 1},
    {"epsilon": 100, "greedy": False, "swap_limit": 1},
    {"epsilon": 10, "greedy": True, "swap_limit": 2},
    {"epsilon": 20, "greedy": True, "swap_limit": 2},
    {"epsilon": 100, "greedy": True, "swap_limit": 2},
    {"epsilon": 10, "greedy": False, "swap_limit": 2},
    {"epsilon": 20, "greedy": False, "swap_limit": 2},
    {"epsilon": 100, "greedy": False, "swap_limit": 2}
]

# Run evaluation
results = evaluate_k_medians_solvers(instances, solvers, ks, local_search_params)

# visualize(random_points, assignments)
# #ax = fig.add_subplot(projection='3d')
#
# fig = plt.figure()
# ax = fig.add_subplot()
# x, y = random_points.T
# ax.scatter(x, y, marker='o')
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')

# plt.show()
