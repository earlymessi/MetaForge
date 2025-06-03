import numpy as np
from MetaForge.core.base_solver import BaseSolver

class GeneticAlgorithmSolver(BaseSolver):
    def __init__(self, problem, population_size=10, generations=20):
        super().__init__(problem)
        self.population_size = population_size
        self.generations = generations

    def solve(self):
        best_solution = None
        best_score = float("inf")
        for _ in range(self.generations):
            candidate = np.random.permutation(len(self.problem.jobs))
            score = self.problem.evaluate(candidate)
            if score < best_score:
                best_score = score
                best_solution = candidate
        return best_solution, best_score
