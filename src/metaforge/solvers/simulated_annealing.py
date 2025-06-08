import numpy as np
import random
from metaforge.core.base_solver import BaseSolver

class SimulatedAnnealingSolver(BaseSolver):
    """
    Simulated Annealing solver for job-shop scheduling using object-based problem model.
    """

    def __init__(self, problem, initial_temp=1000, cooling_rate=0.95, max_iterations=1000):
        super().__init__(problem)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.job_counts = [len(job.tasks) for job in self.problem.jobs]  # updated to use job.tasks
        self.total_operations = sum(self.job_counts)

    def initialize_solution(self):
        base = []
        for job_idx, count in enumerate(self.job_counts):
            base += [job_idx] * count
        return random.sample(base, len(base))

    def get_neighbor(self, solution):
        neighbor = solution[:]
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    def run(self, track_schedule=False):
        current_solution = self.initialize_solution()
        current_score = self.problem.evaluate(current_solution)
        best_solution = current_solution[:]
        best_score = current_score

        temp = self.initial_temp
        history = []
        temp_history = []
        all_schedules = [] if track_schedule else None

        for _ in range(self.max_iterations):
            neighbor = self.get_neighbor(current_solution)
            neighbor_score = self.problem.evaluate(neighbor)
            delta = neighbor_score - current_score

            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_solution = neighbor
                current_score = neighbor_score

                if current_score < best_score:
                    best_solution = current_solution[:]
                    best_score = current_score

            history.append(best_score)
            temp_history.append(temp)

            if track_schedule:
                schedule = self.problem.get_schedule(current_solution[:])
                all_schedules.append(schedule)

            temp *= self.cooling_rate
            if temp < 1e-5:
                break

        return best_solution, best_score, history, temp_history, all_schedules
