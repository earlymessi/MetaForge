import random
import math
from metaforge.core.base_solver import BaseSolver

class AntColonySolver(BaseSolver):
    def __init__(self, problem, num_ants=10, alpha=1.0, beta=2.0, evaporation=0.5, Q=100, iterations=100):
        super().__init__(problem)
        self.num_ants = num_ants
        self.alpha = alpha            # Influence of pheromone
        self.beta = beta              # Influence of heuristic (1/distance)
        self.evaporation = evaporation
        self.Q = Q                    # Pheromone deposit factor
        self.iterations = iterations

        self.job_counts = [len(job.tasks) for job in self.problem.jobs]  # ✅ Refactored
        self.total_ops = sum(self.job_counts)
        self.pheromone = [[1.0] * self.total_ops for _ in range(self.total_ops)]

    def generate_initial_sequence(self):
        base = []
        for job_idx, count in enumerate(self.job_counts):
            base += [job_idx] * count
        return base

    def construct_solution(self):
        base = self.generate_initial_sequence()
        solution = []
        unvisited = list(range(len(base)))
        visited_counts = {i: 0 for i in range(self.problem.num_jobs)}

        current = random.choice(unvisited)
        solution.append(base[current])
        unvisited.remove(current)
        visited_counts[base[current]] += 1

        while unvisited:
            probabilities = []
            for j in unvisited:
                from_idx = len(solution) - 1
                to_idx = j
                tau = self.pheromone[from_idx][to_idx] ** self.alpha
                eta = (1.0 / (visited_counts[base[j]] + 1)) ** self.beta
                probabilities.append(tau * eta)

            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]

            next_index = random.choices(unvisited, weights=probabilities, k=1)[0]
            solution.append(base[next_index])
            visited_counts[base[next_index]] += 1
            unvisited.remove(next_index)

        return solution

    def run(self, track_history=True, track_schedule=False):
        best_solution = None
        best_score = float("inf")
        history = [] if track_history else None
        all_schedules = [] if track_schedule else None

        for _ in range(self.iterations):
            ants = [self.construct_solution() for _ in range(self.num_ants)]
            scores = [self.problem.evaluate(a) for a in ants]

            # Update best
            for sol, score in zip(ants, scores):
                if score < best_score:
                    best_solution = sol
                    best_score = score

            if track_history:
                history.append(best_score)
            if track_schedule:
                all_schedules.append(self.problem.get_schedule(best_solution[:]))

            # Pheromone evaporation
            for i in range(len(self.pheromone)):
                for j in range(len(self.pheromone[i])):
                    self.pheromone[i][j] *= (1 - self.evaporation)

            # Pheromone deposit
            for sol, score in zip(ants, scores):
                for i in range(len(sol) - 1):
                    from_idx = i
                    to_idx = i + 1
                    self.pheromone[from_idx][to_idx] += self.Q / score

        return best_solution, best_score, history, all_schedules
