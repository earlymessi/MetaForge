import numpy as np
import random
from metaforge.core.base_solver import BaseSolver

class GeneticAlgorithmSolver(BaseSolver):
    def __init__(self, problem, population_size=20, generations=50, crossover_rate=0.9, mutation_rate=0.2):
        super().__init__(problem)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.job_counts = [len(job.tasks) for job in self.problem.jobs]  # ✅ Updated for object model
        self.total_operations = sum(self.job_counts)

    def initialize_population(self):
        base = []
        for job_idx, count in enumerate(self.job_counts):
            base += [job_idx] * count
        population = [random.sample(base, len(base)) for _ in range(self.population_size)]
        return population

    def evaluate_population(self, population):
        return [self.problem.evaluate(ind) for ind in population]

    def select_parents(self, population, scores):
        selected = []
        for _ in range(self.population_size):
            i, j = random.sample(range(self.population_size), 2)
            winner = population[i] if scores[i] < scores[j] else population[j]
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]
        p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while parent2[p2_idx] in child[start:end] and child.count(parent2[p2_idx]) >= parent1.count(parent2[p2_idx]):
                    p2_idx += 1
                child[i] = parent2[p2_idx]
                p2_idx += 1
        return child

    def mutate(self, individual):
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]
        return individual

    def run(self, track_history=True, track_schedule=False):
        history = [] if track_history else None
        all_schedules = [] if track_schedule else None

        population = self.initialize_population()
        best_solution = None
        best_score = float("inf")

        for gen in range(self.generations):
            scores = self.evaluate_population(population)
            gen_best = min(scores)

            if track_history:
                history.append(gen_best)

            if gen_best < best_score:
                best_score = gen_best
                best_solution = population[scores.index(gen_best)]

            if track_schedule:
                schedule = self.problem.get_schedule(best_solution[:])
                all_schedules.append(schedule)

            selected = self.select_parents(population, scores)
            children = []

            for i in range(0, self.population_size, 2):
                p1, p2 = selected[i], selected[(i + 1) % self.population_size]
                if random.random() < self.crossover_rate:
                    c1 = self.crossover(p1, p2)
                    c2 = self.crossover(p2, p1)
                else:
                    c1, c2 = p1[:], p2[:]
                if random.random() < self.mutation_rate:
                    c1 = self.mutate(c1)
                if random.random() < self.mutation_rate:
                    c2 = self.mutate(c2)
                children.extend([c1, c2])

            population = children[:self.population_size]

        return best_solution, best_score, history, all_schedules
