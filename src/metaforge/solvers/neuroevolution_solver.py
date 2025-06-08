import torch
import torch.nn as nn
import numpy as np
import random
from copy import deepcopy
from metaforge.core.base_solver import BaseSolver

class EvoNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        sizes = [input_size] + hidden_sizes
        for in_s, out_s in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_s, out_s))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def get_flat_params(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat):
        pointer = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat[pointer:pointer + numel].view_as(p))
            pointer += numel


def evaluate_network(problem, net, device):
    job_counts = [len(job.tasks) for job in problem.jobs]  # ✅ updated
    num_jobs = len(job_counts)
    job_ptrs = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * problem.num_machines
    sequence = []

    while any(ptr < job_counts[j] for j, ptr in enumerate(job_ptrs)):
        available = [j for j in range(num_jobs) if job_ptrs[j] < job_counts[j]]
        state = torch.tensor(job_ptrs + job_ready, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = net(state)
            scores = logits.cpu().numpy()

        masked = np.full(num_jobs, -np.inf)
        for j in available:
            masked[j] = scores[j]

        selected = int(np.argmax(masked))
        sequence.append(selected)

        op_idx = job_ptrs[selected]
        task = problem.jobs[selected].tasks[op_idx]  # ✅ updated
        machine, proc_time = task.machine_id, task.duration
        start_time = max(machine_ready[machine], job_ready[selected])
        end_time = start_time + proc_time

        job_ptrs[selected] += 1
        job_ready[selected] = end_time
        machine_ready[machine] = end_time

    return sequence, problem.evaluate(sequence), problem.get_schedule(sequence)


class NeuroevolutionSolver(BaseSolver):
    def __init__(self, problem, pop_size=30, generations=50, mutation_rate=0.1, elite_size=2, hidden_sizes=[32, 32]):
        super().__init__(problem)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.hidden_sizes = hidden_sizes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_size = len(problem.jobs) * 2
        self.output_size = len(problem.jobs)
        self.model_template = EvoNetwork(self.input_size, hidden_sizes, self.output_size).to(self.device)
        self.param_size = len(self.model_template.get_flat_params())

    def initialize_population(self):
        return [torch.randn(self.param_size) for _ in range(self.pop_size)]

    def mutate(self, weights):
        noise = torch.randn_like(weights) * self.mutation_rate
        return weights + noise

    def crossover(self, parent1, parent2):
        mask = torch.rand_like(parent1) < 0.5
        child = torch.where(mask, parent1, parent2)
        return child

    def run(self, track_history=True, track_schedule=False):
        population = self.initialize_population()
        best_score = float("inf")
        best_solution = None
        best_schedule = None
        history = [] if track_history else None
        all_schedules = [] if track_schedule else None

        for gen in range(self.generations):
            scored = []
            for flat_params in population:
                net = deepcopy(self.model_template)
                net.set_flat_params(flat_params.to(self.device))
                sequence, score, _ = evaluate_network(self.problem, net, self.device)
                scored.append((flat_params, score, sequence))

            scored.sort(key=lambda x: x[1])
            elites = scored[:self.elite_size]

            if scored[0][1] < best_score:
                best_score = scored[0][1]
                best_solution = scored[0][2]
                _, _, best_schedule = evaluate_network(self.problem, deepcopy(self.model_template).to(self.device).eval().requires_grad_(False), self.device)

            if track_history:
                history.append(best_score)
            if track_schedule:
                all_schedules.append(best_schedule)

            next_gen = [e[0] for e in elites]
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(elites, 2)
                child = self.crossover(p1[0], p2[0])
                child = self.mutate(child)
                next_gen.append(child)

            population = next_gen

        return best_solution, best_score, history, all_schedules
