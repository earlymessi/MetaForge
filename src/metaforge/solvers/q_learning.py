import random
import numpy as np
from metaforge.core.base_solver import BaseSolver
from collections import defaultdict

class QAgentSolver(BaseSolver):
    def __init__(self, problem, alpha=0.1, gamma=0.95, epsilon=0.1, episodes=200):
        super().__init__(problem)
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.episodes = episodes

        self.num_jobs = len(problem.jobs)
        self.job_counts = [len(job.tasks) for job in problem.jobs]  # ✅ updated

        self.q_table = defaultdict(lambda: np.zeros(self.num_jobs))

    def _get_initial_state(self):
        return tuple([0] * self.num_jobs)

    def _is_terminal(self, state):
        return all(state[j] >= self.job_counts[j] for j in range(self.num_jobs))

    def _get_available_jobs(self, state):
        return [j for j in range(self.num_jobs) if state[j] < self.job_counts[j]]

    def _get_next_state(self, state, action):
        state = list(state)
        state[action] += 1
        return tuple(state)

    def _simulate(self, job_sequence):
        return self.problem.evaluate(job_sequence)

    def run(self, track_history=True, track_schedule=False):
        best_sequence = None
        best_makespan = float("inf")
        history = [] if track_history else None
        all_schedules = [] if track_schedule else None

        for ep in range(self.episodes):
            state = self._get_initial_state()
            sequence = []

            while not self._is_terminal(state):
                available = self._get_available_jobs(state)
                if random.random() < self.epsilon:
                    action = random.choice(available)
                else:
                    q_values = self.q_table[state]
                    mask = np.full(self.num_jobs, -np.inf)
                    mask[available] = q_values[available]
                    action = int(np.argmax(mask))

                sequence.append(action)
                next_state = self._get_next_state(state, action)

                reward = -1
                if self._is_terminal(next_state):
                    makespan = self._simulate(sequence)
                    reward = -makespan
                    if makespan < best_makespan:
                        best_makespan = makespan
                        best_sequence = sequence[:]

                future_q = 0 if self._is_terminal(next_state) else np.max(self.q_table[next_state])
                self.q_table[state][action] += self.alpha * (reward + self.gamma * future_q - self.q_table[state][action])
                state = next_state

            if track_history:
                history.append(best_makespan)
            if track_schedule:
                all_schedules.append(self.problem.get_schedule(best_sequence[:]))

        return best_sequence, best_makespan, history, all_schedules