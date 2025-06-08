import random
import copy

class TabuSearchSolver:
    def __init__(self, problem, max_iter=100, tabu_tenure=10, neighborhood_size=20):
        self.problem = problem
        self.max_iter = max_iter
        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size

    def generate_initial_solution(self):
        return self.problem.generate_random_solution()

    def evaluate(self, solution):
        return self.problem.evaluate(solution)

    def get_neighbors(self, solution):
        neighbors = []
        for _ in range(self.neighborhood_size):
            neighbor = self.problem.perturb(solution)
            move = self.problem.get_move(solution, neighbor)
            neighbors.append((neighbor, move))
        return neighbors

    def run(self, track_schedule=False):
        tabu_list = []
        tabu_dict = {}
        history = []
        all_schedules = [] if track_schedule else None

        current_solution = self.generate_initial_solution()
        current_cost = self.evaluate(current_solution)
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost

        for iteration in range(self.max_iter):
            neighbors = self.get_neighbors(current_solution)
            neighbors = sorted(neighbors, key=lambda x: self.evaluate(x[0]))

            found_move = False
            for neighbor, move in neighbors:
                move_key = str(move)
                cost = self.evaluate(neighbor)

                if (move_key not in tabu_dict) or (cost < best_cost):  # Aspiration
                    current_solution = neighbor
                    current_cost = cost
                    if cost < best_cost:
                        best_solution = copy.deepcopy(neighbor)
                        best_cost = cost

                    tabu_list.append(move_key)
                    tabu_dict[move_key] = self.tabu_tenure
                    found_move = True
                    break

            # Decay Tabu Tenure
            expired = []
            for move in tabu_dict:
                tabu_dict[move] -= 1
                if tabu_dict[move] <= 0:
                    expired.append(move)
            for move in expired:
                tabu_list.remove(move)
                del tabu_dict[move]

            if not found_move:
                print("No valid non-tabu move found. Stopping early.")
                break

            history.append(best_cost)

            if track_schedule:
                schedule = self.problem.get_schedule(current_solution[:])
                all_schedules.append(schedule)

        return best_solution, best_cost, history, all_schedules
