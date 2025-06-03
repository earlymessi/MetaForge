class BaseSolver:
    def __init__(self, problem):
        self.problem = problem

    def solve(self):
        raise NotImplementedError("Each solver must implement the solve method.")
