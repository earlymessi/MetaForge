import numpy as np

class JobShopProblem:
    def __init__(self, jobs):
        self.jobs = jobs

    def evaluate(self, schedule):
        # Dummy cost function, replace with real evaluation
        return np.sum(schedule)
