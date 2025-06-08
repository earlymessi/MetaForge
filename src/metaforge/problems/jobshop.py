import numpy as np
import random

class JobShopProblem:
    """
    Represents a job shop scheduling problem using object-oriented modeling.
    """

    def __init__(self, jobs):
        """
        Args:
            jobs (List[Job]): A list of Job objects.
        """
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_machines = max(task.machine_id for job in jobs for task in job.tasks) + 1
        self.machines = [Machine(i) for i in range(self.num_machines)]

    def evaluate(self, operation_order):
        """
        Evaluate the makespan for a given operation order.

        Args:
            operation_order (List[int]): List of job indices in operation order.

        Returns:
            int: The makespan.
        """
        job_ptr = [0] * self.num_jobs
        machine_ready_time = [0] * self.num_machines
        job_ready_time = [0] * self.num_jobs

        for job_idx in operation_order:
            job = self.jobs[job_idx]
            op_idx = job_ptr[job_idx]

            if op_idx >= len(job):
                continue

            task = job.tasks[op_idx]
            start_time = max(machine_ready_time[task.machine_id], job_ready_time[job_idx])
            end_time = start_time + task.duration

            machine_ready_time[task.machine_id] = end_time
            job_ready_time[job_idx] = end_time
            job_ptr[job_idx] += 1

        return max(job_ready_time)

    def get_schedule(self, operation_order):
        """
        Get detailed scheduling info (start/end times) for visualization.

        Args:
            operation_order (List[int]): Sequence of job indices.

        Returns:
            List[dict]: List of scheduled task dicts.
        """
        job_ptr = [0] * self.num_jobs
        machine_ready_time = [0] * self.num_machines
        job_ready_time = [0] * self.num_jobs
        schedule = []

        for job_idx in operation_order:
            job = self.jobs[job_idx]
            op_idx = job_ptr[job_idx]

            if op_idx >= len(job):
                continue

            task = job.tasks[op_idx]
            start_time = max(machine_ready_time[task.machine_id], job_ready_time[job_idx])
            end_time = start_time + task.duration

            schedule.append({
                "job": job_idx,
                "operation": op_idx,
                "machine": task.machine_id,
                "start": start_time,
                "end": end_time
            })

            machine_ready_time[task.machine_id] = end_time
            job_ready_time[job_idx] = end_time
            job_ptr[job_idx] += 1

        return schedule

    def generate_random_solution(self):
        """
        Generates a valid random job-based operation order.
        """
        operation_order = []
        for job_id, job in enumerate(self.jobs):
            operation_order += [job_id] * len(job)
        random.shuffle(operation_order)
        return operation_order

    def perturb(self, operation_order):
        """
        Generate a neighbor by swapping two random job positions.
        """
        neighbor = operation_order[:]
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    def get_move(self, old_solution, new_solution):
        """
        Return the move (swap) that generated the new solution.
        """
        for i in range(len(old_solution)):
            if old_solution[i] != new_solution[i]:
                for j in range(i + 1, len(old_solution)):
                    if (old_solution[i] == new_solution[j] and
                        old_solution[j] == new_solution[i]):
                        return (i, j)
        return None


class Task:
    def __init__(self, machine_id, duration, id=None):
        self.machine_id = machine_id
        self.duration = duration
        self.id = id

    def __repr__(self):
        return f"Task(machine={self.machine_id}, duration={self.duration}, id={self.id})"


class Job:
    def __init__(self, tasks, id=None):
        self.tasks = tasks  # List[Task]
        self.id = id

    def __len__(self):
        return len(self.tasks)

    def __repr__(self):
        return f"Job(id={self.id}, tasks={self.tasks})"


class Machine:
    def __init__(self, id):
        self.id = id
        self.calendar = []      # Reserved for future availability windows
        self.maintenance = []   # Reserved for future maintenance periods

    def __repr__(self):
        return f"Machine(id={self.id})"
