import numpy as np
import random

class JobShopProblem:
    """
    Represents a job shop scheduling problem using object-oriented modeling.
    """

    def __init__(self, jobs, tardiness_penalty_per_unit=100, overtime_cost_per_unit=10):
        """
        Args:
            jobs (List[Job]): A list of Job objects.
        """
        """
               【修改】初始化函数，增加了成本相关的参数。

               Args:
                   jobs (List[Job]): A list of Job objects.
                   tardiness_penalty_per_unit (float): 每单位延误时间的惩罚成本.
                   overtime_cost_per_unit (float): 每单位加班时间的额外成本.
               """
        self.jobs = jobs
        self.num_jobs = len(jobs)
        # 【修改】确保即使有新job加入，也能正确计算机器数量
        all_tasks = [task for job in jobs for task in job.tasks]
        self.num_machines = max(task.machine_id for job in jobs for task in job.tasks) + 1
        self.machines = [Machine(i) for i in range(self.num_machines)]
        # 【新增】存储成本参数，供DQN环境和奖励函数使用
        self.TARDINESS_PENALTY = tardiness_penalty_per_unit
        self.OVERTIME_COST = overtime_cost_per_unit

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
    def __init__(self, tasks, id=None, arrival_time=0, due_date=None, due_date_k_factor=2.5):
        """
                【修改】初始化函数，增加了 arrival_time 和 due_date。

                Args:
                    tasks (List[Task]): 工序列表.
                    id (any, optional): Job的ID.
                    arrival_time (int, optional): Job到达车间的时间. 默认为 0.
                    due_date (int, optional): Job必须完成的交付日期. 如果为None，会自动计算.
                    due_date_k_factor (float, optional): 用于自动计算交付日期的宽松系数.
                                                         due_date = arrival_time + total_duration * k_factor.
                """
        self.tasks = tasks  # List[Task]
        self.id = id
        self.arrival_time = arrival_time
        # 【新增】为 Job 设置交付日期 (due_date)
        if due_date is not None:
            self.due_date = due_date
        else:
            # 如果未提供due_date，则根据总处理时间自动计算
            total_processing_time = sum(task.duration for task in self.tasks)
            self.due_date = self.arrival_time + total_processing_time * due_date_k_factor
    def __len__(self):
        return len(self.tasks)

    def __repr__(self):
        return f"Job(id={self.id}, tasks={self.tasks})"


class Machine:
    def __init__(self, id):
        self.id = id
        # 【新增】机器的工作模式和成本率
        self.mode = 'normal'  # 'normal', 'overtime', 'maintenance'
        self.cost_rate_multiplier = {
            'normal': 1.0,  # 标准成本系数
            'overtime': 1.5  # 加班成本系数是标准的1.5倍
        }
        self.calendar = []      # Reserved for future availability windows
        self.maintenance = []   # Reserved for future maintenance periods

    def get_current_cost_rate(self):
        """获取当前模式下的成本率乘数"""
        return self.cost_rate_multiplier.get(self.mode, 1.0)
    def __repr__(self):
        return f"Machine(id={self.id})"
