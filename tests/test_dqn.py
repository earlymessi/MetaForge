import copy
from pathlib import Path


# 注意：不再需要 from sympy import false
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.utils.compare_solvers import compare_solvers
# 确保导入了正确的绘图函数（尽管它会被自动调用）
from metaforge.utils.visualization import plot_comprehensive_comparison
from metaforge.problems.jobshop import Job, Task, JobShopProblem

# === 1. 名称映射 ===
pretty_names = {
    "dqn-dyn": "DQN (Dynamic)"
}

# === 2. 加载基础问题实例 ===
root_dir = Path(__file__).resolve().parents[1]
file_path = root_dir / "data" / "benchmarks" / "ft06.txt"
# 这是一个临时的 problem 对象，我们只用它来获取 jobs 列表
temp_problem = load_job_shop_instance(str(file_path), format="orlib")
initial_jobs_from_file = temp_problem.jobs

# === 3. 【核心修改】为加载的 Jobs 和新 Jobs 添加交付时间 ===

# 【新增】为从文件中加载的初始Jobs设置交付时间
# 所有初始Jobs都在时间0到达
for job in initial_jobs_from_file:
    job.arrival_time = 0
    # 使用我们在Job类中定义的方法自动计算交付日期
    # due_date = arrival_time + total_duration * k_factor
    total_processing_time = sum(task.duration for task in job.tasks)
    job.due_date = job.arrival_time + total_processing_time * 3.0 # 使用一个比较紧的宽松系数，例如 2.0

# 【新增】定义动态事件，并为新Job设置交付时间
# 这个新Job在时间20到达
new_job_arrival_time = 20
new_job = Job(
    tasks=[
        Task(machine_id=5, duration=5),
        Task(machine_id=1, duration=5),
        Task(machine_id=2, duration=5),
    ],
    id=len(initial_jobs_from_file), # 给一个唯一的ID
    arrival_time=new_job_arrival_time,
    due_date_k_factor=5 # 为这个新job使用不同的宽松系数
)
print(f"New Job 1 created with due date: {new_job.due_date}")

# 这个新Job在时间35到达
new_job2_arrival_time = 35
new_job2 = Job(
    tasks=[
        Task(machine_id=0, duration=4),
        Task(machine_id=3, duration=6),
        Task(machine_id=4, duration=7),
    ],
    id=len(initial_jobs_from_file) + 1,
    arrival_time=new_job2_arrival_time,
    due_date_k_factor=5
)
print(f"New Job 2 created with due date: {new_job2.due_date}")


dynamic_events = [
    {'type': 'machine_breakdown_on_nth_task', 'trigger': {'machine': 2, 'task_count': 3}, 'duration': 15},
    {'type': 'machine_breakdown_on_nth_task', 'trigger': {'machine': 4, 'task_count': 5}, 'duration': 8},
    {'type': 'new_job_arrival', 'time': new_job_arrival_time, 'job': new_job},
    {'type': 'new_job_arrival', 'time': new_job2_arrival_time, 'job': new_job2},
]

# === 4. 【核心修改】重新实例化 JobShopProblem 以包含成本参数 ===

# 【新增】创建一个包含所有丰富信息（jobs带due_date，以及成本参数）的最终problem对象
problem = JobShopProblem(
    jobs=copy.deepcopy(initial_jobs_from_file),
    tardiness_penalty_per_unit=100, # 每单位延误时间，罚款100
    overtime_cost_per_unit=15       # 每单位加班时间，成本15
)

# 仍然需要为DQN求解器保留初始状态的引用
problem.initial_jobs = copy.deepcopy(problem.jobs)

print("\n--- Problem Setup ---")
print(f"Initial jobs: {len(problem.initial_jobs)}")
print(f"Tardiness Penalty: {problem.TARDINESS_PENALTY}")
print(f"Overtime Cost: {problem.OVERTIME_COST}")
for job in problem.jobs:
    print(f"  - Job {job.id}: Arrival={job.arrival_time}, Due Date={job.due_date}")
print("---------------------\n")


# === 5. 定义求解器 ===
solvers = [
    "dqn-dyn"
]

# === 6. 运行综合对比 ===
print("🚀 开始在动态场景下对所有求解器进行综合对比...")
results = compare_solvers(solvers, problem, track_schedule=True, plot=False, dynamic_events=dynamic_events)

# === 7. 打印摘要 ===
print("\n--- 综合结果摘要 ---")
for name, res in results.items():
    if res and res.get('best_score') is not None:
        # 注意：这里的 'best_score' 仍然是 makespan。
        # 你的DQN奖励函数会优化一个复合目标（延误+成本），但compare_solvers的评估标准默认是makespan。
        # 你可能需要自定义评估逻辑或关注DQN内部的奖励值来判断真实性能。
        label = pretty_names.get(name, name)
        print(f"Solver: {label}")
        print(f"  Best Makespan: {res['best_score']:.2f}")
        print(f"  Runtime (sec): {res['runtime_sec']:.2f}")
        print()
    else:
        print(f"求解器: {pretty_names.get(name, name)} 未能成功运行或返回结果。")

# === 8. 结束语 ===
print("✅ 实验完成。")