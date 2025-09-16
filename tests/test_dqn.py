import copy
from pathlib import Path

# 注意：不再需要 from sympy import false
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.utils.compare_solvers import compare_solvers
# 确保导入了正确的绘图函数（尽管它会被自动调用）
from metaforge.utils.visualization import plot_comprehensive_comparison
from metaforge.problems.jobshop import Job, Task

# === 1. 名称映射 ===
pretty_names = {
    "dqn-dyn": "DQN (Dynamic)"
}

# === 2. 加载基础问题实例 ===
root_dir = Path(__file__).resolve().parents[1]
file_path = root_dir / "data" / "benchmarks" / "ft06.txt"
problem = load_job_shop_instance(str(file_path), format="orlib")

problem.initial_jobs = copy.deepcopy(problem.jobs)
problem.initial_num_jobs = len(problem.jobs)
initial_num_jobs = len(problem.jobs)

# === 3. 定义动态事件 ===
new_job = Job(tasks=[
    Task(machine_id=5, duration=5),
    Task(machine_id=1, duration=5),
    Task(machine_id=2, duration=5),
])
new_job2 = Job(tasks=[
    Task(machine_id=0, duration=4),
    Task(machine_id=3, duration=6),
    Task(machine_id=4, duration=7),
])

dynamic_events = [
    {'type': 'machine_breakdown_on_nth_task', 'trigger': {'machine': 2, 'task_count': 3}, 'duration': 15},
    {'type': 'machine_breakdown_on_nth_task', 'trigger': {'machine': 4, 'task_count': 5}, 'duration': 8},
    {'type': 'new_job_arrival', 'time': 20, 'job': new_job},
    {'type': 'new_job_arrival', 'time': 35, 'job': new_job2},
]

# === 4. 定义求解器 ===
solvers = [
    "dqn-dyn"
]

# === 5. 运行综合对比（并自动绘图） ===
print("🚀 开始在动态场景下对所有求解器进行综合对比...")
# 关键：确保 plot=True 以触发自动绘图
results = compare_solvers(solvers, problem, track_schedule=True, plot=False, dynamic_events=dynamic_events)

# === 6. 打印摘要 ===
print("\n--- 综合结果摘要 ---")
for name, res in results.items():
    if res and res.get('best_score') is not None:
        label = pretty_names.get(name, name)
        print(f"Solver: {label}")
        print(f"  Best Makespan: {res['best_score']:.2f}")
        print(f"  Runtime (sec): {res['runtime_sec']:.2f}")
        # print(f"  Solution: {res['best_solution']}") # 解决方案序列通常太长，可以不打印
        print()
    else:
        print(f"求解器: {pretty_names.get(name, name)} 未能成功运行或返回结果。")

# === 7. 结束语 ===
print("✅ 实验完成。绘图已由 `compare_solvers` 自动处理。")