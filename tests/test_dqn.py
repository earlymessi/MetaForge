from pathlib import Path
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.utils.compare_solvers import compare_solvers
from metaforge.utils.visualization import plot_multiple_gantt

# === Pretty name mapping for solvers ===
pretty_names = {
    "dqn-naive": "DQN (naive)",
    "dqn-replay": "DQN (replay)",
}

# Load the problem instance
root_dir = Path(__file__).resolve().parents[1]  # MetaForge/
file_path = root_dir / "data" / "benchmarks" / "ft06.txt"
problem = load_job_shop_instance(str(file_path), format="orlib")

# Define only the solvers you want to compare
solvers = ["dqn-naive", "dqn-replay"]

# Run comparison and plot results
results = compare_solvers(solvers, problem, track_schedule=True, plot=True,)

# Print summary
for name, res in results.items():
    label = pretty_names.get(name, name)
    print(f"Solver: {label}")
    print(f"  Best Sequence : {res['best_solution']}")
    print(f"  Best Makespan: {res['best_score']}")
    print(f"  Runtime (sec): {res['runtime_sec']}")
    print()

# Plot Gantt charts for both solvers
plot_multiple_gantt(
    schedules_dict={
        "DQN (naive)": results["dqn-naive"]["all_schedules"][-1],
        "DQN (replay)": results["dqn-replay"]["all_schedules"][-1],
    },
    num_machines=problem.num_machines,
    num_jobs=len(problem.jobs)
)
