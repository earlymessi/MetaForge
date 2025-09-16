
from pathlib import Path
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.utils.compare_solvers import compare_solvers


# === Pretty name mapping for solvers ===
pretty_names = {
    "sa": "Simulated Annealing",
    "ts": "Tabu Search",
    "ga": "Genetic Algorithm",
    "aco": "Ant Colony Optimization",
    "q": "Q-Learning",
    "dqn-naive": "DQN (naive)",
    "dqn-replay": "DQN (replay)",
    "neuroevo": "Neuroevolution",
}

# Load the problem instance
root_dir = Path(__file__).resolve().parents[1]  # MetaForge/
file_path = root_dir / "data" / "benchmarks" / "ft10.txt"
#
problem = load_job_shop_instance(str(file_path), format="orlib")

# Define the solvers you want to compare
solvers = ["sa", "ts", "ga", "aco", "q", "dqn-naive", "dqn-replay", "neuroevo","dqn-dyn"]

# Run comparison and plot results
results = compare_solvers(solvers, problem, track_schedule=True, plot=True)

# Print summary
for name, res in results.items():
    label = pretty_names.get(name, name)
    print(f"Solver: {label}")
    print(f"  Best Makespan: {res['best_score']}")
    print(f"  Runtime (sec): {res['runtime_sec']}")
    print()


from metaforge.utils.visualization import plot_multiple_gantt

plot_multiple_gantt(
    schedules_dict={
        "SA": results["sa"]["all_schedules"][-1],
        "TS": results["ts"]["all_schedules"][-1],
        "GA": results["ga"]["all_schedules"][-1],
        "ACO": results["aco"]["all_schedules"][-1],
        "Q-Learning": results["q"]["all_schedules"][-1],
        "DQN (naive)": results["dqn-naive"]["all_schedules"][-1],
        "DQN (replay)": results["dqn-replay"]["all_schedules"][-1],
        "Neuroevo": results["neuroevo"]["all_schedules"][-1],
        "DQN (dyn)": results["dqn-dyn"]["all_schedules"][-1],
    },
    num_machines=problem.num_machines,
    num_jobs=len(problem.jobs)
)


