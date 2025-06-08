
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
file_path = root_dir / "data" / "benchmarks" / "ft06.txt"
#
problem = load_job_shop_instance(str(file_path), format="orlib")

# Define the solvers you want to compare
solvers = ["sa", "ts", "ga", "aco", "q", "dqn", "dqn-replay", "neuroevo"]

# Run comparison and plot results
results = compare_solvers(solvers, problem, track_schedule=True, plot=True)

# Print summary
for name, res in results.items():
    label = pretty_names.get(name, name)
    print(f"Solver: {label}")
    print(f"  Best Makespan: {res['best_score']}")
    print(f"  Runtime (sec): {res['runtime_sec']}")
    print()
