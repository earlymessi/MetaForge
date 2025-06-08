import os
import time
import csv
import matplotlib.pyplot as plt
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.utils.timer import Timer
from metaforge.metaforge_runner import run_solver

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

def compare_solvers(solver_names, problem, track_schedule=True, plot=True):
    """
    Run and compare multiple solvers on the same job shop problem instance.

    Args:
        solver_names (List[str]): List of solver identifiers (e.g., ["ts", "aco", "dqn"]).
        problem (JobShopProblem): The job shop problem instance.
        track_schedule (bool): Whether to collect history and best schedules.
        plot (bool): Whether to display visual comparisons.

    Returns:
        Dict[str, Dict]: A mapping of solver name to its results:
            {
                "solver_name": {
                    "best_score": ...,
                    "runtime_sec": ...,
                    "best_solution": ...,
                    "all_schedules": ...,
                    "history": ...
                },
                ...
            }
    """
    results = {}

    for solver in solver_names:
        print(f"🔧 Running solver: {solver}...")
        start = time.time()
        output = run_solver(solver, problem, track_schedule=track_schedule)
        end = time.time()

        results[solver] = {
            "best_score": output["makespan"],
            "runtime_sec": round(end - start, 2),
            "best_solution": output.get("solution"),
            "all_schedules": output.get("schedules"),
            "history": output.get("history")
        }

    if plot:
        plot_comparison(results)

    return results


def plot_comparison(results):
    """
    Plot convergence and runtime comparison of solver results.
    """
    plt.figure(figsize=(10, 5))

    # Plot convergence history
    plt.subplot(1, 2, 1)
    for solver, res in results.items():
        history = res.get("history", [])
        if history:
            label = pretty_names.get(solver, solver)
            plt.plot(history, label=f"{label} (final: {res['best_score']})")
    plt.title("Convergence (Makespan)")
    plt.xlabel("Iteration")
    plt.ylabel("Makespan")
    plt.legend()
    plt.grid(True)

    # Plot runtime
    plt.subplot(1, 2, 2)
    solvers = list(results.keys())
    times = [results[s]["runtime_sec"] for s in solvers]
    plt.bar(solvers, times, color='gray')
    plt.title("Runtime (sec)")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def compare_all_benchmarks(
    benchmark_folder,
    solvers,
    format="orlib",
    output_csv="results/benchmark_comparison.csv",
    track_schedule=False,
    plot=False
):
    if output_csv:
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    benchmark_files = [
        f for f in os.listdir(benchmark_folder)
        if f.endswith(".txt")
    ]
    benchmark_files.sort()

    results = []

    for benchmark_file in benchmark_files:
        path = os.path.join(benchmark_folder, benchmark_file)
        problem = load_job_shop_instance(path, format=format)

        for solver in solvers:
            solver_label = pretty_names.get(solver, solver)
            print(f"Running {solver_label} on {benchmark_file}...")
            timer = Timer()
            result  = run_solver(
                solver,
                problem,
                track_schedule=track_schedule
            )
            elapsed = timer.stop()

            results.append({
                "benchmark": benchmark_file,
                "solver": solver_label,
                "best_score": result["makespan"],
                "runtime_sec": elapsed,
                "best_solution": result["solution"],
                "all_schedules": result["schedules"],
                "history": result["history"],
            })
    
    # Write valid fields results to CSV
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["benchmark", "solver", "best_score", "runtime_sec"])
            writer.writeheader()
            for row in results:
                writer.writerow({
                    "benchmark": row["benchmark"],
                    "solver": row["solver"],
                    "best_score": row["best_score"],
                    "runtime_sec": row["runtime_sec"],
                })
        print(f"\n✅ All results saved to {output_csv}")

    # Optional plotting
    if plot:
        from metaforge.utils.visualization import (
            plot_results_from_csv,
            plot_runtime_from_csv,
        )
        plot_results_from_csv(output_csv)
        plot_runtime_from_csv(output_csv)

    return results


if __name__ == "__main__":
    compare_all_benchmarks(
        benchmark_folder="data/benchmarks",  # update path if needed
        solvers=["ts", "ga", "aco"],
        output_csv="results/test_output.csv",
        track_schedule=True,
        plot=True
    )

input("📈 Done! Press Enter to close plots...")