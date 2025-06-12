import os
import time
import csv
import matplotlib.pyplot as plt
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.utils.timer import Timer
from metaforge.metaforge_runner import run_solver
from metaforge.utils.plotting import plot_solver_comparison
from metaforge.utils.pretty_names import pretty_names


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
        plot_solver_comparison(results)

    return results


def compare_all_benchmarks(
    benchmark_source,
    solvers,
    format="orlib",
    output_csv="results/benchmark_comparison.csv",
    track_schedule=False,
    plot=False
):
    # Determine if source is a URL or local path
    is_url = benchmark_source.startswith("http://") or benchmark_source.startswith("https://")

    if output_csv:
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # Define benchmark files (common ORLib ones by default)
    benchmark_files = [
        "ft06.txt", "ft10.txt", "ft20.txt",
        "la01.txt", "la02.txt", "la03.txt",
        "la04.txt", "la05.txt"
    ]

    results = []

    for benchmark_file in benchmark_files:
        path = (
            f"{benchmark_source.rstrip('/')}/{benchmark_file}" if is_url
            else os.path.join(benchmark_source, benchmark_file)
        )

        try:
            problem = load_job_shop_instance(path, format=format)
        except Exception as e:
            print(f"⚠️ Failed to load {benchmark_file}: {e}")
            continue

        for solver in solvers:
            solver_label = pretty_names.get(solver, solver)
            print(f"Running {solver_label} on {benchmark_file}...")

            timer = Timer()
            result = run_solver(
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

    # Write summary results to CSV
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
