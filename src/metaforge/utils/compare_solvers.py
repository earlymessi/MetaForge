import os
import time
import csv
import copy  # === 关键点2：导入 copy 模块 ===
import matplotlib.pyplot as plt
from sympy import false

from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.utils.timer import Timer
from metaforge.metaforge_runner import run_solver
from metaforge.utils.plotting import plot_solver_comparison
# === 关键点3：导入我们新的“主”绘图函数，替换旧的 ===
from metaforge.utils.visualization import plot_comprehensive_comparison
from metaforge.utils.pretty_names import pretty_names # pretty_names 在绘图函数内部加载，这里不需要


def compare_solvers(solver_names, problem, track_schedule=True, plot=False, dynamic_events=None):
    """
    运行并比较多个求解器，并在需要时自动生成一套完整的对比图表。
    """
    results = {}

    # 存储初始工件数量，以便之后传递给绘图函数
    initial_num_jobs = len(problem.jobs)

    for solver in solver_names:
        print(f"🔧 Running solver: {solver}...")

        # === 关键点2：为每个求解器创建问题的深拷贝，保证公平 ===
        problem_copy = copy.deepcopy(problem)
        events_copy = copy.deepcopy(dynamic_events) if dynamic_events else None

        start = time.time()
        # === 关键点1：将 dynamic_events 传递给 run_solver ===
        output = run_solver(
            solver,
            problem_copy,
            track_schedule=track_schedule,
            dynamic_events=events_copy

        )
        end = time.time()
        # --- 【核心修复】 ---
        # 在将 run_solver 的输出存入最终的 results 字典时，
        # 对 schedules 列表进行深拷贝，以确保其完全独立和安全。
        schedules_copy = copy.deepcopy(output.get("schedules"))
        results[solver] = {
            "best_score": output["makespan"],
            "runtime_sec": round(end - start, 2),
            "best_solution": output.get("solution"),
            "all_schedules": output.get("schedules"),
            "history": output.get("history")
        }
    if plot:
        plot_solver_comparison(results)
    # === 关键点3：当 plot=True 时，调用我们升级后的“主”绘图函数 ===
    elif plot == False:
        # 动态加载 pretty_names 以避免在顶层导入时可能产生的循环依赖问题
        from metaforge.utils.pretty_names import pretty_names
        plot_solver_comparison(results)
        # 调用这个函数，它会自动生成所有三个图表
        plot_comprehensive_comparison(
            results=results,
            problem=problem,  # 传递原始problem以获取机器数等信息
            pretty_names=pretty_names,
            dynamic_events=dynamic_events,
            initial_num_jobs=initial_num_jobs
        )

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
