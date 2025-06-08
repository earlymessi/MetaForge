import argparse
import sys
import os

# Ensure the MetaForge directory is in the import path
sys.path.append(os.path.join(os.path.dirname(__file__), 'MetaForge'))

from utils.compare_solvers import compare_all_benchmarks
from utils.visualization import plot_results_from_csv, plot_runtime_from_csv


def main():
    parser = argparse.ArgumentParser(description="Run MetaForge solvers on multiple benchmarks.")
    parser.add_argument('--solvers', type=str, default="ts,ga,aco,dqn,dqn-replay,neuroevo",
                        help="Comma-separated list of solvers (e.g., ts,ga,dqn)")
    parser.add_argument('--benchmarks', type=str, default="MetaForge/problems/benchmarks",
                        help="Path to benchmark folder containing .txt files")
    parser.add_argument('--output', type=str, default="benchmark_results.csv",
                        help="Path to save CSV results")
    parser.add_argument('--plot', action='store_true', help="Generate plots after run")

    args = parser.parse_args()
    solver_list = [s.strip() for s in args.solvers.split(",")]

    print(f"\n🧠 Running solvers: {solver_list}")
    print(f"📁 Benchmarks from: {args.benchmarks}")
    print(f"📄 Output CSV: {args.output}")

    df = compare_all_benchmarks(
        solvers=solver_list,
        benchmark_folder=args.benchmarks,
        output_csv=args.output,
        track_schedule=False,
    )

    if args.plot:
        print("\n📊 Generating plots...")
        plot_results_from_csv(args.output)
        plot_runtime_from_csv(args.output)

if __name__ == "__main__":
    main()
