# 🚀 MetaForge Usage Guide

MetaForge lets you test, compare, and visualize optimization solvers on Job Shop Scheduling Problems (JSSP) — both classic benchmarks and your own custom datasets.

---

## ⚙️ 1. Installation

Install MetaForge via pip:

```bash
pip install metaforge
```

Or directly from GitHub:

```bash
pip install git+https://github.com/mageed-ghaleb/metaforge.git
```

---

## 🧪 2. Run One Solver on One Problem

To quickly test solvers on a single benchmark, use:

```bash
python tests/test_compare_solvers.py
```

This script:
- Loads the `ft06.txt` problem
- Runs all solvers listed in the script
- Plots convergence and runtime
- Prints makespan and runtime for each solver

✅ You can edit the file to:
- Run only one solver (e.g. `"ga"`)
- Change the benchmark file
- Enable/disable plotting

---

## 📊 3. Compare All Solvers on All Benchmarks

Use the batch comparison entrypoint in:

```bash
python -m src.metaforge.utils.compare_solvers
```

This:
- Runs all solvers on all `.txt` files in `data/benchmarks/`
- Saves results to `results/benchmark_comparison.csv`
- Plots:
  - Best Makespan per solver
  - Runtime per solver

---

### 🔧 To customize:
Edit the last section of `compare_solvers.py`:

```python
if __name__ == "__main__":
    compare_all_benchmarks(
        benchmark_folder="data/benchmarks",
        solvers=["ts", "ga", "aco"],
        output_csv="results/my_run.csv",
        track_schedule=False,
        plot=True
    )
```

---

## 📂 4. CSV Format

The output CSV will contain:

| Column         | Description                     |
|----------------|---------------------------------|
| `benchmark`    | Benchmark file name (e.g., `ft06.txt`) |
| `solver`       | Solver name                     |
| `best_score`   | Best makespan found             |
| `runtime_sec`  | Runtime in seconds              |

---

## 📈 5. Plotting Results

To visualize a saved CSV at any time:

```python
from metaforge.utils.visualization import (
    plot_results_from_csv,
    plot_runtime_from_csv
)

plot_results_from_csv("results/my_run.csv")
plot_runtime_from_csv("results/my_run.csv")
```

---

## 🖼️ 6. Visualizing Gantt Charts

If `track_schedule=True` is enabled, you can visualize final schedules:

```python
from metaforge.utils.visualization import plot_gantt_chart

plot_gantt_chart(
    schedule=your_schedule,
    num_machines=problem.num_machines,
    num_jobs=len(problem.jobs),
    title="Best Schedule"
)
```

To compare multiple solvers:

```python
plot_multiple_gantt(
    schedules_dict={
        "TS": schedule_ts,
        "GA": schedule_ga
    },
    num_machines=problem.num_machines,
    num_jobs=len(problem.jobs)
)
```

---

## 🧠 7. Supported Solvers

See [`docs/solvers.md`](./solvers.md) for the full list of solver IDs, classes, and descriptions.

---

## 💡 Tips

- Use `track_history=True` to record convergence for plotting.
- Use `track_schedule=True` to enable Gantt chart support.
- Benchmarks can be `.txt` (OR-Library) or `.json` (custom format).
- You can add your own benchmarks in `data/benchmarks/`.

---

## 🛠️ Next Steps

- [ ] Add your own benchmark formats (see `docs/datasets.md`)
- [ ] Create new solvers or tweak existing ones (see `docs/development.md`)
- [ ] Explore the [MetaForge roadmap](./roadmap.md)

---

Happy scheduling! 🧠🛠️
