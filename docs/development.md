# 🛠️ MetaForge Development Guide

This guide walks you through how to contribute to MetaForge by:

- Adding a new solver
- Integrating it into the comparison framework
- Ensuring it supports logging, plotting, and Gantt visualization

---

## 📁 Project Structure Overview

```
MetaForge/
├── src/
│   └── metaforge/
│       ├── solvers/              ← Your solvers go here
│       ├── core/                 ← Base solver classes
│       ├── problems/             ← Problem definition and loaders
│       ├── utils/                ← Benchmark runners, plotting, helpers
├── data/benchmarks/             ← OR-Library problem instances (.txt)
├── results/                     ← Output CSVs, plots, Gantt images
├── tests/                       ← Sanity test scripts
```

---

## 🧱 1. Create a New Solver

1. Create a new file under `src/metaforge/solvers/`  
   Example: `my_new_solver.py`

2. Inherit from `BaseSolver`:

```python
from metaforge.core.base_solver import BaseSolver

class MyNewSolver(BaseSolver):
    def __init__(self, problem, **kwargs):
        super().__init__(problem)
        # add solver-specific parameters here

    def run(self, track_history=True, track_schedule=False):
        ...
        return best_solution, best_score, history, all_schedules
```

3. Return values (required):
```python
return best_solution, best_score, history, all_schedules
```

---

## 🧠 2. Register the Solver

In `src/metaforge/metaforge_runner.py`, register your solver like this:

```python
from metaforge.solvers.my_new_solver import MyNewSolver

SOLVER_REGISTRY = {
    ...
    "my-solver": MyNewSolver,
}
```

Now it can be used in:
```bash
python -m src.metaforge.utils.compare_solvers
```

---

## 📈 3. Support for Plotting and Gantt

To enable convergence plots:
- Append makespan per iteration to `history` list when `track_history=True`

To enable Gantt charts:
- At the end, call:
```python
self.problem.get_schedule(best_solution[:])
```
- Append it to `all_schedules` when `track_schedule=True`

---

## ✅ 4. Test Your Solver

Update `tests/test_compare_solvers.py` to include your solver ID.

Run:
```bash
python tests/test_compare_solvers.py
```

Or test across all benchmarks:
```bash
python -m src.metaforge.utils.compare_solvers
```

---

## 🧪 5. Optional: Add Solver to Docs

Once your solver is tested and integrated, document it in [`docs/solvers.md`](./solvers.md):

```markdown
### 🚀 My New Solver
- **ID**: `my-solver`
- **Class**: `MyNewSolver`
- **Description**: Short explanation of how the solver works.
```

---

## 🧰 Tips

- Use `print(...)` sparingly inside `run()` — results will be logged automatically.
- Keep state clean: avoid writing to globals.
- Store all random sampling via `random` or `numpy.random` for reproducibility.
- For RL solvers, use the `torch.device(...)` pattern to support CPU/GPU.

---

## 🙌 Thanks for Contributing

MetaForge is designed to be modular, educational, and extendable.  
By contributing a new solver, you’re helping push the frontier of open optimization forward 🚀
