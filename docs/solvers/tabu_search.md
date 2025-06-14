# 🚫 Tabu Search (TS)

Tabu Search is a **deterministic local search metaheuristic** that explores the neighborhood of solutions while avoiding cycles by using a memory structure called the **tabu list**.

It is especially effective for combinatorial optimization problems like Job Shop Scheduling.

---

## 🔍 How It Works

1. Start with an initial solution
2. Generate a set of neighbors (via small perturbations)
3. Select the best non-tabu neighbor (or one that satisfies aspiration criteria)
4. Update the tabu list to prevent revisiting recent moves
5. Repeat for a fixed number of iterations or until convergence

---

## ⚙️ Key Parameters

| Parameter        | Description                              |
|------------------|------------------------------------------|
| `iterations`     | Total number of iterations               |
| `tabu_tenure`    | Number of iterations a move is considered tabu |
| `neighborhood_size` | How many neighbors to evaluate at each step |

---

## 🚀 Example Run

```python
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.metaforge_runner import run_solver
from metaforge.utils.plotting import plot_solver_dashboard
from metaforge.utils.visualization import plot_gantt_chart

# Load the benchmark problem
url = "https://raw.githubusercontent.com/Mageed-Ghaleb/MetaForge/main/data/benchmarks/ft06.txt"
problem = load_job_shop_instance(url)

# Run Tabu Search
result = run_solver("ts", problem, track_schedule=True)

# Makespan
print("Best Makespan:", result["makespan"])

# 📈 Plot convergence curve
plot_solver_dashboard(result["history"], title="Tabu Search Progress", solver_name="TS")

# 📊 Plot final Gantt chart
schedule = result["schedules"][-1]
plot_gantt_chart(schedule, num_machines=problem.num_machines, num_jobs=len(problem.jobs))
```

---

## ✅ Strengths

- Strong local search performance
- Efficient at escaping local optima
- Very effective for small to medium JSSP instances

---

## ⚠️ Limitations

- Sensitive to tabu tenure setting
- Risk of cycling if tenure is too short
- Requires designing a good neighborhood function

---

## 📈 When to Use

| Situation                           | TS Good? |
|------------------------------------|----------|
| Small to mid-size JSSP             | ✅ Yes   |
| You need reproducibility           | ✅ Yes   |
| You want full exploration          | ❌ No    |
| You're using continuous variables  | ❌ No    |

---

## 📚 Related Solvers

- Simulated Annealing: randomized local search
- Genetic Algorithm: population-based
- Q-Learning: reinforcement-driven

---

> “Tabu Search doesn't forget — it learns from the past to avoid repetition.” 🧠
