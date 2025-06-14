# 🧊 Simulated Annealing (SA)

Simulated Annealing (SA) is a **probabilistic metaheuristic** inspired by the process of annealing in metallurgy. It searches for global optima by allowing occasional uphill moves to escape local minima, controlled by a **cooling schedule**.

---

## 🔍 How It Works

1. Start with a random solution
2. Generate a neighboring solution
3. Accept it if it's better
4. If worse, accept it with a probability that decreases over time
5. Repeat while gradually reducing the "temperature"

---

## ⚙️ Key Parameters

| Parameter       | Description                            |
|----------------|----------------------------------------|
| `initial_temp`  | Starting temperature (e.g., 100)       |
| `cooling_rate`  | Multiplicative decay (e.g., 0.95)      |
| `iterations`    | Total number of steps (e.g., 500)      |

---

## 🚀 Example Run

```python
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.metaforge_runner import run_solver
from metaforge.utils.visualization import plot_gantt_chart
from metaforge.utils.plotting import plot_solver_dashboard

url = "https://raw.githubusercontent.com/Mageed-Ghaleb/MetaForge/main/data/benchmarks/ft06.txt"
problem = load_job_shop_instance(url)

result = run_solver("sa", problem, track_schedule=True)

# Makespan
print("Best Makespan:", result["makespan"])

# 📊 Plot convergence + optional temperature curve
plot_solver_dashboard(result["history"], title="Simulated Annealing Progress", solver_name="SA")

# 🏁 Gantt chart
schedule = result["schedules"][-1]
plot_gantt_chart(schedule, num_machines=problem.num_machines, num_jobs=len(problem.jobs))
```

---

## ✅ Strengths

- Simple and effective for small to mid-size problems
- Can escape local optima early in search
- Easy to tune and adapt

---

## ⚠️ Limitations

- Performance sensitive to cooling schedule
- Slow convergence at high temperatures
- May require many iterations for complex problems

---

## 📈 When to Use

| Situation                     | SA Good? |
|------------------------------|----------|
| You want fast prototyping    | ✅ Yes   |
| You're escaping local traps  | ✅ Yes   |
| You need large-scale control | ❌ No    |
| You want deep learning combo | ❌ No    |

---

## 📚 Related Solvers

- Tabu Search: deterministic memory-based search
- Genetic Algorithm: population-based exploration
- DQN: neural network + RL-based scheduling

---

> “Sometimes, you need a little heat to find the best solution.” 🔥

