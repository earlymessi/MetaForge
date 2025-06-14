# 🐜 Ant Colony Optimization (ACO)

Ant Colony Optimization (ACO) is a **bio-inspired metaheuristic** based on how ants find the shortest paths using pheromone trails. In scheduling, artificial "ants" construct solutions and lay virtual pheromones on good paths, which guide future solution construction.

---

## 🔍 How It Works

1. Each ant constructs a complete solution based on probabilities and pheromone strengths
2. Solutions are evaluated using a fitness function (e.g., makespan)
3. Pheromones are updated:
   - Reinforce good solutions
   - Evaporate over time to prevent stagnation
4. Repeat over multiple iterations

---

## ⚙️ Key Parameters

| Parameter           | Description                                      |
|---------------------|--------------------------------------------------|
| `num_ants`          | Number of ants constructing solutions per iteration |
| `iterations`        | Number of construction cycles                    |
| `alpha`             | Influence of pheromone                          |
| `beta`              | Influence of heuristic (e.g., processing time)  |
| `evaporation_rate`  | Rate at which pheromones decay                  |

---

## 🚀 Example Run

```python
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.metaforge_runner import run_solver
from metaforge.utils.plotting import plot_solver_dashboard
from metaforge.utils.visualization import plot_gantt_chart

# Load benchmark problem
url = "https://raw.githubusercontent.com/Mageed-Ghaleb/MetaForge/main/data/benchmarks/ft06.txt"
problem = load_job_shop_instance(url)

# Run ACO
result = run_solver("aco", problem, track_schedule=True)

# Print best result
print("Best Makespan:", result["makespan"])

# 📈 Plot convergence
plot_solver_dashboard(result["history"], title="Ant Colony Optimization Progress", solver_name="ACO")

# 📊 Plot Gantt chart
schedule = result["schedules"][-1]
plot_gantt_chart(schedule, num_machines=problem.num_machines, num_jobs=len(problem.jobs))
```

---

## ✅ Strengths

- Strong for combinatorial optimization
- Balances exploration and exploitation
- Can adapt dynamically based on search history

---

## ⚠️ Limitations

- Sensitive to parameter tuning
- Can stagnate without diversity controls
- Slower than greedy heuristics

---

## 📈 When to Use

| Situation                          | ACO Good? |
|-----------------------------------|-----------|
| You want intelligent exploration  | ✅ Yes    |
| Need hybrid learning behavior     | ✅ Yes    |
| Need fast prototype               | ❌ No     |
| Very large problem instance       | ⚠️ Medium |

---

## 📚 Related Solvers

- Genetic Algorithm: population-based evolution
- Q-Learning: reinforcement learning instead of pheromones
- Simulated Annealing: greedy randomized search

---

> “Ants may be small, but together, they discover the most efficient paths.” 🐜
