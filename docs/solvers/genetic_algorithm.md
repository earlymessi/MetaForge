# 🧬 Genetic Algorithm (GA)

Genetic Algorithm (GA) is a **population-based metaheuristic** inspired by the process of natural selection and evolution. It evolves a pool of candidate solutions over time using operators like **selection**, **crossover**, and **mutation**.

---

## 🔍 How It Works

1. Initialize a population of random solutions
2. Evaluate fitness (e.g., makespan)
3. Select parents based on fitness
4. Generate offspring via crossover (recombination)
5. Apply random mutations to maintain diversity
6. Repeat for multiple generations

---

## ⚙️ Key Parameters

| Parameter          | Description                                       |
|--------------------|---------------------------------------------------|
| `population_size`  | Number of solutions per generation                |
| `generations`      | Number of generations to evolve                   |
| `crossover_rate`   | Probability of combining parents                  |
| `mutation_rate`    | Probability of modifying a gene (job ID position) |

---

## 🚀 Example Run

```python
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.metaforge_runner import run_solver
from metaforge.utils.plotting import plot_solver_dashboard
from metaforge.utils.visualization import plot_gantt_chart

# Load benchmark
url = "https://raw.githubusercontent.com/Mageed-Ghaleb/MetaForge/main/data/benchmarks/ft06.txt"
problem = load_job_shop_instance(url)

# Run Genetic Algorithm
result = run_solver("ga", problem, track_schedule=True)

# Print best result
print("Best Makespan:", result["makespan"])

# 📈 Plot convergence
plot_solver_dashboard(result["history"], title="Genetic Algorithm Progress", solver_name="GA")

# 📊 Plot final Gantt chart
schedule = result["schedules"][-1]
plot_gantt_chart(schedule, num_machines=problem.num_machines, num_jobs=len(problem.jobs))
```

---

## ✅ Strengths

- Explores diverse regions of the solution space
- Adaptable to many problem types
- Parallelizable

---

## ⚠️ Limitations

- May converge prematurely (need tuning)
- Slower than greedy algorithms
- Crossover design is problem-sensitive

---

## 📈 When to Use

| Situation                          | GA Good? |
|-----------------------------------|----------|
| Exploring large solution spaces   | ✅ Yes   |
| Fast convergence needed           | ❌ No    |
| Real-world JSSP with noise        | ✅ Yes   |
| Simple job layouts                | ✅ Yes   |

---

## 📚 Related Solvers

- Simulated Annealing: greedy with randomness
- Tabu Search: deterministic with memory
- Neuroevolution: learning-based evolution

---

> “GA evolves your schedule generation process — one gene at a time.” 🧬
