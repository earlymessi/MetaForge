# 🧬 Neuroevolution for Scheduling

**Neuroevolution** is a class of algorithms that optimize **neural network parameters (weights)** using **evolutionary strategies** instead of backpropagation. In MetaForge, Neuroevolution evolves deep policies that learn to schedule job operations without gradient descent or value functions.

---

## 🔍 How It Works

1. **Initialize a population** of neural networks with random weights
2. For each generation:
   - Evaluate each policy by running it as a schedule agent
   - Rank policies by fitness (e.g., inverse of makespan)
   - Select top performers (elitism)
   - **Generate offspring** via mutation and crossover
3. Repeat until convergence or max generations

---

## ⚙️ Key Parameters

| Parameter         | Description                                      |
|-------------------|--------------------------------------------------|
| `population_size` | Number of neural networks per generation         |
| `generations`     | How many generations to evolve                   |
| `mutation_rate`   | How often weights are perturbed                  |
| `hidden_dim`      | Size of hidden layer in the evolved networks     |

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

# Run Neuroevolution
result = run_solver("neuroevo", problem, track_schedule=True)

# Output best makespan
print("Best Makespan:", result["makespan"])

# Plot convergence
plot_solver_dashboard(result["history"], title="Neuroevolution Convergence", solver_name="Neuroevolution")

# Plot Gantt chart
schedule = result["schedules"][-1]
plot_gantt_chart(schedule, num_machines=problem.num_machines, num_jobs=len(problem.jobs))
```

---

## ✅ Strengths

- No gradients or backprop needed
- Works well for black-box optimization
- Can evolve robust, interpretable neural schedulers

---

## ⚠️ Limitations

- Slower convergence compared to gradient-based methods
- Sensitive to mutation/crossover design
- Needs proper scaling of rewards/fitness

---

## 📈 When to Use

| Scenario                           | Neuroevo Good? |
|------------------------------------|----------------|
| You want interpretable policies    | ✅ Yes         |
| Black-box or noisy scheduling env  | ✅ Yes         |
| Need fast convergence              | ❌ No          |
| Have strict compute constraints    | ❌ No          |

---

## 🤖 MetaForge Integration

MetaForge implements Neuroevolution with:
- Feedforward neural networks (1–2 hidden layers)
- Gaussian mutation strategy for weight updates
- Fitness evaluation based on makespan or completion time

You can extend it with:
- Crossover operators
- Multi-objective optimization
- Policy constraints

---

## 📚 Related Solvers

- Genetic Algorithm: similar mutation/crossover without networks
- DQN: gradient-based policy learning
- SA/TS: classic non-learning heuristics

---

> “Neuroevolution rewires the brain of your scheduler — no gradients required.” 🧬🔥
