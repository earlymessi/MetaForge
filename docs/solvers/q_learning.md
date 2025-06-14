# 🧠 Q-Learning for Job Shop Scheduling

**Q-Learning** is a type of **model-free reinforcement learning** (RL) where an agent learns to make decisions by interacting with an environment. In MetaForge, the agent learns how to **select the next job operation** to schedule by maximizing cumulative rewards (e.g., minimizing makespan).

---

## 🔍 How It Works

Q-Learning learns a **Q-table** that estimates the value of taking an action in a given state.

1. Initialize a Q-table `Q[state][action]` with zeros
2. For each episode:
   - Observe current state (e.g., available machines, next jobs)
   - Choose an action (e.g., select job index) via ε-greedy policy
   - Apply the action and observe reward and next state
   - Update Q-value:

   ```text
   Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') - Q(s, a)]
   ```

3. Repeat for many episodes
4. Derive a schedule by following the learned policy

---

## ⚙️ Key Parameters

| Parameter     | Description                              |
|---------------|------------------------------------------|
| `alpha`       | Learning rate (e.g., 0.1)                |
| `gamma`       | Discount factor for future rewards       |
| `epsilon`     | Exploration probability (0.1–0.3 typical)|
| `episodes`    | Number of training episodes              |

---

## 🚀 Example Run

```python
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.metaforge_runner import run_solver
from metaforge.utils.plotting import plot_solver_dashboard
from metaforge.utils.visualization import plot_gantt_chart

# Load problem
url = "https://raw.githubusercontent.com/Mageed-Ghaleb/MetaForge/main/data/benchmarks/ft06.txt"
problem = load_job_shop_instance(url)

# Run Q-Learning
result = run_solver("q", problem, track_schedule=True)

# Output best makespan
print("Best Makespan:", result["makespan"])

# Plot learning curve
plot_solver_dashboard(result["history"], title="Q-Learning Convergence", solver_name="Q-Learning")

# Plot Gantt chart of final schedule
schedule = result["schedules"][-1]
plot_gantt_chart(schedule, num_machines=problem.num_machines, num_jobs=len(problem.jobs))
```

---

## ✅ Strengths

- Can learn general scheduling policies
- Adapts to reward signals without needing explicit models
- Fast inference after training

---

## ⚠️ Limitations

- Needs discretized state/action spaces
- Q-table grows exponentially with problem size
- Limited generalization to unseen problems

---

## 📈 When to Use

| Situation                            | Q-Learning Good? |
|-------------------------------------|------------------|
| Educational demos / learning curves | ✅ Yes           |
| Small to medium JSSPs               | ✅ Yes           |
| Real-time policy decisions          | ⚠️ Medium        |
| Large-scale scheduling              | ❌ No            |

---

## 💡 What Makes MetaForge Special

In MetaForge, Q-Learning:
- Treats job selections as actions
- Learns via simulated episodes
- Schedules jobs to machines in a way that **minimizes final makespan**

You can also inspect the learned `Q-table` or derive your own policy logic from it.

---

## 📚 Related Solvers

- DQN: Q-learning with deep neural networks
- SA, GA: non-learning heuristics
- Neuroevolution: learning policy parameters via evolution

---

> “Q-Learning is where intelligent scheduling begins — one update at a time.” 📚
