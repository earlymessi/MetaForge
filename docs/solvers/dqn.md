# 🧠 Deep Q-Network (DQN)

**Deep Q-Networks (DQN)** combine **reinforcement learning** with **deep neural networks**, enabling agents to learn effective scheduling policies in **large or continuous state spaces** — where traditional Q-tables break down.

In MetaForge, DQN learns to select the next job operation to schedule based on a **neural network approximation** of the Q-function.

---

## 🔍 How It Works

1. States are represented as vectors (e.g., machine/job readiness, remaining ops)
2. A neural network `Q(s, a; θ)` approximates the Q-value for each possible action
3. Actions are selected via ε-greedy exploration
4. The network is trained using the **Bellman equation**:

   ```text
   Q(s, a) ← r + γ × max Q(s', a'; θ)
   ```

5. Training is done by minimizing the difference between predicted and target Q-values

### 🌀 Replay Buffer Variant
In the **DQN-Replay** variant, a memory buffer stores past transitions and trains the network on **random mini-batches**, improving stability and convergence.

---

## ⚙️ Key Parameters

| Parameter          | Description                                   |
|--------------------|-----------------------------------------------|
| `gamma`            | Discount factor (e.g., 0.95)                  |
| `epsilon`          | Exploration rate (starts high, decays)       |
| `lr`               | Learning rate for the optimizer               |
| `hidden_dim`       | Size of hidden layers                         |
| `episodes`         | Total training episodes                       |
| `replay_buffer`    | Enable experience replay (True/False)         |
| `batch_size`       | Number of samples per training batch (if replay) |

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

# Run DQN (Replay or Naive)
result = run_solver("dqn-replay", problem, track_schedule=True)

# Show best result
print("Best Makespan:", result["makespan"])

# Plot learning curve
plot_solver_dashboard(result["history"], title="DQN (Replay) Convergence", solver_name="DQN-Replay")

# Plot final schedule
schedule = result["schedules"][-1]
plot_gantt_chart(schedule, num_machines=problem.num_machines, num_jobs=len(problem.jobs))
```

---

## ✅ Strengths

- Handles high-dimensional and continuous state spaces
- Learns generalizable scheduling policies
- Better scalability than tabular Q-learning

---

## ⚠️ Limitations

- Requires tuning (learning rate, epsilon decay, etc.)
- Sensitive to architecture and reward design
- Needs more training time than heuristics

---

## 📈 When to Use

| Scenario                             | DQN Good? |
|-------------------------------------|-----------|
| You need scalability                 | ✅ Yes    |
| Want to reuse learned policies       | ✅ Yes    |
| Need interpretable scheduling logic | ❌ No     |
| Low-resource environments            | ❌ No     |

---

## 🤖 MetaForge Integration

In MetaForge, DQN is used as:

- **DQN (Naive)**: learns online, updates every step
- **DQN (Replay)**: stores transitions and samples batches
- Network architecture: fully connected feedforward NN with customizable hidden layers

You can plug in your own network structure or extend reward shaping strategies.

---

## 📚 Related Solvers

- Q-Learning: tabular approach
- Neuroevolution: learns network weights via evolution
- GA/SA: classic heuristics

---

> “DQN brings deep learning to scheduling — letting agents learn smarter schedules over time.” 🧠📈
