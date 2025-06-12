# 🔧 MetaForge

**MetaForge** is a modular Python toolkit for solving **Job Shop Scheduling Problems (JSSP)** using classic **metaheuristics** and modern **reinforcement learning** methods.

🚀 From Tabu Search and Genetic Algorithms to Deep Q-Networks (DQN) and Neuroevolution — MetaForge brings together the best of optimization and AI in one clean, extensible framework.

---

## 🎯 Key Features

- ✅ Solve classic benchmark problems (OR-Library, JSON)
- 🧠 Built-in solvers:
  - Tabu Search (TS)
  - Simulated Annealing (SA)
  - Genetic Algorithm (GA)
  - Ant Colony Optimization (ACO)
  - Q-Learning
  - DQN (with and without replay buffer)
  - Neuroevolution
- 📊 Beautiful convergence and Gantt chart visualizations
- 🤖 Reinforcement Learning support out-of-the-box
- 🧪 Designed for research, education, and real-world production scheduling

---

## 🚀 Quick Start

### 1. Install MetaForge

```bash
pip install metaforge
```

Or for local development:

```bash
git clone https://github.com/Mageed-Ghaleb/MetaForge.git
cd MetaForge
pip install -e .
```

---

### 2. Run a Solver (Example)

```python
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.metaforge_runner import run_solver

problem = load_job_shop_instance("https://raw.githubusercontent.com/Mageed-Ghaleb/MetaForge/main/data/benchmarks/ft06.txt")
result = run_solver("ts", problem, track_schedule=True)

print("Best Makespan:", result["makespan"])
```

---

### 3. Visualize the Final Schedule

```python
from metaforge.utils.visualization import plot_gantt_chart

schedule = result["schedules"][-1]
plot_gantt_chart(schedule, num_machines=problem.num_machines, num_jobs=len(problem.jobs))
```

---

### 4. Interactive Colab Notebooks 🚀

#### 📝 1. Hands-on Demo Notebook  
Explore MetaForge interactively with a guided walk-through:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mageed-Ghaleb/MetaForge/blob/main/notebooks/MetaForge_Colab_Demo.ipynb)

Covers:
- Loading benchmark problems
- Running various solvers (TS, GA, DQN, etc.)
- Plotting convergence + Gantt charts

---

#### 📊 2. Compare Solvers Notebook  
Run all solvers on all benchmark files and generate visual comparisons:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mageed-Ghaleb/MetaForge/blob/main/notebooks/MetaForge_Compare_Solvers.ipynb)

Covers:
- Running `ts`, `ga`, `sa`, `aco`, etc. across all problems
- Convergence plots per benchmark
- Solver performance summary plots

---

## 📚 Documentation

- 📖 [Usage Guide](docs/usage.md)
- 🧠 [Solvers Overview](docs/solvers.md)
- 📂 [Benchmark Format](docs/datasets.md)

---

## 🧠 Why MetaForge?

Most libraries focus on one type of solver. MetaForge unifies traditional algorithms and deep reinforcement learning into one clean package. Whether you’re teaching, publishing, or scheduling in a factory — MetaForge is your launchpad. 🚀

---

## 📈 Contributing

We're just getting started! Feel free to:

- Suggest solvers or enhancements
- Fork and extend
- Submit PRs — code, docs, notebooks, anything

---

## 📄 License

MIT License — free for academic and commercial use.

---

## 👨‍💻 Author

**Mageed Ghaleb**  
📧 mageed.ghaleb@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/mageed-ghaleb/)  
🔗 [GitHub](https://github.com/mageed-ghaleb)

---

> Built with ❤️ for solvers, schedules, and scientific curiosity.


---

## 🔎 Keywords (for discoverability)

MetaForge is designed for:

- Job Shop Scheduling Problems (JSSP)
- Metaheuristics (Tabu Search, Genetic Algorithm, ACO, SA)
- Reinforcement Learning in Scheduling (Q-Learning, DQN)
- Production Scheduling Optimization
- Flexible Flowshops & Real-world Scheduling
- Benchmark Comparisons and Solver Visualization
