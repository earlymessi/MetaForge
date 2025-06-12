# 🔧 MetaForge

**MetaForge** is a modular optimization framework for solving **Job Shop Scheduling Problems (JSSP)** using advanced **metaheuristics** and **learning-based approaches**.

🚀 From classic solvers like Tabu Search and Genetic Algorithms to modern DQN-based reinforcement learning and Neuroevolution — MetaForge brings it all together with clarity, structure, and fun.

---

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mageed-Ghaleb/MetaForge/blob/main/notebooks/MetaForge_Colab_Demo.ipynb)

---

## 🎯 Key Features

- ✅ Support for classic OR-Library and custom JSON benchmark formats
- 🧠 Modular solvers:
  - Tabu Search
  - Simulated Annealing
  - Genetic Algorithm
  - Ant Colony Optimization
  - Q-Learning
  - DQN (naive & replay-based)
  - Neuroevolution
- 📊 Beautiful convergence plots, runtime comparisons, and Gantt chart visualizations
- 📦 Easy packaging, CLI usage, and extension with new solvers
- 🔬 Designed for researchers, students, and practitioners alike

---

## 🚀 Quick Start

### 1. Install MetaForge

```bash
pip install metaforge
```

Or clone locally for development:

```bash
git clone https://github.com/Mageed-Ghaleb/MetaForge.git
cd MetaForge
pip install -e .
```

---

### 2.Run a Solver (Local or Remote Benchmark)

```python
import os
import requests
from metaforge.problems.benchmark_loader import load_job_shop_instance
from metaforge.metaforge_runner import run_solver

# You can pass a local path or a URL to load the problem!
problem = load_job_shop_instance("https://raw.githubusercontent.com/Mageed-Ghaleb/MetaForge/main/data/benchmarks/ft06.txt")

# Run a solver (sa, ts, ga, aco, pso, ...)
result = run_solver("ga", problem, track_schedule=True)

print("Best Makespan:", result["makespan"])
```

---

### 3. Run All Solvers on All Benchmarks

```bash
python -m src.metaforge.utils.compare_solvers
```

Generates CSV, plots, and Gantt charts from the `data/benchmarks/` folder.

---

### 4. Interactive Notebook

Explore MetaForge hands-on with our example notebook:

📓 [`MetaForge_Example.ipynb`](https://github.com/Mageed-Ghaleb/MetaForge/blob/main/notebooks/MetaForge_Example.ipynb)

It walks you through:

- Loading benchmark problems
- Running various solvers (TS, GA, DQN, etc.)
- Plotting convergence + Gantt charts
- Comparing performance across solvers

> ✅ Coming soon: One-click Colab version for browser-based use — no setup needed.

---

## 📚 Documentation

> 📘 Full documentation site coming soon! In the meantime, browse core modules below:

- 📖 [Usage Guide (source)](https://github.com/Mageed-Ghaleb/MetaForge/blob/main/docs/usage.md)
- 🧠 [Solvers Overview (source)](https://github.com/Mageed-Ghaleb/MetaForge/blob/main/docs/solvers.md)
- 📂 [Benchmark Datasets (source)](https://github.com/Mageed-Ghaleb/MetaForge/blob/main/docs/datasets.md)

---

## 🧠 Why MetaForge?

Most libraries focus only on one type of solver. MetaForge unifies traditional, bio-inspired, and learning-based approaches in one clean, extensible Python package — built for experimentation, benchmarking, and educational use.

Whether you're doing a thesis, publishing research, or solving real-world factory problems — **MetaForge is your launchpad**. 🚀

---

## 📈 Contributing

We're just getting started! Feel free to:

- Suggest solvers, features, or dataset formats
- Fork and extend
- Submit pull requests with improvements

---

## 📄 License

MIT License — free for academic and commercial use.

---

## 👨‍💻 Author

**Mageed Ghaleb**  
📧 mageed.ghaleb@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/mageed-ghaleb/)  
🔗 [GitHub](https://github.com/Mageed-Ghaleb)

---

> Built with ❤️ for solvers, schedules, and scientific curiosity.
