# 🧠 MetaForge Solvers

MetaForge includes a wide variety of solvers for the Job Shop Scheduling Problem (JSSP), ranging from classic metaheuristics to reinforcement learning-based methods. Each solver is modular and extendable.

---

## ✅ Metaheuristic Solvers

### 🔧 Simulated Annealing
- **ID**: `sa`
- **Class**: `SimulatedAnnealingSolver`
- **Description**: A probabilistic local search that explores the solution space by accepting worse solutions based on a cooling schedule. Helps escape local optima.

---

### 🔧 Tabu Search
- **ID**: `ts`
- **Class**: `TabuSearchSolver`
- **Description**: Uses a memory structure (tabu list) to avoid cycles and encourages exploring new areas of the solution space.

---

### 🔧 Genetic Algorithm
- **ID**: `ga`
- **Class**: `GeneticAlgorithmSolver`
- **Description**: Population-based search that evolves schedules using crossover and mutation. Good for diverse exploration.

---

### 🔧 Ant Colony Optimization
- **ID**: `aco`
- **Class**: `AntColonySolver`
- **Description**: Probabilistic graph-based approach inspired by the pheromone trails of ants. Learns preferences over iterations.

---

## 🤖 Reinforcement Learning Solvers

### 🤖 Q-Learning
- **ID**: `q`
- **Class**: `QLearningSolver`
- **Description**: Tabular Q-learning with state-action pairs representing job pointers. Good for small problems; fast to test.

---

### 🤖 DQN (naive)
- **ID**: `dqn-naive`
- **Class**: `DQNAgentSolver`
- **Description**: Basic Deep Q-Network using a feedforward model. No replay buffer or target network. Useful as a simple baseline.

---

### 🤖 DQN (replay)
- **ID**: `dqn-replay`
- **Class**: `DQNAgentSolverReplay`
- **Description**: Advanced DQN with:
  - Experience replay
  - Target network
  - Reward shaping
  - Epsilon decay
  This is the **recommended RL baseline**.

---

### 🧬 Neuroevolution
- **ID**: `neuroevo`
- **Class**: `NeuroevolutionSolver`
- **Description**: Evolves neural network weights for job selection using evolutionary strategies. No gradients needed. Black-box and adaptive.

---

## 🛠️ Notes

- All solvers return a consistent output structure with:
  - `makespan` (best score)
  - `solution` (job sequence)
  - `history` (makespan over iterations)
  - `schedules` (optional: full schedule for Gantt)

- Use `track_history=True` to enable convergence tracking.
- Use `track_schedule=True` for Gantt chart outputs.

---

