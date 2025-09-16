import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from metaforge.core.base_solver import BaseSolver
from metaforge.solvers.dqn_solver import QNetwork, ReplayBuffer


class DQNAgentSolverReplayDynamic(BaseSolver):
    """
    一个支持动态事件（机器故障、新订单到达）的DQN求解器。
    这个版本经过重构，以确保状态管理的一致性，并能生成正确的动态调度方案用于可视化。
    """

    def __init__(self, problem, dynamic_events=None, episodes=300, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995, gamma=0.95, lr=1e-3,
                 buffer_capacity=10000, batch_size=64, target_update_freq=10):
        super().__init__(problem)
        if not hasattr(self.problem, 'initial_jobs'):
            self.problem.initial_jobs = [job for job in self.problem.jobs]
        self.original_dynamic_events = [e.copy() for e in dynamic_events] if dynamic_events else []
        self.episodes = episodes
        self.epsilon_init = epsilon
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    def _initialize_problem_parameters(self):
        self.job_counts = [len(job.tasks) for job in self.problem.jobs]
        self.num_jobs = len(self.job_counts)
        self.total_ops = sum(self.job_counts)
        self.input_size = self.num_jobs * 2 + self.problem.num_machines
        self.output_size = self.num_jobs

    def _initialize_networks(self):
        self.qnet = QNetwork(self.input_size, self.output_size).to(self.device)
        self.target_qnet = QNetwork(self.input_size, self.output_size).to(self.device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def _expand_networks(self, old_input_size, old_output_size):
        print(f"🧠 正在扩展网络：输入 {old_input_size} -> {self.input_size}, 输出 {old_output_size} -> {self.output_size}")
        new_qnet = QNetwork(self.input_size, self.output_size).to(self.device)
        new_target_qnet = QNetwork(self.input_size, self.output_size).to(self.device)
        old_state_dict = self.qnet.state_dict()
        new_state_dict = new_qnet.state_dict()
        layer_keys = [key for key in old_state_dict.keys() if 'weight' in key]
        if not layer_keys: raise RuntimeError("QNetwork中没有找到任何带权重的层。")
        first_layer_w_key, last_layer_w_key = layer_keys[0], layer_keys[-1]
        first_layer_b_key, last_layer_b_key = first_layer_w_key.replace('weight', 'bias'), last_layer_w_key.replace('weight', 'bias')
        print(f"   自动检测到第一层: '{first_layer_w_key}'"); print(f"   自动检测到最后一层:  '{last_layer_w_key}'")
        for key in old_state_dict:
            old_job_dims = old_output_size * 2
            if key == first_layer_w_key:
                new_state_dict[key][:, :old_job_dims] = old_state_dict[key][:, :old_job_dims]
                new_state_dict[key][:, -self.problem.num_machines:] = old_state_dict[key][:, -self.problem.num_machines:]
            elif key == last_layer_w_key: new_state_dict[key][:old_output_size, :] = old_state_dict[key]
            elif key == last_layer_b_key: new_state_dict[key][:old_output_size] = old_state_dict[key]
            elif key in new_state_dict: new_state_dict[key] = old_state_dict[key]
        new_qnet.load_state_dict(new_state_dict); new_target_qnet.load_state_dict(new_state_dict)
        self.qnet, self.target_qnet = new_qnet, new_target_qnet
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def _build_state_tensor(self, job_ptrs, job_ready, machine_status):
        padded_ptrs = job_ptrs + [0] * (self.num_jobs - len(job_ptrs))
        padded_ready = job_ready + [0] * (self.num_jobs - len(job_ready))
        return torch.tensor(padded_ptrs + padded_ready + machine_status, dtype=torch.float32, device=self.device)

    def _get_available_jobs(self, job_ptrs, machine_status):
        available = []
        for j in range(len(job_ptrs)):
            if job_ptrs[j] < self.job_counts[j]:
                op_idx = job_ptrs[j]
                machine_req = self.problem.jobs[j].tasks[op_idx].machine_id
                if machine_status[machine_req] == 1: available.append(j)
        return available

    def _is_terminal(self, job_ptrs):
        if len(job_ptrs) != len(self.job_counts): return False
        return all(job_ptrs[j] >= self.job_counts[j] for j in range(len(job_ptrs)))

    def _generate_dynamic_schedule(self, sequence, dynamic_events):
        schedule, breakdowns = [], [e for e in dynamic_events if e['type'] == 'machine_breakdown_on_nth_task']
        time_events = sorted([e.copy() for e in dynamic_events if e.get('time') is not None], key=lambda x: x['time'])
        machine_ready_time = [0] * self.problem.num_machines
        job_ready_time = [0] * self.num_jobs
        job_op_counters = [0] * self.num_jobs
        machine_task_counters = [0] * self.problem.num_machines
        new_job_idx = len(self.problem.initial_jobs)
        for event in time_events:
            if event['type'] == 'new_job_arrival' and new_job_idx < self.num_jobs:
                job_ready_time[new_job_idx] = event['time']
                new_job_idx += 1
        for job_id in sequence:
            if job_id >= len(self.problem.jobs): continue
            op_idx = job_op_counters[job_id]
            if op_idx >= len(self.problem.jobs[job_id].tasks): continue
            task = self.problem.jobs[job_id].tasks[op_idx]
            machine_id, duration = task.machine_id, task.duration
            start_time = max(machine_ready_time[machine_id], job_ready_time[job_id])
            end_time = start_time + duration
            machine_ready_time[machine_id], job_ready_time[job_id] = end_time, end_time
            job_op_counters[job_id] += 1
            machine_task_counters[machine_id] += 1
            schedule.append({"job": job_id, "operation": op_idx, "machine": machine_id, "start": start_time, "end": end_time})
            current_task_count = machine_task_counters[machine_id]
            for event in breakdowns:
                trigger = event['trigger']
                if trigger['machine'] == machine_id and trigger['task_count'] == current_task_count:
                    machine_ready_time[machine_id] += event['duration']
                    break
        return schedule

    def _handle_time_based_events(self, job_ptrs, job_ready):
        state_space_changed = False
        for event in list(self.dynamic_events):
            if event['type'] == 'new_job_arrival' and event.get('time') is not None and event['time'] <= self.current_time:
                print(f"⏰ Time-based event triggered at t={self.current_time:.2f}: {event['type']}")
                state_space_changed, old_input_size, old_output_size = True, self.input_size, self.output_size
                self.problem.jobs.append(event['job'])
                self._initialize_problem_parameters()
                self._expand_networks(old_input_size, old_output_size)
                self.buffer.clear()
                job_ptrs.append(0); job_ready.append(event['time'])
                new_job_id = self.num_jobs - 1
                print(f"✨ New job (ID: {new_job_id}) arrived. Total jobs now: {self.num_jobs}")
                self.dynamic_events.remove(event)
        return state_space_changed

    def _handle_machine_usage_events(self, machine_id, current_end_time, machine_ready, machine_task_counters):
        machine_task_counters[machine_id] += 1
        current_task_count = machine_task_counters[machine_id]
        for event in list(self.dynamic_events):
            if event['type'] == 'machine_breakdown_on_nth_task':
                trigger = event['trigger']
                if trigger['machine'] == machine_id and trigger['task_count'] == current_task_count:
                    print(f"💥 Machine usage event: M{machine_id} broke down after its {current_task_count}th task at t={current_end_time:.2f}!")
                    duration = event['duration']
                    machine_ready[machine_id] = max(machine_ready[machine_id], current_end_time + duration)
                    print(f"   Machine {machine_id} will now be ready at t={machine_ready[machine_id]:.2f}")
                    self.dynamic_events.remove(event)

    def train_step(self):
        if len(self.buffer) < self.batch_size: return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        padded_states = [torch.cat((s, torch.zeros(self.input_size - len(s)))).to(self.device) for s in states]
        padded_next_states = [torch.cat((ns, torch.zeros(self.input_size - len(ns)))).to(self.device) for ns in next_states]
        states, next_states = torch.stack(padded_states), torch.stack(padded_next_states)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        q_selected = self.qnet(states).gather(1, actions).squeeze(1)
        with torch.no_grad(): max_target_q = self.target_qnet(next_states).max(dim=1)[0]
        targets = rewards + self.gamma * max_target_q * (1 - dones)
        loss = self.loss_fn(q_selected, targets)
        self.optimizer.zero_grad(), loss.backward(), self.optimizer.step()

    def run(self, track_history=True, track_schedule=False):
        best_solution, best_score, history, all_schedules = None, float("inf"), [], []
        for ep in range(self.episodes):
            print(f"DQNAgentSolver (dynamic) - Episode {ep + 1}/{self.episodes}")
            self.problem.jobs = [job for job in self.problem.initial_jobs]
            self._initialize_problem_parameters(); self._initialize_networks()
            self.dynamic_events = [e.copy() for e in self.original_dynamic_events]
            self.epsilon = self.epsilon_init * (self.epsilon_decay ** ep)
            self.buffer.clear()
            self.job_ptrs, job_ready = [0] * self.num_jobs, [0] * self.num_jobs
            machine_ready = [0] * self.problem.num_machines
            machine_status = [1] * self.problem.num_machines
            machine_task_counters = [0] * self.problem.num_machines
            self.current_time, sequence = 0, []
            state = self._build_state_tensor(self.job_ptrs, job_ready, machine_status)
            while not self._is_terminal(self.job_ptrs):
                if self._handle_time_based_events(self.job_ptrs, job_ready):
                    state = self._build_state_tensor(self.job_ptrs, job_ready, machine_status)
                for m in range(self.problem.num_machines): machine_status[m] = 1 if machine_ready[m] <= self.current_time else 0
                available = self._get_available_jobs(self.job_ptrs, machine_status)
                if not available:
                    ready_times = [t for t in machine_ready if t > self.current_time]
                    if not ready_times: break
                    self.current_time = min(ready_times)
                    state = self._build_state_tensor(self.job_ptrs, job_ready, machine_status)
                    continue
                if random.random() < self.epsilon:
                    action = random.choice(available)
                else:
                    with torch.no_grad(): q_vals = self.qnet(state.unsqueeze(0)).cpu().numpy()[0]
                    masked_q = np.full(self.output_size, -np.inf)
                    for j in available:
                        if j < len(q_vals): masked_q[j] = q_vals[j]
                    action = int(np.argmax(masked_q))
                op_idx = self.job_ptrs[action]
                task = self.problem.jobs[action].tasks[op_idx]
                machine, proc_time = task.machine_id, task.duration
                start_time = max(machine_ready[machine], job_ready[action])
                self.current_time = max(self.current_time, start_time)
                end_time = self.current_time + proc_time
                machine_ready[machine], job_ready[action] = end_time, end_time
                sequence.append(action)
                self._handle_machine_usage_events(machine, end_time, machine_ready, machine_task_counters)
                self.job_ptrs[action] += 1
                next_state = self._build_state_tensor(self.job_ptrs, job_ready, machine_status)
                reward, done = -proc_time, self._is_terminal(self.job_ptrs)
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                self.train_step()
            if ep % self.target_update_freq == 0: self.target_qnet.load_state_dict(self.qnet.state_dict())
            score = max(machine_ready) if machine_ready else 0
            if score < best_score: best_score, best_solution = score, sequence[:]
            if track_history: history.append(best_score)
            if track_schedule and best_solution:
                valid_schedule = self._generate_dynamic_schedule(best_solution, self.original_dynamic_events)
                all_schedules.append(valid_schedule)
        return best_solution, best_score, history, all_schedules