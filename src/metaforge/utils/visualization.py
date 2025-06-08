import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import animation
import seaborn as sns
import pandas as pd

def plot_gantt_chart(schedule, num_machines, num_jobs, title="Job-Shop Schedule", figsize=(12, 5), save_as=None):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    fig, ax = plt.subplots(figsize=figsize)

    for op in schedule:
        machine = op["machine"]
        job = op["job"]
        op_idx = op["operation"]
        start = op["start"]
        end = op["end"]
        color = colors[job % len(colors)]

        ax.barh(machine, end - start, left=start, color=color, edgecolor='black')
        ax.text(start + (end - start) / 2, machine, f"J{job}-O{op_idx}", 
                va='center', ha='center', color='white', fontsize=8, fontweight='bold')

    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {i}" for i in range(num_machines)])
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True, axis='x')

    legend_handles = [Patch(color=colors[j % len(colors)], label=f"Job {j}") for j in range(num_jobs)]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=300)

    plt.show()

def plot_multiple_gantt(schedules_dict, num_machines, num_jobs, figsize=(16, 5)):
    """
    Plot final Gantt charts from multiple solvers side by side.

    Args:
        schedules_dict (dict): {solver_name: schedule}, where schedule is a list of ops with keys: job, operation, machine, start, end
        num_machines (int): Total number of machines
        num_jobs (int): Total number of jobs
        figsize (tuple): Size of the figure
    """
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    solver_names = list(schedules_dict.keys())
    n = len(solver_names)

    fig, axs = plt.subplots(1, n, figsize=figsize, sharey=True)

    if n == 1:
        axs = [axs]

    for idx, solver_name in enumerate(solver_names):
        schedule = schedules_dict[solver_name]
        ax = axs[idx]

        for op in schedule:
            machine = op["machine"]
            job = op["job"]
            op_idx = op["operation"]
            start = op["start"]
            end = op["end"]
            color = colors[job % len(colors)]

            ax.barh(machine, end - start, left=start, color=color, edgecolor='black')
            ax.text(start + (end - start) / 2, machine, f"J{job}-O{op_idx}",
                    va='center', ha='center', color='white', fontsize=7, fontweight='bold')

        ax.set_title(solver_name)
        ax.set_xlabel("Time")
        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f"M{i}" for i in range(num_machines)])
        ax.grid(True, axis='x')

    # Add job legend to last axis
    legend_handles = [Patch(color=colors[j % len(colors)], label=f"Job {j}") for j in range(num_jobs)]
    axs[-1].legend(handles=legend_handles, loc="upper right")

    fig.suptitle("Final Gantt Charts by Solver", fontsize=14)
    plt.tight_layout()
    plt.show()

def animate_gantt_evolution(schedule_frames, num_machines, num_jobs, interval=400, save_path=None):
    """
    Create a Gantt chart animation from a sequence of schedule frames.

    Args:
        schedule_frames (List[List[Dict]]): List of schedules (one per iteration)
        num_machines (int): Number of machines
        num_jobs (int): Number of jobs
        interval (int): Delay between frames in milliseconds
        save_path (str, optional): If provided, saves animation as .gif
    """
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    fig, ax = plt.subplots(figsize=(12, 5))

    def update(frame_index):
        ax.clear()
        schedule = schedule_frames[frame_index]

        for op in schedule:
            machine = op["machine"]
            job = op["job"]
            op_idx = op["operation"]
            start = op["start"]
            end = op["end"]
            color = colors[job % len(colors)]

            ax.barh(machine, end - start, left=start, color=color, edgecolor='black')
            ax.text(start + (end - start) / 2, machine, f"J{job}-O{op_idx}",
                    va='center', ha='center', color='white', fontsize=7)

        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f"M{i}" for i in range(num_machines)])
        ax.set_title(f"Gantt Evolution - Iteration {frame_index + 1}")
        ax.set_xlabel("Time")
        ax.grid(True, axis='x')

    anim = animation.FuncAnimation(fig, update, frames=len(schedule_frames), interval=interval, repeat=False)

    if save_path:
        anim.save(save_path, writer='pillow', fps=1000 // interval)
    else:
        plt.close(fig)  # prevent duplicate static output in Jupyter
        return anim

def plot_results_from_csv(csv_path, show=True, save_path=None):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 6))
    # sns.barplot(data=df, x="Solver", y="BestMakespan", hue="Benchmark")
    sns.barplot(data=df, x="solver", y="best_score", hue="benchmark")
    plt.title("Best Makespan per Solver across Benchmarks")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📊 Plot saved to {save_path}")
    if show:
        plt.show()


def plot_runtime_from_csv(csv_path, show=True, save_path=None):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 6))
    # sns.barplot(data=df, x="Solver", y="RuntimeSec", hue="Benchmark")
    sns.barplot(data=df, x="solver", y="runtime_sec", hue="benchmark")
    plt.title("Runtime (seconds) per Solver across Benchmarks")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📊 Runtime plot saved to {save_path}")
    if show:
        plt.show()