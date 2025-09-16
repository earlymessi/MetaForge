import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import animation
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

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


def plot_dynamic_multiple_gantt(schedules_dict, num_machines, num_jobs, dynamic_events=None, new_job_start_index=None,
                                figsize=(20, 5)):
    """
    绘制具有动态事件的甘特图，严格遵循原始风格。
    此版本已升级，可显示'machine_breakdown_on_nth_task'事件。
    """
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    solver_names = list(schedules_dict.keys())
    n = len(solver_names)
    fig, axs = plt.subplots(1, n, figsize=figsize, sharey=True, squeeze=False)
    axs = axs.flatten()

    for idx, solver_name in enumerate(solver_names):
        schedule = schedules_dict.get(solver_name)
        ax = axs[idx]

        if not schedule:
            ax.text(0.5, 0.5, 'No Schedule Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(solver_name);
            ax.set_xlabel("Time")
            continue

        for op in schedule:
            machine, job, op_idx, start, end = op["machine"], op["job"], op["operation"], op["start"], op["end"]
            color = colors[job % len(colors)]
            if new_job_start_index is not None and job >= new_job_start_index:
                # 新插入订单：鲜明红色 + 斜条纹
                ax.barh(machine, end - start, left=start, color=color, edgecolor='black', hatch='///')
                ax.text(start + (end - start) / 2, machine, f"J{job}-O{op_idx}",
                    va='center', ha='center', color='white', fontsize=7, fontweight='bold')
            else:
                # 普通订单：沿用原有颜色方案

                ax.barh(machine, end - start, left=start, color=color, edgecolor='black')
                ax.text(start + (end - start) / 2, machine, f"J{job}-O{op_idx}",
                    va='center', ha='center', color='white', fontsize=7, fontweight='bold')

        if dynamic_events:
            machine_task_counters = [0] * num_machines
            sorted_schedule = sorted(schedule, key=lambda x: x['end'])
            for op in sorted_schedule:
                machine_id, end_time = op["machine"], op["end"]
                machine_task_counters[machine_id] += 1
                for event in dynamic_events:
                    if event['type'] == 'machine_breakdown_on_nth_task':
                        trigger = event['trigger']
                        if trigger['machine'] == machine_id and trigger['task_count'] == machine_task_counters[
                            machine_id]:
                            duration = event['duration']
                            ax.barh(machine_id, duration, left=end_time, color='#8B0000', edgecolor='black', hatch='xxx',
                                    alpha=0.8)
                            ax.text(end_time + duration / 2, machine_id, "Breakdown",
                                    va='center', ha='center', color='white', fontsize=8, fontweight='bold')

        ax.set_title(solver_name);
        ax.set_xlabel("Time")
        ax.set_yticks(range(num_machines));
        ax.set_yticklabels([f"M{i}" for i in range(num_machines)])
        ax.grid(True, axis='x')

    legend_handles = []
    # (此处的图例生成逻辑保持您提供的版本)
    if new_job_start_index is not None:
        for j in range(new_job_start_index):
            legend_handles.append(Patch(color=colors[j % len(colors)], label=f"Job {j}"))
    else:
        for j in range(num_jobs):
            legend_handles.append(Patch(color=colors[j % len(colors)], label=f"Job {j}"))

    if new_job_start_index is not None:
        for j in range(new_job_start_index, num_jobs):
            legend_handles.append(
                Patch(facecolor=colors[j % len(colors)], edgecolor='black', hatch='///', label=f"Job {j} (New)")
            )

    if dynamic_events and any(
            e['type'] in ['machine_breakdown', 'machine_breakdown_on_nth_task'] for e in dynamic_events):
        legend_handles.append(
            mpatches.Patch(facecolor='#8B0000', hatch='xxx', edgecolor='black', label='Machine Breakdown'))

    if n > 0: axs[-1].legend(handles=legend_handles, loc="upper right")
    fig.suptitle("Dynamic Gantt Charts Comparison by Solver", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_comprehensive_comparison(results, problem, pretty_names, dynamic_events=None, initial_num_jobs=None):
    """
    一个“主”绘图函数，目前只生成动态Gantt图。
    """
    print("\n" + "=" * 50)
    print("📊 Generating Comprehensive Analysis Plots...")
    print("=" * 50)
    print("--> Plotting Dynamic Gantt Charts...")

    schedules_to_plot = {}
    for name, res in results.items():
        if res and res.get("all_schedules") and res["all_schedules"]:
            pretty_name = pretty_names.get(name, name)
            schedules_to_plot[pretty_name] = res["all_schedules"][-1]

    final_num_jobs = initial_num_jobs or 0
    final_num_jobs += sum(1 for e in (dynamic_events or []) if e['type'] == 'new_job_arrival')

    if schedules_to_plot:
        plot_dynamic_multiple_gantt(
            schedules_dict=schedules_to_plot,
            num_machines=problem.num_machines,
            num_jobs=final_num_jobs,
            dynamic_events=dynamic_events,
            new_job_start_index=initial_num_jobs,
            figsize=(20, len(schedules_to_plot) * 4)
        )
    else:
        print("    (Skipped: No valid schedules found to plot Gantt charts)")
    print("\n✅ All plots have been generated.")