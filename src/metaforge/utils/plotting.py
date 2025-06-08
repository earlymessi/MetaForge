import matplotlib.pyplot as plt

def plot_solver_dashboard(history, temperature=None, title="Solver Performance", solver_name="Solver"):
    """
    Plots convergence and temperature progression in one view.
    """
    num_plots = 2 if temperature else 1
    fig, axs = plt.subplots(1, num_plots, figsize=(12, 4))
    if num_plots == 1:
        axs = [axs]

    # Convergence Plot
    axs[0].plot(history, marker='o', label='Best Makespan')
    axs[0].set_title(f"{solver_name} Convergence")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Best Makespan")
    axs[0].grid(True)
    axs[0].legend()

    # Temperature Decay
    if temperature:
        axs[1].plot(temperature, color='orange', label='Temperature')
        axs[1].set_title(f"{solver_name} Temperature Decay")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Temperature")
        axs[1].grid(True)
        axs[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_convergence_comparison(histories, title="Solver Convergence Comparison"):
    """
    Plots convergence curves for multiple solvers, with clear markers and final score labels.
    
    Args:
        histories (dict): Dictionary {solver_name: [best_score_per_iteration]}
        title (str): Title for the plot
    """
    plt.figure(figsize=(10, 5))

    markers = ['o', 's', '^', 'D', 'x', '*', 'v']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    for i, (solver_name, history) in enumerate(histories.items()):
        plt.plot(
            history,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=solver_name,
            alpha=0.85,
            markevery=max(len(history) // 25, 1)  # reduce clutter if too many points
        )
        # Annotate final value
        final_iter = len(history) - 1
        final_score = history[-1]
        plt.text(
            final_iter,
            final_score,
            f"{solver_name}: {final_score:.2f}",
            fontsize=8,
            va='bottom',
            ha='right',
            color=colors[i % len(colors)]
        )

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best Makespan So Far")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_solver_summary(results, title="Solver Performance Summary"):
    """
    Plots runtime and makespan comparison from multiple solver results.

    Args:
        results (dict): {
            "GA": result_dict,
            "SA": result_dict,
            "TS": result_dict
        }
    """
    solver_names = list(results.keys())
    times = [results[k]["time"] for k in solver_names]
    scores = [results[k]["makespan"] for k in solver_names]
    iters = [len(results[k]["history"]) for k in solver_names]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    axs[0].bar(solver_names, times, color='tab:blue')
    axs[0].set_title("Runtime (seconds)")
    axs[0].set_ylabel("Time (s)")

    axs[1].bar(solver_names, scores, color='tab:green')
    axs[1].set_title("Final Makespan")
    axs[1].set_ylabel("Makespan")

    axs[2].bar(solver_names, iters, color='tab:orange')
    axs[2].set_title("Iterations")
    axs[2].set_ylabel("Count")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
