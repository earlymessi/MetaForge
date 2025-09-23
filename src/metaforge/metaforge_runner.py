import time
from metaforge.solvers.tabu_search import TabuSearchSolver
from metaforge.solvers.simulated_annealing import SimulatedAnnealingSolver
from metaforge.solvers.genetic_algorithm import GeneticAlgorithmSolver
from metaforge.solvers.ant_colony import AntColonySolver
from metaforge.solvers.q_learning import QAgentSolver
from metaforge.solvers.dqn_solver import DQNAgentSolver, DQNAgentSolverReplay
from metaforge.solvers.neuroevolution_solver import NeuroevolutionSolver
from metaforge.solvers.dqn_dyn import DQNAgentSolverReplayDynamic #引入动态情景

def run_solver(solver_name, problem, params=None, track_history=True, track_schedule=True, dynamic_events=None):
    """
    Runs a solver by name with optional tracking and returns a unified result dict.

    Args:
        solver_name (str): One of "ga", "sa", "ts", "aco", "q", "dqn", "dqn-replay", "neuroevo"
        problem: Instance of JobShopProblem
        params (dict): Optional solver parameters
        track_history (bool): Whether to log best scores
        track_schedule (bool): Whether to log schedule evolution

    Returns:
        dict: {
            "solver": name,
            "solution": best_solution,
            "makespan": best_score,
            "history": [...],
            "schedules": [...],
            "time": total_seconds
        }
    """
    params = params or {}
    start = time.time()

    if solver_name.lower() == "ts":
        solver = TabuSearchSolver(problem, **params)
        solution, score, history, schedules = solver.run(track_schedule=track_schedule)
    elif solver_name.lower() == "sa":
        solver = SimulatedAnnealingSolver(problem, **params)
        solution, score, history, temps, schedules = solver.run(
            track_schedule=track_schedule
        )
    elif solver_name.lower() == "ga":
        solver = GeneticAlgorithmSolver(problem, **params)
        solution, score, history, schedules = solver.run(
            track_history=track_history,
            track_schedule=track_schedule
        )
    elif solver_name.lower() == "aco":
        solver = AntColonySolver(problem, **params)
        solution, score, history, schedules = solver.run(
            track_history=track_history,
            track_schedule=track_schedule
        )
    elif solver_name.lower() == "q":
        solver = QAgentSolver(problem, **params)
        solution, score, history, schedules = solver.run(
            track_history=track_history,
            track_schedule=track_schedule
        )
    elif solver_name.lower() == "dqn-naive":
        solver = DQNAgentSolver(problem, **params)
        solution, score, history, schedules = solver.run(
            track_history=track_history,
            track_schedule=track_schedule
        )
    elif solver_name.lower() == "dqn-replay":
        solver = DQNAgentSolverReplay(problem, **params)
        solution, score, history, schedules = solver.run(
            track_history=track_history,
            track_schedule=track_schedule
        )
    elif solver_name.lower() == "neuroevo":
        solver = NeuroevolutionSolver(problem, **params)
        solution, score, history, schedules = solver.run(
            track_history=track_history,
            track_schedule=track_schedule
        )
        # === 2. 新增 elif 分支来处理动态求解器 ===
    elif solver_name.lower() == "dqn-dyn":
        # 实例化动态求解器
        solver = DQNAgentSolverReplayDynamic(problem, dynamic_events=dynamic_events, **params)

        # 1. 将返回的整个字典存入一个变量中，不再进行解包
        result_dict = solver.run(
            track_history=track_history,
            track_schedule=track_schedule
        )

        # 2. 从字典中通过键（key）来获取需要的值
        score = result_dict.get("makespan")
        history = result_dict.get("history")
        schedules = result_dict.get("all_schedules")  # 这已经包含了最终的最佳调度方案
        solution = result_dict.get("solution")  # solution 就是最佳调度方案

    else:
        raise ValueError(f"未知的求解器: {solver_name}")

    total_time = time.time() - start

    return {
        "solver": solver_name,
        "solution": solution,
        "makespan": score,
        "history": history,
        "schedules": schedules,
        "time": total_time
    }
