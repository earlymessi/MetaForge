import time

from metaforge.solvers.tabu_search import TabuSearchSolver
from metaforge.solvers.simulated_annealing import SimulatedAnnealingSolver
from metaforge.solvers.genetic_algorithm import GeneticAlgorithmSolver
from metaforge.solvers.ant_colony import AntColonySolver
from metaforge.solvers.q_learning import QAgentSolver
from metaforge.solvers.dqn_solver import DQNAgentSolver, DQNAgentSolverReplay
from metaforge.solvers.neuroevolution_solver import NeuroevolutionSolver

def run_solver(solver_name, problem, params=None, track_history=True, track_schedule=False):
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
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    total_time = time.time() - start

    return {
        "solver": solver_name,
        "solution": solution,
        "makespan": score,
        "history": history,
        "schedules": schedules,
        "time": total_time
    }
