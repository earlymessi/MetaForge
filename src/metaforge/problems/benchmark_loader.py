
import json
import os
import requests
import tempfile
from metaforge.problems.jobshop import JobShopProblem, Job, Task

def load_job_shop_instance(path, format='orlib'):
    """
    Load a Job Shop Scheduling Problem instance from a file or URL.

    Args:
        path (str): Path to the file or URL.
        format (str): 'orlib' for .txt benchmarks, 'json' for custom format.

    Returns:
        JobShopProblem: The parsed problem.
    """
    if path.startswith("http://") or path.startswith("https://"):
        # Download to temporary file
        response = requests.get(path)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt") as tmp:
            tmp.write(response.text)
            tmp_path = tmp.name
    else:
        tmp_path = path

    if format == 'orlib':
        return _load_orlib_format(tmp_path)
    elif format == 'json':
        return _load_json_format(tmp_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_orlib_format(path):
    """
    OR-Library benchmark format loader.
    Format: first line is num_jobs num_machines
    Each job line contains: machine duration machine duration ...
    """
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    num_jobs, num_machines = map(int, lines[0].split())
    jobs = []

    for job_id, line in enumerate(lines[1:num_jobs+1]):
        parts = list(map(int, line.split()))
        tasks = [Task(machine_id=parts[i], duration=parts[i + 1]) for i in range(0, len(parts), 2)]
        jobs.append(Job(tasks=tasks, id=job_id))

    return JobShopProblem(jobs)


def _load_json_format(path):
    """
    Custom JSON format loader.

    Format:
    {
        "jobs": [
            [ {"machine": 0, "duration": 3}, {"machine": 1, "duration": 2} ],
            ...
        ]
    }
    """
    with open(path, 'r') as f:
        data = json.load(f)

    jobs = []
    for job_id, job_data in enumerate(data["jobs"]):
        tasks = [Task(machine_id=task["machine"], duration=task["duration"]) for task in job_data]
        jobs.append(Job(tasks=tasks, id=job_id))

    return JobShopProblem(jobs)
