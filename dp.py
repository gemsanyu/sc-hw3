import numpy as np
from salb import Salb, Solution


def dp_caller(problem: Salb, last_known_best_ct:float=99999999) -> Solution:
    memo = np.full((problem.num_tasks, problem.num_tasks), fill_value=-1, dtype=float)

    best_ct = dp(0,0, memo, problem.num_tasks, problem.num_tasks, problem.task_times, last_known_best_ct)
    solution = Solution(problem.num_tasks, problem.num_tasks, problem.task_times)
    return solution


def dp(ti: int, wi: int, memo: np.ndarray, num_tasks:int, num_work_stations:int, task_times: np.ndarray, last_known_best_ct:float)->float:
    if ti == num_tasks:
        return 0
    if memo[ti, wi] > 0:
        return memo[ti, wi]
    if wi == num_work_stations-1:
        memo[ti, wi] = np.sum(task_times[ti:])
        return memo[ti, wi]
    load: float = 0.
    best_ct: float = 9999999
    for i in range(ti, num_tasks):
        load += task_times[i]
        if load > last_known_best_ct:
            break
        ct = max(dp(i+1, wi+1, memo, num_tasks, num_work_stations, task_times, last_known_best_ct), load)
        best_ct = min(ct, best_ct)
    memo[ti, wi] = best_ct
    return memo[ti, wi] 