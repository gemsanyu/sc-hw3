from typing import Tuple, List

import numpy as np
import numba as nb


def dp_caller(task_times: np.ndarray, num_workstations:int=-1)->List[List[List[int]]]:
    num_tasks = len(task_times)
    num_workstations = num_tasks
    memo = np.empty((num_tasks, num_workstations), dtype=float)
    policy = np.empty((num_tasks, num_workstations), dtype=int)
    memo, policy = dp(memo, policy, task_times)
    
    list_of_assignments:List[List[List[int]]] = []
    for nw in range(num_workstations):
        print("nw:",nw)
        assignments:List[List[int]] = [[] for _ in range(nw+1)]
        wi = nw
        ti = num_tasks-1
        while ti > -1 and wi>0:
            tj = policy[ti, wi]
            if tj==-1:
                wi -= 1
            else:
                assignments[wi] = list(range(tj, ti+1))
                ti = tj-1
            wi -= 1
        if wi == 0 and ti >-1:
            assignments[0] = list(range(ti+1))
        list_of_assignments += [assignments]
    return list_of_assignments

@nb.njit(nb.types.Tuple((nb.float64[:,:], nb.int64[:,:]))(nb.float64[:,:], nb.int64[:,:],nb.float64[:]))
def dp(memo:np.ndarray, policy:np.ndarray, task_times:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
    num_tasks, num_workstations = memo.shape
    
    # base case
    for ti in range(num_tasks):
        memo[ti, 0] = np.sum(task_times[:ti+1])
        policy[ti, 0] = 0
    for wi in range(num_workstations):
        memo[0, wi] = task_times[0]
        policy[0, wi] = 0
    
    for ti in range(1, num_tasks):
        for wi in range(1, num_workstations):
            best_ct = memo[ti, wi-1]
            best_tj = -1
            current_load = 0
            for tj in range(ti, -1, -1):
                current_load += task_times[tj]
                if tj == 0:
                    current_ct = current_load
                else:
                    current_ct = max(current_load, memo[tj-1, wi-1])
                if best_ct > current_ct:
                    best_tj = tj
                    best_ct = current_ct
            memo[ti, wi] = best_ct
            policy[ti, wi] = best_tj
    return memo, policy