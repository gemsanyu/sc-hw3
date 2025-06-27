import json
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from salb import Salb, Task
from solution import Solution




def visualize_pf(pf: np.ndarray, filepath=None):
    plt.scatter(pf[:, 0], pf[:, 1])
    plt.xlabel("Number of workstations")
    plt.ylabel("Cycle time")
    plt.xticks(pf[:,0])
    plt.yticks(pf[:,1])
    if filepath is not None:
        plt.savefig(filepath.absolute(), bbox_inches='tight', dpi=600)
    else:
        plt.show()
    plt.close()

def is_topo_order_valid(topo_order:np.ndarray, precedence_adjacency_list: List[List[int]])->bool:
    position = {ti:i for i, ti in enumerate(topo_order)}
    for ti, adj_list in enumerate(precedence_adjacency_list):
        for nti in adj_list:
            pti, pnti = position[ti], position[nti]
            if pti > pnti:
                return False
    return True

def precompute_valid_masks(task_times: np.ndarray, dependency_lists:List[List[int]])->List[Tuple[int, np.ndarray, float]]:
    num_tasks = len(task_times)
    valid_mask_binarr_list:List[Tuple[int, np.ndarray, float]] = [(0,np.zeros((num_tasks,), dtype=int),0)]
    mask_dict = {}
    powers = 2 ** np.arange(num_tasks)
    for ti in range(num_tasks):
        binarr = np.zeros((num_tasks,), dtype=int)
        binarr[dependency_lists[ti]+[ti]]=1
        mask = np.sum(binarr * powers)
        mask_dict[mask]=True
        total_task_time = task_times[dependency_lists[ti]+[ti]].sum()
        valid_mask_binarr_list.append((mask, binarr, total_task_time))
    return valid_mask_binarr_list
            
def get_mask_adjacency_list(masks_arr: np.ndarray)->List[List[int]]:
    num_valid_masks = len(masks_arr)
    adj_list: List[List[int]] = [[] for _ in range(num_valid_masks)]
    for mi in range(num_valid_masks):
        mask_i = masks_arr[mi]
        for mj in range(num_valid_masks):
            mask_j = masks_arr[mj]
            if mask_i >= mask_j:
                continue
            if (mask_i & mask_j) != mask_i:
                continue
            adj_list[mi].append(mj) 
    return adj_list
            
        

def mask_to_binarr(mask: int, num_tasks: int) -> np.ndarray:
    binarr = np.zeros((num_tasks,), dtype=bool)
    for ti in range(num_tasks):
        if (mask & (1 << ti)) != 0:
            binarr[ti] = True
    return binarr

def dp(mi:int, 
       wi:int,
       masks_arr:np.ndarray, 
       mask_total_task_times_arr:np.ndarray,
       adjacency_lists:List[List[int]], 
       memo:np.ndarray,
       policy:np.ndarray)->float:
    num_masks = masks_arr.shape[0]
    # no more tasks, ends
    if mi == num_masks-1:
        return 0.
    if memo[mi, wi]>-1:
        return memo[mi, wi]
    
    current_total_task_time = mask_total_task_times_arr[mi]
    # at the last available workstations, all remaining tasks must be assigned here
    if wi==0:
        total_all_time = mask_total_task_times_arr[-1]
        tt_diff = total_all_time-current_total_task_time
        memo[mi, wi] = tt_diff
        policy[mi, wi] = num_masks-1
        return memo[mi, wi]
    
    adj_list = adjacency_lists[mi]
    best_ct = 999999
    best_mj = 999999
    for mj in adj_list:
        next_mask, next_total_task_time = masks_arr[mj], mask_total_task_times_arr[mj]
        tt_diff = next_total_task_time-current_total_task_time
        ct = max(tt_diff, dp(mj, wi-1, masks_arr, mask_total_task_times_arr, adjacency_lists, memo, policy))
        if best_ct > ct:
            best_ct = ct
            best_mj = mj
    memo[mi, wi] = best_ct
    policy[mi, wi] = best_mj
    return memo[mi, wi]    
    
def get_assignments(num_tasks: int, 
                    num_workstations:int, 
                    masks_arr: np.ndarray, 
                    policy:np.ndarray)->List[List[int]]:
    mi = 0
    wi = num_workstations-1
    assignments:List[List[int]] = []
    while wi >= 0:
        mask_i = masks_arr[mi]
        mj = policy[mi, wi]
        mask_j = masks_arr[mj]
        mask_diff = mask_j & ~mask_i
        binarr = mask_to_binarr(mask_diff, num_tasks)
        assignment = np.nonzero(binarr)[0].tolist()
        assignments.append(assignment)
        mi = mj
        wi -= 1
    return assignments


@dataclass
class State:
    mask: int
    binarr: np.ndarray
    current_ws_load: float
    cycle_time: float
    num_workstations: int
        

def isDominateSoft(state_a: State, state_b: State)->bool:
    # is_a_superset_b = (state_a.mask & state_b.mask) == state_b.mask
    # if not is_a_superset_b:
    #     return False
    is_same_mask =state_a.mask == state_b.mask
    if not is_same_mask:
        return False
    is_dominate = state_a.cycle_time <= state_b.cycle_time and  state_a.num_workstations <= state_b.num_workstations
    return is_dominate 

def isDominateHard(state_a: State, state_b: State)->bool:
    is_a_superset_b = (state_a.mask & state_b.mask) == state_b.mask
    if not is_a_superset_b:
        return False
    is_dominate = state_a.cycle_time <= state_b.cycle_time and  state_a.num_workstations <= state_b.num_workstations
    return is_dominate

def dp_v2(problem: Salb, valid_mask_binarr_list)->List[State]:
    nondominated_solutions: List[State] = []
    _, binarr_example, _ = valid_mask_binarr_list[0]
    num_tasks = len(binarr_example)
    binarr_0 = np.zeros((num_tasks,), dtype=int)
    states_to_expand: List[State] = []
    states_to_expand.append(State(0,binarr_0,0,0,1))
    task_times = problem.task_times
    while len(states_to_expand) > 0:
        print(len(states_to_expand))
        current_state: State = states_to_expand.pop()
        if current_state.binarr.sum()==num_tasks:
            is_new_solution_nondominated = True
            for solution in nondominated_solutions:
                if isDominateHard(solution, current_state):
                    is_new_solution_nondominated = False 
                    break
            if is_new_solution_nondominated:
                nondominated_solutions = [solution for solution in nondominated_solutions if not isDominateHard(current_state, solution)]            
                nondominated_solutions.append(current_state)
            continue
        
        current_mask, current_binarr = current_state.mask, current_state.binarr
        # horizontal move
        for i in range(num_tasks):
            if current_binarr[i]==0:
                mask_i, binarr_i, total_task_time_i = valid_mask_binarr_list[i+1]
                new_mask = current_mask | mask_i
                new_binarr = current_binarr | binarr_i
                binarr_diff = binarr_i & (~current_binarr)
                additional_task_time = task_times[binarr_diff].sum()
                new_ws_load = current_state.current_ws_load + additional_task_time
                new_cycle_time = max(current_state.cycle_time, new_ws_load)
                new_state = State(new_mask, new_binarr, new_ws_load, new_cycle_time, current_state.num_workstations)
                is_new_state_nondominated = True
                for si, state in enumerate(states_to_expand):
                    if isDominateSoft(state, new_state):
                        is_new_state_nondominated = False
                        break
                if is_new_state_nondominated:
                    states_to_expand = [state for state in states_to_expand if not isDominateSoft(new_state, state)]
                    states_to_expand.append(new_state)
                
        # vertical move is only available if the current ws is not empty, otherwise, do not let adding ws if current is empty
        if current_state.current_ws_load > 0:
            new_state = State(current_state.mask, current_state.binarr, 0, current_state.cycle_time, current_state.num_workstations+1)
            is_new_state_nondominated = True
            for si, state in enumerate(states_to_expand):
                if isDominateSoft(state, new_state):
                    is_new_state_nondominated = False
                    break
            if is_new_state_nondominated:
                states_to_expand = [state for state in states_to_expand if not isDominateSoft(new_state, state)]
                states_to_expand.append(new_state)
    
    return nondominated_solutions
    

if __name__ == "__main__":
    instance_dir = pathlib.Path()/"instances"
    instance_filepath = instance_dir/"instance_n=50_1.json"
    # instance_filepath = pathlib.Path()/"small_instance.json"
    tasks: List[Task] = []
    precedence_list: List[Tuple[int, int]] = []
    
    with open(instance_filepath.absolute(), "r") as json_file:
        json_data = json.load(json_file)
        task_dict_list = json_data["tasks"]
        for task_dict in task_dict_list:
            task = Task(task_dict["idx"], task_dict["task_time"])
            tasks.append(task)
        precedence_dict_list = json_data["precedence_list"]
        for precedence_dict in precedence_dict_list:
            i, j = precedence_dict["i"], precedence_dict["j"]
            precedence_list.append((i,j))
    task_priorities = np.random.random((len(tasks),))    
    problem = Salb(tasks, precedence_list, len(tasks))
    # problem.visualize_precedence_graph()
    valid_mask_binarr_list = precompute_valid_masks(problem.task_times, problem.dependency_lists)
    
    nondominated_solutions = dp_v2(problem, valid_mask_binarr_list)
    for solution in nondominated_solutions:
        print(solution.mask, solution.num_workstations, solution.cycle_time)
    
    
    
    
    
    
    
    
    
    # print(len(valid_mask_binarr_list))
    # for valid_mask in valid_mask_binarr_list:
    #     print(valid_mask)
    # exit()
    # # for adj_list in adj_lists:
    # #     print(len(adj_list))
    
    # num_valid_masks = len(valid_mask_binarr_list)
    # memo = np.full((num_valid_masks, problem.num_tasks+1), -1, dtype=float)
    # policy = np.zeros((num_valid_masks, problem.num_tasks+1), dtype=int)
    # masks_arr = np.asanyarray([mask for (mask, _, _) in valid_mask_binarr_list])
    # binarr_arr = np.stack([binarr for (_, binarr, _) in valid_mask_binarr_list])
    # mask_total_task_times_arr = np.asanyarray([tt_time for (_, _, tt_time) in valid_mask_binarr_list])
    
    # sorted_idx = np.argsort(masks_arr)
    # masks_arr = masks_arr[sorted_idx]
    # binarr_arr = binarr_arr[sorted_idx]
    
    # mask_total_task_times_arr = mask_total_task_times_arr[sorted_idx]
    # adj_lists = get_mask_adjacency_list(masks_arr)
    # print(len(adj_lists))
    # solutions:List[Solution] = []
    # pf = []
    # for wi in range(problem.num_tasks):
    #     best_ct = dp(0, wi, masks_arr, mask_total_task_times_arr, adj_lists, memo, policy)    
    #     assignments = get_assignments(problem.num_tasks, wi+1, masks_arr, policy)
    #     solution = Solution(wi+1, problem.num_tasks, problem.task_times)
    #     for wj, assignment in enumerate(assignments):
    #         if len(assignment)==0:
    #             continue
    #         assignment = problem.get_possible_sequence(assignment)
    #         solution.task_sequences[wj] = assignment
    #     num_used_workstations = 0
    #     for assignment in assignments:
    #         if len(assignment)>0:
    #             num_used_workstations += 1
    #     solution.num_used_work_stations = num_used_workstations
    #     solutions.append(solution)
    #     pf.append((solution.obj))
        
    # result_dir = pathlib.Path()/"results"
    # result_dir.mkdir(parents=True, exist_ok=True)
    # for i, solution in enumerate(solutions):
        
        
    #     filepath = result_dir/(f"solutions-{i}.jpg")
    #     solution.save_visualization(filepath)
    
    # pf = np.asanyarray(pf)
    # filepath = result_dir/"pf.jpg"
    # visualize_pf(pf[:18])
                #  , filepath)
            
    
    