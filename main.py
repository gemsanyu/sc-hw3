import json
import multiprocessing as mp
import pathlib
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from salb import Salb, Task
from solution import Solution
from dp import dp_caller

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


if __name__ == "__main__":
    instance_filepath = pathlib.Path()/"instance.json"
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
        
    problem = Salb(tasks, precedence_list)
    list_of_assignments = []
    ts_order = None
    for ts in nx.all_topological_sorts(problem.precedence_graph):
        task_times = problem.task_times
        ts_arr = np.asanyarray(ts, dtype=int)
        ts_order = ts_arr
        ordered_task_times = task_times[ts_arr-1]
        list_of_assignments = dp_caller(ordered_task_times)
        break
    correct_list_of_assignments = []
    solutions: List[Solution] = []
    
    for assignments in list_of_assignments:
        real_assignments = []
        for assignment in assignments:
            if len(assignment)==0:
                continue
            real_assignment = ts_order[assignment]
            real_assignments.append(real_assignment)
        num_used_workstations = len(real_assignments)
        solution = Solution(num_used_workstations, len(tasks), task_times)
        solution.num_used_work_stations = num_used_workstations
        for wi, real_assignment in enumerate(real_assignments):
            for ti in real_assignment:
                solution.task_ws_assignment[ti-1]=wi
            solution.task_sequences[wi] = real_assignment-1
        
        
    
        
    
    