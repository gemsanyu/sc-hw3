import json
import multiprocessing as mp
import pathlib
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
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
    ts_list = []
    for ts in nx.all_topological_sorts(problem.precedence_graph):
        ts_list.append(ts)
        if len(ts_list)>10000:
            break
    print(len(ts_list))
    # print()
    print(problem.precedence_graph)
    
    