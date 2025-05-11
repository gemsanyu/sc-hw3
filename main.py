import pathlib
import json
from typing import List, Tuple
import multiprocessing as mp

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize

from pymoo.util.reference_direction import UniformReferenceDirectionFactory
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import StarmapParallelization

from salb import Salb, Task, DuplicateElimination
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
        
    # pool = mp.Pool(4)
    # runner = StarmapParallelization(pool.starmap)
    problem = Salb(tasks, precedence_list)#, elementwise_runner=runner)
    duplicate_elimination = DuplicateElimination(problem.task_times)
    
    algo = NSGA2(pop_size=500)
    algo_name = "NSGA-II"
    
    # ref_dirs = UniformReferenceDirectionFactory(2, n_partitions=500).do()
    # algo = MOEAD(ref_dirs=ref_dirs)
    # algo_name = "MOEAD"
    
    # algo = SMSEMOA(pop_size=500)
    # algo_name = "SMSEMOA"
    
    
    result = minimize(problem, algo, verbose=True)
    final_pop = result.algorithm.opt
    no_duplicate_pop = duplicate_elimination.do(final_pop)
    
    X = np.stack([no_duplicate_pop[i].X for i in range(len(no_duplicate_pop))])
    pf = np.stack([no_duplicate_pop[i].F for i in range(len(no_duplicate_pop))])
    # print(result.pf)
    result_dir = pathlib.Path()/"results"/algo_name
    result_dir.mkdir(parents=True, exist_ok=True)
    f_csv_filepath = result_dir/"f.csv"
    pf_fig_filepath = result_dir/"pf.jpg"
    visualize_pf(pf, pf_fig_filepath)
    np.savetxt(f_csv_filepath.absolute(), pf, delimiter=",", fmt="%g")
    for i,x in enumerate(X):
        solution = problem.decode(x)
        fig_filepath = result_dir/(f"solution_visualization_{i}.jpg")
        solution.save_visualization(fig_filepath)
    