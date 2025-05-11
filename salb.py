import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from queue import Queue

from matplotlib.patches import FancyArrowPatch
import numpy as np
import numba as nb
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.individual import Individual
import networkx as nx
import matplotlib.pyplot as plt

from solution import Solution

@dataclass
class Task:
    idx: int
    task_time: float


@nb.njit(nb.types.Tuple((nb.int64[:,:],nb.int64,nb.float64))(nb.float64[:], nb.float64[:]))
def decode_without_sequencing(x:np.ndarray, task_times:np.ndarray)->Tuple[np.ndarray, int, float]:
    num_tasks = x.shape[0]
    assignment_map = np.zeros((num_tasks, num_tasks), dtype=np.int64)
    for i in range(num_tasks):
        wi = int(math.floor(x[i]*num_tasks))
        assignment_map[wi,i]=True
    
    num_used_workstations:int = 0
    for wi in range(num_tasks):
        for i in range(num_tasks):
            if assignment_map[wi,i]:
                num_used_workstations += 1
                break
    cycle_times = np.zeros((num_tasks), dtype=np.float64)
    for wi in range(num_tasks):
        cycle_times[wi] = np.sum(assignment_map[wi,:]*task_times)   
    max_cycle_time = np.max(cycle_times)
    # obj = (num_used_workstations, max_cycle_time)
    
    return assignment_map, num_used_workstations, max_cycle_time

    
class DuplicateElimination(ElementwiseDuplicateElimination):
    def __init__(self, task_times: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.task_times:np.ndarray = task_times
    
    def is_equal(self, a:Individual, b:Individual)->bool:
        assignment_map_a, num_used_workstations_a, max_cycle_time_a = decode_without_sequencing(a.x, self.task_times)
        assignment_map_b, num_used_workstations_b, max_cycle_time_b = decode_without_sequencing(b.x, self.task_times)
        f_a = (num_used_workstations_a, max_cycle_time_a)
        f_b = (num_used_workstations_b, max_cycle_time_b)
        if f_a[0] != f_b[0]:
            return False
        
        if f_a[0] == f_b[0]:
            if math.fabs(f_a[1]-f_b[1]) > 1e-5:
                return False
        return True
    

class Salb(ElementwiseProblem):
    def __init__(self, 
                 tasks: List[Task], 
                 precedence_list: List[Tuple[int, int]],
                 elementwise=True, 
                 **kwargs):
        super().__init__(elementwise, **kwargs)            
        self.tasks: List[Task] = tasks
        self.precedence_list_original: List[Tuple[int, int]] = precedence_list
        self.num_tasks: int = len(tasks)
        self.n_var = self.num_tasks
        self.xl = np.zeros((self.n_var,), dtype=float) 
        self.xu = np.ones((self.n_var,), dtype=float)
        self.n_obj = 2
        
        self.task_times: np.ndarray = np.asanyarray([task.task_time for task in self.tasks], dtype=float)
        self.task_idx_arr_idx_dict = {}
        for i, task in enumerate(self.tasks):
            self.task_idx_arr_idx_dict[task.idx] = i
            
        self.precedence_lists: List[List[int]] = [[] for _ in range(self.num_tasks)]
        for (idx_i, idx_j) in self.precedence_list_original:
            i, j = self.task_idx_arr_idx_dict[idx_i], self.task_idx_arr_idx_dict[idx_j]
            self.precedence_lists[i].append(j)
        
        self.complete_precedence_matrix: np.ndarray = self.init_complete_precedence_matrix()
        self.precedence_graph: nx.DiGraph = self.get_precedence_graph()
        self.complete_precedence_graph: nx.DiGraph = self.get_complete_precedence_graph()
    
    
    def get_precedence_graph(self)->nx.DiGraph:
        graph = nx.DiGraph()

        # Add nodes with task time as label
        for task in self.tasks:
            graph.add_node(task.idx, label=f"{task.idx}\n{task.task_time:.1f}")

        # Add directed edges for precedence constraints
        for u, v in self.precedence_list_original:
            graph.add_edge(u, v)
        return graph
    
    def get_complete_precedence_graph(self)->nx.DiGraph:
        graph = nx.DiGraph()

        # Add nodes with task time as label
        for task in self.tasks:
            graph.add_node(task.idx, label=f"{task.idx}\n{task.task_time:.1f}")

        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                if self.complete_precedence_matrix[i,j]:
                    graph.add_edge(i+1,j+1)
        return graph
    
    def visualize_complete_precedence_graph(self, chosen_tasks: Optional[List[int]] = None):
        # If chosen_tasks is provided, filter the graph to only include the chosen tasks
        if chosen_tasks is not None:
            # Get subgraph with only the chosen tasks
            subgraph = self.complete_precedence_graph.subgraph(chosen_tasks)
        else:
            # Use the entire graph
            subgraph = self.complete_precedence_graph

        # Layout for nicer graph (you can experiment with different layouts)
        # pos = nx.spring_layout(subgraph, seed=42)  # Positions of nodes for visualization
        pos = self.layered_topo_pos(subgraph)

        fig, ax = plt.subplots()


        # Draw nodes and labels
        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color='skyblue', node_size=1500)
        nx.draw_networkx_labels(subgraph, pos, ax=ax)

        # Draw edges with curvature
        for u, v in subgraph.edges():
            if u == v:
                continue  # Skip self-loop
            rad = 0.2  # curvature radius
            arrow = FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle='-|>',                # Better-looking arrowhead
                color='gray',
                linewidth=2,
                shrinkA=15, shrinkB=15,          # Shrink to avoid node overlap
                mutation_scale=20               # Controls size of arrowhead
            )
            ax.add_patch(arrow)

        # ax.set_title("Complete Precedence (Sub)Graph")
        ax.axis('off')
        x_vals, y_vals = zip(*pos.values())
        plt.xlim(min(x_vals) - 1, max(x_vals) + 1)
        plt.ylim(min(y_vals) - 1, max(y_vals) + 1)
        plt.tight_layout()
        plt.show()
        
    def visualize_precedence_graph(self, chosen_tasks: Optional[List[int]] = None):
        # If chosen_tasks is provided, filter the graph to only include the chosen tasks
        if chosen_tasks is not None:
            # Get subgraph with only the chosen tasks
            subgraph = self.precedence_graph.subgraph(chosen_tasks)
        else:
            # Use the entire graph
            subgraph = self.precedence_graph

        # Layout for nicer graph (you can experiment with different layouts)
        # pos = nx.spring_layout(subgraph, seed=42)  # Positions of nodes for visualization
        pos = self.layered_topo_pos(subgraph)

        fig, ax = plt.subplots()


        # Draw nodes and labels
        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color='skyblue', node_size=1500)
        nx.draw_networkx_labels(subgraph, pos, ax=ax)

        # Draw edges with curvature
        for u, v in subgraph.edges():
            if u == v:
                continue  # Skip self-loop
            rad = 0.2  # curvature radius
            arrow = FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle='-|>',                # Better-looking arrowhead
                color='gray',
                linewidth=2,
                shrinkA=15, shrinkB=15,          # Shrink to avoid node overlap
                mutation_scale=20               # Controls size of arrowhead
            )
            ax.add_patch(arrow)

        # ax.set_title("Complete Precedence (Sub)Graph")
        ax.axis('off')
        x_vals, y_vals = zip(*pos.values())
        plt.xlim(min(x_vals) - 1, max(x_vals) + 1)
        plt.ylim(min(y_vals) - 1, max(y_vals) + 1)
        plt.tight_layout()
        plt.show()
    
    
    def layered_topo_pos(self, subgraph: nx.DiGraph):
        G_full = self.complete_precedence_graph  # Full graph with all dependencies

        from collections import deque

        # Compute topological levels using full graph
        in_deg = dict(G_full.in_degree())
        levels = {}
        q = deque([n for n in G_full.nodes if in_deg[n] == 0])

        for node in q:
            levels[node] = 0

        while q:
            u = q.popleft()
            for v in G_full.successors(u):
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    levels[v] = levels[u] + 1
                    q.append(v)

        # Group subgraph nodes by their topological level
        layer_nodes = {}
        for node in subgraph.nodes:
            level = levels.get(node, 0)
            layer_nodes.setdefault(level, []).append(node)

        # Assign positions: x = level, y = vertical index
        pos = {}
        for x, nodes_at_level in layer_nodes.items():
            for y, node in enumerate(sorted(nodes_at_level)):  # sort to make vertical order stable
                pos[node] = (x, -y)

        return pos

        
        
    def init_complete_precedence_matrix(self)->np.ndarray:
        complete_precedence_matrix: np.ndarray = np.zeros((self.num_tasks, self.num_tasks), dtype=bool)
        for i in range(self.num_tasks):
            q:Queue = Queue()
            for j in self.precedence_lists[i]:
                q.put(j)
            while not q.empty():
                j = q.get()
                complete_precedence_matrix[i,j] = True
                for next_j in self.precedence_lists[j]:
                    q.put(next_j)
        return complete_precedence_matrix
    
    def get_possible_sequence(self, tasks_idx_list:List[int])->List[int]:
        # Induce subgraph over the selected nodes
        subgraph = self.complete_precedence_graph.subgraph([task_idx+1 for task_idx in tasks_idx_list]).copy()
        
        # Now perform topological sort (raises error if cycle exists)
        sequence = list(nx.topological_sort(subgraph))
        for i in range(len(sequence)):
            sequence[i]-=1
        return sequence
    
    def decode(self, x)->Solution:
        num_tasks = len(x)
        solution: Solution = Solution(num_tasks, num_tasks, self.task_times)
        
        for i in range(self.num_tasks):
            wi = int(math.floor(x[i]*solution.num_available_work_stations))
            solution.task_ws_assignment[i]=wi
        # rearrange workstations, so only used workstations are first
        used_workstations_idx = list(set([wi for wi in solution.task_ws_assignment]))
        new_wi_dict = {wi:i for i, wi in enumerate(used_workstations_idx)}
        for i in range(self.num_tasks):
            wi = solution.task_ws_assignment[i]
            solution.task_ws_assignment[i] = new_wi_dict[wi]
        solution.num_used_work_stations = max(solution.task_ws_assignment)+1
        for i in range(self.num_tasks):
            wi = solution.task_ws_assignment[i]
            solution.task_sequences[wi].append(i)
        
        for wi in range(solution.num_used_work_stations):
            solution.task_sequences[wi] = self.get_possible_sequence(solution.task_sequences[wi])
        return solution
    
    def _evaluate(self, x, out, *args, **kwargs):
        assignment_map, num_used_workstations, max_cycle_time = decode_without_sequencing(x, self.task_times)
        obj = (num_used_workstations, max_cycle_time)
        out["F"] = obj
        