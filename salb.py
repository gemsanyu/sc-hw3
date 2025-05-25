import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from queue import Queue

from matplotlib.patches import FancyArrowPatch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


@dataclass
class Task:
    idx: int
    task_time: float
    
    

class Salb:
    def __init__(self, 
                 tasks: List[Task], 
                 precedence_list: List[Tuple[int, int]],
                 num_workstations: int,):
        self.tasks: List[Task] = tasks
        self.precedence_list_original: List[Tuple[int, int]] = precedence_list
        self.num_tasks: int = len(tasks)
        self.num_workstations = num_workstations
        
        self.task_times: np.ndarray = np.asanyarray([task.task_time for task in self.tasks], dtype=float)
        ref_point = (self.num_tasks, np.sum(self.task_times))
        
        self.task_idx_arr_idx_dict = {}
        for i, task in enumerate(self.tasks):
            self.task_idx_arr_idx_dict[task.idx] = i
            
        self.precedence_lists: List[List[int]] = [[] for _ in range(self.num_tasks)]
        self.indegrees: np.ndarray = np.zeros((len(tasks),), dtype=int)
        for (idx_i, idx_j) in self.precedence_list_original:
            i, j = self.task_idx_arr_idx_dict[idx_i], self.task_idx_arr_idx_dict[idx_j]
            self.precedence_lists[i].append(j)
            self.indegrees[j] += 1
        
        self.complete_precedence_matrix: np.ndarray = self.init_complete_precedence_matrix()
        self.dependency_lists:List[List[int]] = self.init_dependency_lists()
        self.precedence_graph: nx.DiGraph = self.get_precedence_graph()
        self.complete_precedence_graph: nx.DiGraph = self.get_complete_precedence_graph()
        self.non_dependency_graph: nx.DiGraph = self.get_non_dependency_graph()
        
        
    def init_dependency_lists(self):
        dependency_lists:List[List[int]] = [[] for _ in range(self.num_tasks)]
        for ti in range(self.num_tasks):
            for nti in range(self.num_tasks):
                if self.complete_precedence_matrix[ti,nti]:
                    dependency_lists[nti].append(ti)
        return dependency_lists
        
    def get_non_dependency_graph(self):
    # Assume `self.complete_precedence_graph` is a DiGraph
        G = self.complete_precedence_graph
        nodes = list(G.nodes)
        n = len(nodes)

        ndg = nx.Graph()  # non-dependency graph is undirected
        ndg.add_nodes_from(nodes)

        # Check for non-dependent pairs
        for i in range(n):
            for j in range(i+1, n):
                u, v = nodes[i], nodes[j]
                # If no path between u and v in either direction, they are non-dependent
                if not nx.has_path(G, u, v) and not nx.has_path(G, v, u):
                    ndg.add_edge(u, v)

        return ndg
      
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
        
    def visualize_non_dependency_graph(self, chosen_tasks: Optional[List[int]] = None):
        # If chosen_tasks is provided, filter the graph to only include the chosen tasks
        if chosen_tasks is not None:
            # Get subgraph with only the chosen tasks
            subgraph = self.non_dependency_graph.subgraph(chosen_tasks)
        else:
            # Use the entire graph
            subgraph = self.non_dependency_graph

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
    