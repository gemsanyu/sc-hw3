from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self, num_available_work_stations: int, num_tasks: int, task_times: np.ndarray):
        self.num_available_work_stations = num_available_work_stations
        self.num_tasks = num_tasks
        self.task_times = task_times
        
        self.task_sequences: List[List[int]] = [[] for _ in range(num_available_work_stations)]
    
        self.num_used_work_stations: int = 0
    
    
    @property
    def cycle_time(self)->float:
        total_times_arr = np.asanyarray([np.sum(self.task_times[self.task_sequences[wi]]) for wi in range(self.num_used_work_stations)])
        return np.max(total_times_arr)
    
    @property
    def obj(self)->Tuple[int, float]:
        return self.num_used_work_stations, self.cycle_time
        
    def visualize(self):
        fig, ax = plt.subplots(figsize=(12, 0.8 * self.num_used_work_stations + 2))
        
        colors = plt.cm.get_cmap('tab20', self.num_tasks)

        for ws_idx in range(self.num_used_work_stations):
            current_time = 0
            for task_id in self.task_sequences[ws_idx]:
                duration = self.task_times[task_id]
                ax.barh(ws_idx, duration, left=current_time, color=colors(task_id), edgecolor='black')
                ax.text(current_time + duration / 2, ws_idx, f"T{task_id+1}", va='center', ha='center', fontsize=9, color='white')
                current_time += duration

        ax.set_yticks(range(self.num_used_work_stations))
        ax.set_yticklabels([f"WS {i+1}" for i in range(self.num_used_work_stations)])
        ax.set_xlabel("Time")
        ax.set_title("Workstation Task Schedule")
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show() 
    
    def save_visualization(self, filepath):
        fig, ax = plt.subplots(figsize=(12, 0.8 * self.num_used_work_stations + 2))
        
        colors = plt.cm.get_cmap('tab20', self.num_tasks)

        for ws_idx in range(self.num_used_work_stations):
            current_time = 0
            for task_id in self.task_sequences[ws_idx]:
                duration = self.task_times[task_id]
                ax.barh(ws_idx, duration, left=current_time, color=colors(task_id), edgecolor='black')
                ax.text(current_time + duration / 2, ws_idx, f"T{task_id+1}", va='center', ha='center', fontsize=9, color='white')
                current_time += duration

        ax.set_yticks(range(self.num_used_work_stations))
        ax.set_yticklabels([f"WS {i+1}" for i in range(self.num_used_work_stations)])
        ax.set_xlabel("Time")
        ax.set_title(f"Workstation Task Schedule, CT={self.cycle_time}")
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        # plt.show() 
        
        # Save figure to file
        plt.savefig(filepath.absolute(), bbox_inches='tight')  # e.g., 'schedule.png' or 'schedule.pdf'
        plt.close()
        # plt.show()
    