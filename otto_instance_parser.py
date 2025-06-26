import pathlib
from typing import List
import json

from salb import Task, Salb

if __name__ == "__main__":
    raw_instance_dir = pathlib.Path()/"instances"/"raw"/"otto-small"
    instance_name = "instance_n=20_1.alb"
    raw_instance_path = raw_instance_dir/instance_name
    
    instance_dict = {}
    instance_dict["tasks"] = []
    lines = []
    with open(raw_instance_path.absolute(), "r") as f:
        lines = f.readlines()
    num_tasks = int(lines[1])
    si = 0
    while ("<task times>" not in lines[si]):
        si+=1
    si+=1
    for i in range(num_tasks):
        line = lines[si+i].split(" ")
        idx, task_time = int(line[0]), float(line[1])
        instance_dict["tasks"].append({
            "idx":idx, 
            "task_time":task_time
        })
    
    si += num_tasks + 1
    while ("<precedence relations>" not in lines[si]):
        si += 1
    si += 1
    instance_dict["precedence_list"] = []
    while ("<end>" not in lines[si]):    
        print(lines[si]) 
        if len(lines[si])<=1:
            break
        precedence = lines[si].split(",")
        idxa, idxb = int(precedence[0]), int(precedence[1])
        instance_dict["precedence_list"].append({
            "i":idxa,
            "j":idxb
        })
        si += 1

    converted_instance_dir = pathlib.Path()/"instances"
    converted_instance_name = instance_name[:-3]+"json"
    converted_instance_path = converted_instance_dir/converted_instance_name
    with open(converted_instance_path.absolute(), "w") as f:
        json.dump(instance_dict, f, indent=4)