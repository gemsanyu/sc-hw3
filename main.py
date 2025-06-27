import json
import multiprocessing
import pathlib
import time
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
    mi = 0
    while mi < len(valid_mask_binarr_list):
        # print(len(valid_mask_binarr_list))
        mask_i, binarr_i, tt_time_i = valid_mask_binarr_list[mi]
        for mj in range(mi):
            mask_j, binarr_j, tt_time_j = valid_mask_binarr_list[mj]
            new_mask = mask_i | mask_j
            if new_mask in mask_dict.keys():
               continue
            mask_dict[new_mask]=True
            new_binarr = np.maximum(binarr_i, binarr_j)
            new_tt_time = (task_times[new_binarr.astype(bool)]).sum()
            valid_mask_binarr_list.append((new_mask, new_binarr, new_tt_time))
        mi += 1
        
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

def run(instance_name):
    instance_dir = pathlib.Path()/"instances"
    instance_filepath = instance_dir/instance_name
    instance_name_without_ext = instance_name[:-5]
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
    problem = Salb(tasks, precedence_list, 18)
    start_time = time.time()
    valid_mask_binarr_list = precompute_valid_masks(problem.task_times, problem.dependency_lists)
    # for adj_list in adj_lists:
    #     print(len(adj_list))
    
    num_valid_masks = len(valid_mask_binarr_list)
    memo = np.full((num_valid_masks, problem.num_tasks+1), -1, dtype=float)
    policy = np.zeros((num_valid_masks, problem.num_tasks+1), dtype=int)
    masks_arr = np.asanyarray([mask for (mask, _, _) in valid_mask_binarr_list])
    binarr_arr = np.stack([binarr for (_, binarr, _) in valid_mask_binarr_list])
    mask_total_task_times_arr = np.asanyarray([tt_time for (_, _, tt_time) in valid_mask_binarr_list])
    
    sorted_idx = np.argsort(masks_arr)
    masks_arr = masks_arr[sorted_idx]
    binarr_arr = binarr_arr[sorted_idx]
    
    mask_total_task_times_arr = mask_total_task_times_arr[sorted_idx]
    adj_lists = get_mask_adjacency_list(masks_arr)
    precomputation_time = time.time()-start_time
    # print(len(adj_lists))
    solutions:List[Solution] = []
    pf = []
    start_solving_time = time.time()
    for wi in range(problem.num_tasks):
        best_ct = dp(0, wi, masks_arr, mask_total_task_times_arr, adj_lists, memo, policy)    
        assignments = get_assignments(problem.num_tasks, wi+1, masks_arr, policy)
        solution = Solution(wi+1, problem.num_tasks, problem.task_times)
        for wj, assignment in enumerate(assignments):
            if len(assignment)==0:
                continue
            assignment = problem.get_possible_sequence(assignment)
            solution.task_sequences[wj] = assignment
        num_used_workstations = 0
        for assignment in assignments:
            if len(assignment)>0:
                num_used_workstations += 1
        solution.num_used_work_stations = num_used_workstations
        solutions.append(solution)
        pf.append((solution.obj))
    solving_time = time.time() - start_solving_time
    running_time = precomputation_time + solving_time
    result_dir = pathlib.Path()/"results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_filepath = result_dir/f"{instance_name_without_ext}.csv"
    with open(result_filepath.absolute(), "w") as f:
        f.write(f"{instance_name_without_ext},{precomputation_time},{solving_time},{running_time},{len(valid_mask_binarr_list)}")
        
    # result_dir = pathlib.Path()/"results"
    # result_dir.mkdir(parents=True, exist_ok=True)
    # for i, solution in enumerate(solutions):
        
        
    #     filepath = result_dir/(f"solutions-{i}.jpg")
    #     solution.save_visualization(filepath)
    
    # pf = np.asanyarray(pf)
    # filepath = result_dir/"pf.jpg"
    # visualize_pf(pf[:18])

if __name__ == "__main__":
   instance_names = [
       "instance_n=20_1.json",
        "instance_n=20_10.json",
        "instance_n=20_100.json",
        "instance_n=20_101.json",
        "instance_n=20_102.json",
        "instance_n=20_103.json",
        "instance_n=20_104.json",
        "instance_n=20_105.json",
        "instance_n=20_106.json",
        "instance_n=20_107.json",
        "instance_n=20_108.json",
        "instance_n=20_109.json",
        "instance_n=20_11.json",
        "instance_n=20_110.json",
        "instance_n=20_111.json",
        "instance_n=20_112.json",
        "instance_n=20_113.json",
        "instance_n=20_114.json",
        "instance_n=20_115.json",
        "instance_n=20_116.json",
        "instance_n=20_117.json",
        "instance_n=20_118.json",
        "instance_n=20_119.json",
        "instance_n=20_12.json",
        "instance_n=20_120.json",
        "instance_n=20_121.json",
        "instance_n=20_122.json",
        "instance_n=20_123.json",
        "instance_n=20_124.json",
        "instance_n=20_125.json",
        "instance_n=20_126.json",
        "instance_n=20_127.json",
        "instance_n=20_128.json",
        "instance_n=20_129.json",
        "instance_n=20_13.json",
        "instance_n=20_130.json",
        "instance_n=20_131.json",
        "instance_n=20_132.json",
        "instance_n=20_133.json",
        "instance_n=20_134.json",
        "instance_n=20_135.json",
        "instance_n=20_136.json",
        "instance_n=20_137.json",
        "instance_n=20_138.json",
        "instance_n=20_139.json",
        "instance_n=20_14.json",
        "instance_n=20_140.json",
        "instance_n=20_141.json",
        "instance_n=20_142.json",
        "instance_n=20_143.json",
        "instance_n=20_144.json",
        "instance_n=20_145.json",
        "instance_n=20_146.json",
        "instance_n=20_147.json",
        "instance_n=20_148.json",
        "instance_n=20_149.json",
        "instance_n=20_15.json",
        "instance_n=20_150.json",
        "instance_n=20_151.json",
        "instance_n=20_152.json",
        "instance_n=20_153.json",
        "instance_n=20_154.json",
        "instance_n=20_155.json",
        "instance_n=20_156.json",
        "instance_n=20_157.json",
        "instance_n=20_158.json",
        "instance_n=20_159.json",
        "instance_n=20_16.json",
        "instance_n=20_160.json",
        "instance_n=20_161.json",
        "instance_n=20_162.json",
        "instance_n=20_163.json",
        "instance_n=20_164.json",
        "instance_n=20_165.json",
        "instance_n=20_166.json",
        "instance_n=20_167.json",
        "instance_n=20_168.json",
        "instance_n=20_169.json",
        "instance_n=20_17.json",
        "instance_n=20_170.json",
        "instance_n=20_171.json",
        "instance_n=20_172.json",
        "instance_n=20_173.json",
        "instance_n=20_174.json",
        "instance_n=20_175.json",
        "instance_n=20_176.json",
        "instance_n=20_177.json",
        "instance_n=20_178.json",
        "instance_n=20_179.json",
        "instance_n=20_18.json",
        "instance_n=20_180.json",
        "instance_n=20_181.json",
        "instance_n=20_182.json",
        "instance_n=20_183.json",
        "instance_n=20_184.json",
        "instance_n=20_185.json",
        "instance_n=20_186.json",
        "instance_n=20_187.json",
        "instance_n=20_188.json",
        "instance_n=20_189.json",
        "instance_n=20_19.json",
        "instance_n=20_190.json",
        "instance_n=20_191.json",
        "instance_n=20_192.json",
        "instance_n=20_193.json",
        "instance_n=20_194.json",
        "instance_n=20_195.json",
        "instance_n=20_196.json",
        "instance_n=20_197.json",
        "instance_n=20_198.json",
        "instance_n=20_199.json",
        "instance_n=20_2.json",
        "instance_n=20_20.json",
        "instance_n=20_200.json",
        "instance_n=20_201.json",
        "instance_n=20_202.json",
        "instance_n=20_203.json",
        "instance_n=20_204.json",
        "instance_n=20_205.json",
        "instance_n=20_206.json",
        "instance_n=20_207.json",
        "instance_n=20_208.json",
        "instance_n=20_209.json",
        "instance_n=20_21.json",
        "instance_n=20_210.json",
        "instance_n=20_211.json",
        "instance_n=20_212.json",
        "instance_n=20_213.json",
        "instance_n=20_214.json",
        "instance_n=20_215.json",
        "instance_n=20_216.json",
        "instance_n=20_217.json",
        "instance_n=20_218.json",
        "instance_n=20_219.json",
        "instance_n=20_22.json",
        "instance_n=20_220.json",
        "instance_n=20_221.json",
        "instance_n=20_222.json",
        "instance_n=20_223.json",
        "instance_n=20_224.json",
        "instance_n=20_225.json",
        "instance_n=20_226.json",
        "instance_n=20_227.json",
        "instance_n=20_228.json",
        "instance_n=20_229.json",
        "instance_n=20_23.json",
        "instance_n=20_230.json",
        "instance_n=20_231.json",
        "instance_n=20_232.json",
        "instance_n=20_233.json",
        "instance_n=20_234.json",
        "instance_n=20_235.json",
        "instance_n=20_236.json",
        "instance_n=20_237.json",
        "instance_n=20_238.json",
        "instance_n=20_239.json",
        "instance_n=20_24.json",
        "instance_n=20_240.json",
        "instance_n=20_241.json",
        "instance_n=20_242.json",
        "instance_n=20_243.json",
        "instance_n=20_244.json",
        "instance_n=20_245.json",
        "instance_n=20_246.json",
        "instance_n=20_247.json",
        "instance_n=20_248.json",
        "instance_n=20_249.json",
        "instance_n=20_25.json",
        "instance_n=20_250.json",
        "instance_n=20_251.json",
        "instance_n=20_252.json",
        "instance_n=20_253.json",
        "instance_n=20_254.json",
        "instance_n=20_255.json",
        "instance_n=20_256.json",
        "instance_n=20_257.json",
        "instance_n=20_258.json",
        "instance_n=20_259.json",
        "instance_n=20_26.json",
        "instance_n=20_260.json",
        "instance_n=20_261.json",
        "instance_n=20_262.json",
        "instance_n=20_263.json",
        "instance_n=20_264.json",
        "instance_n=20_265.json",
        "instance_n=20_266.json",
        "instance_n=20_267.json",
        "instance_n=20_268.json",
        "instance_n=20_269.json",
        "instance_n=20_27.json",
        "instance_n=20_270.json",
        "instance_n=20_271.json",
        "instance_n=20_272.json",
        "instance_n=20_273.json",
        "instance_n=20_274.json",
        "instance_n=20_275.json",
        "instance_n=20_276.json",
        "instance_n=20_277.json",
        "instance_n=20_278.json",
        "instance_n=20_279.json",
        "instance_n=20_28.json",
        "instance_n=20_280.json",
        "instance_n=20_281.json",
        "instance_n=20_282.json",
        "instance_n=20_283.json",
        "instance_n=20_284.json",
        "instance_n=20_285.json",
        "instance_n=20_286.json",
        "instance_n=20_287.json",
        "instance_n=20_288.json",
        "instance_n=20_289.json",
        "instance_n=20_29.json",
        "instance_n=20_290.json",
        "instance_n=20_291.json",
        "instance_n=20_292.json",
        "instance_n=20_293.json",
        "instance_n=20_294.json",
        "instance_n=20_295.json",
        "instance_n=20_296.json",
        "instance_n=20_297.json",
        "instance_n=20_298.json",
        "instance_n=20_299.json",
        "instance_n=20_3.json",
        "instance_n=20_30.json",
        "instance_n=20_300.json",
        "instance_n=20_301.json",
        "instance_n=20_302.json",
        "instance_n=20_303.json",
        "instance_n=20_304.json",
        "instance_n=20_305.json",
        "instance_n=20_306.json",
        "instance_n=20_307.json",
        "instance_n=20_308.json",
        "instance_n=20_309.json",
        "instance_n=20_31.json",
        "instance_n=20_310.json",
        "instance_n=20_311.json",
        "instance_n=20_312.json",
        "instance_n=20_313.json",
        "instance_n=20_314.json",
        "instance_n=20_315.json",
        "instance_n=20_316.json",
        "instance_n=20_317.json",
        "instance_n=20_318.json",
        "instance_n=20_319.json",
        "instance_n=20_32.json",
        "instance_n=20_320.json",
        "instance_n=20_321.json",
        "instance_n=20_322.json",
        "instance_n=20_323.json",
        "instance_n=20_324.json",
        "instance_n=20_325.json",
        "instance_n=20_326.json",
        "instance_n=20_327.json",
        "instance_n=20_328.json",
        "instance_n=20_329.json",
        "instance_n=20_33.json",
        "instance_n=20_330.json",
        "instance_n=20_331.json",
        "instance_n=20_332.json",
        "instance_n=20_333.json",
        "instance_n=20_334.json",
        "instance_n=20_335.json",
        "instance_n=20_336.json",
        "instance_n=20_337.json",
        "instance_n=20_338.json",
        "instance_n=20_339.json",
        "instance_n=20_34.json",
        "instance_n=20_340.json",
        "instance_n=20_341.json",
        "instance_n=20_342.json",
        "instance_n=20_343.json",
        "instance_n=20_344.json",
        "instance_n=20_345.json",
        "instance_n=20_346.json",
        "instance_n=20_347.json",
        "instance_n=20_348.json",
        "instance_n=20_349.json",
        "instance_n=20_35.json",
        "instance_n=20_350.json",
        "instance_n=20_351.json",
        "instance_n=20_352.json",
        "instance_n=20_353.json",
        "instance_n=20_354.json",
        "instance_n=20_355.json",
        "instance_n=20_356.json",
        "instance_n=20_357.json",
        "instance_n=20_358.json",
        "instance_n=20_359.json",
        "instance_n=20_36.json",
        "instance_n=20_360.json",
        "instance_n=20_361.json",
        "instance_n=20_362.json",
        "instance_n=20_363.json",
        "instance_n=20_364.json",
        "instance_n=20_365.json",
        "instance_n=20_366.json",
        "instance_n=20_367.json",
        "instance_n=20_368.json",
        "instance_n=20_369.json",
        "instance_n=20_37.json",
        "instance_n=20_370.json",
        "instance_n=20_371.json",
        "instance_n=20_372.json",
        "instance_n=20_373.json",
        "instance_n=20_374.json",
        "instance_n=20_375.json",
        "instance_n=20_376.json",
        "instance_n=20_377.json",
        "instance_n=20_378.json",
        "instance_n=20_379.json",
        "instance_n=20_38.json",
        "instance_n=20_380.json",
        "instance_n=20_381.json",
        "instance_n=20_382.json",
        "instance_n=20_383.json",
        "instance_n=20_384.json",
        "instance_n=20_385.json",
        "instance_n=20_386.json",
        "instance_n=20_387.json",
        "instance_n=20_388.json",
        "instance_n=20_389.json",
        "instance_n=20_39.json",
        "instance_n=20_390.json",
        "instance_n=20_391.json",
        "instance_n=20_392.json",
        "instance_n=20_393.json",
        "instance_n=20_394.json",
        "instance_n=20_395.json",
        "instance_n=20_396.json",
        "instance_n=20_397.json",
        "instance_n=20_398.json",
        "instance_n=20_399.json",
        "instance_n=20_4.json",
        "instance_n=20_40.json",
        "instance_n=20_400.json",
        "instance_n=20_401.json",
        "instance_n=20_402.json",
        "instance_n=20_403.json",
        "instance_n=20_404.json",
        "instance_n=20_405.json",
        "instance_n=20_406.json",
        "instance_n=20_407.json",
        "instance_n=20_408.json",
        "instance_n=20_409.json",
        "instance_n=20_41.json",
        "instance_n=20_410.json",
        "instance_n=20_411.json",
        "instance_n=20_412.json",
        "instance_n=20_413.json",
        "instance_n=20_414.json",
        "instance_n=20_415.json",
        "instance_n=20_416.json",
        "instance_n=20_417.json",
        "instance_n=20_418.json",
        "instance_n=20_419.json",
        "instance_n=20_42.json",
        "instance_n=20_420.json",
        "instance_n=20_421.json",
        "instance_n=20_422.json",
        "instance_n=20_423.json",
        "instance_n=20_424.json",
        "instance_n=20_425.json",
        "instance_n=20_426.json",
        "instance_n=20_427.json",
        "instance_n=20_428.json",
        "instance_n=20_429.json",
        "instance_n=20_43.json",
        "instance_n=20_430.json",
        "instance_n=20_431.json",
        "instance_n=20_432.json",
        "instance_n=20_433.json",
        "instance_n=20_434.json",
        "instance_n=20_435.json",
        "instance_n=20_436.json",
        "instance_n=20_437.json",
        "instance_n=20_438.json",
        "instance_n=20_439.json",
        "instance_n=20_44.json",
        "instance_n=20_440.json",
        "instance_n=20_441.json",
        "instance_n=20_442.json",
        "instance_n=20_443.json",
        "instance_n=20_444.json",
        "instance_n=20_445.json",
        "instance_n=20_446.json",
        "instance_n=20_447.json",
        "instance_n=20_448.json",
        "instance_n=20_449.json",
        "instance_n=20_45.json",
        "instance_n=20_450.json",
        "instance_n=20_451.json",
        "instance_n=20_452.json",
        "instance_n=20_453.json",
        "instance_n=20_454.json",
        "instance_n=20_455.json",
        "instance_n=20_456.json",
        "instance_n=20_457.json",
        "instance_n=20_458.json",
        "instance_n=20_459.json",
        "instance_n=20_46.json",
        "instance_n=20_460.json",
        "instance_n=20_461.json",
        "instance_n=20_462.json",
        "instance_n=20_463.json",
        "instance_n=20_464.json",
        "instance_n=20_465.json",
        "instance_n=20_466.json",
        "instance_n=20_467.json",
        "instance_n=20_468.json",
        "instance_n=20_469.json",
        "instance_n=20_47.json",
        "instance_n=20_470.json",
        "instance_n=20_471.json",
        "instance_n=20_472.json",
        "instance_n=20_473.json",
        "instance_n=20_474.json",
        "instance_n=20_475.json",
        "instance_n=20_476.json",
        "instance_n=20_477.json",
        "instance_n=20_478.json",
        "instance_n=20_479.json",
        "instance_n=20_48.json",
        "instance_n=20_480.json",
        "instance_n=20_481.json",
        "instance_n=20_482.json",
        "instance_n=20_483.json",
        "instance_n=20_484.json",
        "instance_n=20_485.json",
        "instance_n=20_486.json",
        "instance_n=20_487.json",
        "instance_n=20_488.json",
        "instance_n=20_489.json",
        "instance_n=20_49.json",
        "instance_n=20_490.json",
        "instance_n=20_491.json",
        "instance_n=20_492.json",
        "instance_n=20_493.json",
        "instance_n=20_494.json",
        "instance_n=20_495.json",
        "instance_n=20_496.json",
        "instance_n=20_497.json",
        "instance_n=20_498.json",
        "instance_n=20_499.json",
        "instance_n=20_5.json",
        "instance_n=20_50.json",
        "instance_n=20_500.json",
        "instance_n=20_501.json",
        "instance_n=20_502.json",
        "instance_n=20_503.json",
        "instance_n=20_504.json",
        "instance_n=20_505.json",
        "instance_n=20_506.json",
        "instance_n=20_507.json",
        "instance_n=20_508.json",
        "instance_n=20_509.json",
        "instance_n=20_51.json",
        "instance_n=20_510.json",
        "instance_n=20_511.json",
        "instance_n=20_512.json",
        "instance_n=20_513.json",
        "instance_n=20_514.json",
        "instance_n=20_515.json",
        "instance_n=20_516.json",
        "instance_n=20_517.json",
        "instance_n=20_518.json",
        "instance_n=20_519.json",
        "instance_n=20_52.json",
        "instance_n=20_520.json",
        "instance_n=20_521.json",
        "instance_n=20_522.json",
        "instance_n=20_523.json",
        "instance_n=20_524.json",
        "instance_n=20_525.json",
        "instance_n=20_53.json",
        "instance_n=20_54.json",
        "instance_n=20_55.json",
        "instance_n=20_56.json",
        "instance_n=20_57.json",
        "instance_n=20_58.json",
        "instance_n=20_59.json",
        "instance_n=20_6.json",
        "instance_n=20_60.json",
        "instance_n=20_61.json",
        "instance_n=20_62.json",
        "instance_n=20_63.json",
        "instance_n=20_64.json",
        "instance_n=20_65.json",
        "instance_n=20_66.json",
        "instance_n=20_67.json",
        "instance_n=20_68.json",
        "instance_n=20_69.json",
        "instance_n=20_7.json",
        "instance_n=20_70.json",
        "instance_n=20_71.json",
        "instance_n=20_72.json",
        "instance_n=20_73.json",
        "instance_n=20_74.json",
        "instance_n=20_75.json",
        "instance_n=20_76.json",
        "instance_n=20_77.json",
        "instance_n=20_78.json",
        "instance_n=20_79.json",
        "instance_n=20_8.json",
        "instance_n=20_80.json",
        "instance_n=20_81.json",
        "instance_n=20_82.json",
        "instance_n=20_83.json",
        "instance_n=20_84.json",
        "instance_n=20_85.json",
        "instance_n=20_86.json",
        "instance_n=20_87.json",
        "instance_n=20_88.json",
        "instance_n=20_89.json",
        "instance_n=20_9.json",
        "instance_n=20_90.json",
        "instance_n=20_91.json",
        "instance_n=20_92.json",
        "instance_n=20_93.json",
        "instance_n=20_94.json",
        "instance_n=20_95.json",
        "instance_n=20_96.json",
        "instance_n=20_97.json",
        "instance_n=20_98.json",
        "instance_n=20_99.json",
        "instance_n=50_1.json",
   ]
   
   with multiprocessing.Pool(5) as pool:
       pool.map(run, instance_names)
#    for instance_name in instance_names:
#        run(instance_name)
                #  , filepath)
            
    
    