import random
import datetime
import copy
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, List, Optional, Dict, Any

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, nearest_points
from scipy.stats.qmc import Halton

# --- [NEW] Added for efficiency ---
from scipy.spatial import cKDTree  # For fast neighbor search
from shapely.strtree import STRtree # For fast collision detection
# ----------------------------------

from dg_commons import PlayerName
from dg_commons.sim import InitSimGlobalObservations, InitSimObservations, SharedGoalObservation, SimObservations
from dg_commons.sim.agents import Agent, GlobalPlanner
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.diff_drive import DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons.sim.models.obstacles import StaticObstacle
from numpydantic import NDArray
from pydantic import BaseModel

class GlobalPlanMessage(BaseModel):
    # TODO: modify/add here the fields you need to send your global plan
    # fake_id: int
    # fake_name: str
    # fake_np_data: NDArray # If you need to send numpy arrays, annotate them with NDArray
    paths: Mapping[str, Sequence[Tuple[float, float]]]


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 10


# --- TASK ALLOCATION LOGIC ---

@dataclass
class DeliveryTask:
    goal_id: str
    collection_id: str

    def clone(self):
        """Fast shallow copy"""
        return DeliveryTask(self.goal_id, self.collection_id)

@dataclass
class RobotSchedule:
    robot_name: str
    tasks: List[DeliveryTask]

    def clone(self):
        """
        Fast deep copy. 
        Much faster than copy.deepcopy() because we know the structure.
        """
        new_sched = RobotSchedule(self.robot_name, [])
        # List comprehension is faster than loops
        new_sched.tasks = [t.clone() for t in self.tasks]
        return new_sched

class TaskAllocatorBase:
    """Shared Data and Evaluation Logic"""
    def __init__(self,
                 cost_matrix: Dict[str, Dict[str, float]],
                 heading_matrix: Dict[str, Dict[str, Tuple[float, float]]],
                 initial_headings: Dict[str, float],
                 w_max: float,
                 robots: List[str],
                 goals: List[str],
                 collections: List[str]):
        self.matrix = cost_matrix
        self.heading_matrix = heading_matrix
        self.initial_headings = initial_headings
        self.w_max = w_max if w_max > 0.01 else 0.1
        self.robots = robots
        self.goals = goals
        self.collections = collections

    def _evaluate_makespan(self, solution: Dict[str, RobotSchedule]) -> float:
        max_time = 0.0
        for r_name, schedule in solution.items():
            current_node = r_name 
            current_heading = self.initial_headings.get(r_name, 0.0)
            total_time = 0.0
            
            for task in schedule.tasks:
                # --- A. MOVE TO GOAL ---
                d1 = self.matrix.get(current_node, {}).get(task.goal_id, float('inf'))
                if d1 == float('inf'): return float('inf')
                
                angles_1 = self.heading_matrix.get(current_node, {}).get(task.goal_id, (0.0, 0.0))
                diff = (angles_1[0] - current_heading + math.pi) % (2 * math.pi) - math.pi
                total_time += (abs(diff) / self.w_max) + d1
                current_heading = angles_1[1]
                
                # --- B. MOVE TO COLLECTION ---
                d2 = self.matrix.get(task.goal_id, {}).get(task.collection_id, float('inf'))
                if d2 == float('inf'): return float('inf')
                
                angles_2 = self.heading_matrix.get(task.goal_id, {}).get(task.collection_id, (0.0, 0.0))
                diff = (angles_2[0] - current_heading + math.pi) % (2 * math.pi) - math.pi
                total_time += (abs(diff) / self.w_max) + d2
                current_heading = angles_2[1]
                
                current_node = task.collection_id
            
            if total_time > max_time:
                max_time = total_time
        return max_time

    def _calculate_schedule_duration(self, r_name, tasks) -> float:
        """Calculates total time for a specific task list including turns."""
        current_node = r_name
        current_heading = self.initial_headings.get(r_name, 0.0)
        total_time = 0.0
        
        for task in tasks:
            # 1. To Goal
            d1 = self.matrix.get(current_node, {}).get(task.goal_id, 0.0)
            angles_1 = self.heading_matrix.get(current_node, {}).get(task.goal_id, (0.0, 0.0))
            diff1 = (angles_1[0] - current_heading + math.pi) % (2 * math.pi) - math.pi
            total_time += (abs(diff1) / self.w_max) + d1
            current_heading = angles_1[1]
            
            # 2. To Collection
            d2 = self.matrix.get(task.goal_id, {}).get(task.collection_id, 0.0)
            angles_2 = self.heading_matrix.get(task.goal_id, {}).get(task.collection_id, (0.0, 0.0))
            diff2 = (angles_2[0] - current_heading + math.pi) % (2 * math.pi) - math.pi
            total_time += (abs(diff2) / self.w_max) + d2
            current_heading = angles_2[1]
            
            current_node = task.collection_id
            
        return total_time
    
class TaskAllocatorSA(TaskAllocatorBase):
    """Original Simulated Annealing Implementation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache best collections for greedy init (Static selection)
        self.best_collections = {} 
        for g in self.goals:
            best_c = None
            min_c_cost = float('inf')
            if g in self.matrix:
                for c in self.collections:
                    dist = self.matrix[g].get(c, float('inf'))
                    if dist < min_c_cost:
                        min_c_cost = dist
                        best_c = c
            self.best_collections[g] = best_c

    def solve(self, time_limit: float = 2.0) -> Dict[str, List[DeliveryTask]]:
        print(f"--- Running SA Allocator (Time Limit: {time_limit}s) ---")
        start_time = time.time()
        
        current_solution = self._generate_greedy_solution()
        current_cost = self._evaluate_makespan(current_solution)
        
        best_solution_global = {r: sched.clone() for r, sched in current_solution.items()}
        best_cost_global = current_cost
        
        temperature = 100.0
        cooling_rate = 0.95
        iterations = 0

        while (time.time() - start_time) < time_limit:
            iterations += 1
            neighbor_solution = {r: sched.clone() for r, sched in current_solution.items()}
            self._apply_random_mutation(neighbor_solution)
            neighbor_cost = self._evaluate_makespan(neighbor_solution)
            
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-5)):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                if current_cost < best_cost_global:
                    best_solution_global = {r: sched.clone() for r, sched in current_solution.items()}
                    best_cost_global = current_cost
            
            temperature *= cooling_rate
            if temperature < 0.5: temperature = 100.0 # Reheat

        print(f"SA Finished: Best Cost {best_cost_global:.2f} | Iterations: {iterations}")
        return {r: sched.tasks for r, sched in best_solution_global.items()}
    
    def _generate_greedy_solution(self) -> Dict[str, RobotSchedule]:
        """Assigns tasks to the robot that can finish it soonest, accounting for TURN COSTS."""
        schedules = {r: RobotSchedule(r, []) for r in self.robots}
        robot_completion_times = {r: 0.0 for r in self.robots}
        robot_locations = {r: r for r in self.robots} 
        
        robot_headings = {r: self.initial_headings.get(r, 0.0) for r in self.robots}

        unassigned = list(self.goals)
        
        while unassigned:
            best_r = None
            best_g = None
            best_c = None
            
            # We look for the assignment that results in the lowest TOTAL finish time
            min_finish_time = float('inf') 
            
            # Temp vars to update state after selection
            best_g_end_heading = 0.0
            best_c_end_heading = 0.0

            for g in unassigned:
                c = self.best_collections.get(g)
                if not c: continue
                
                for r in self.robots:
                    curr_loc = robot_locations[r]
                    curr_heading = robot_headings[r] # Get current heading
                    
                    # --- 1. Path: Current -> Goal ---
                    dist_g = self.matrix.get(curr_loc, {}).get(g, float('inf'))
                    if dist_g == float('inf'): continue

                    # Calculate Turn Cost (Transition to Path)
                    angles_g = self.heading_matrix.get(curr_loc, {}).get(g, (0.0, 0.0))
                    path_start_heading = angles_g[0]
                    
                    diff = path_start_heading - curr_heading
                    diff = (diff + math.pi) % (2 * math.pi) - math.pi
                    turn_cost_g = abs(diff) / self.w_max
                    
                    # --- 2. Path: Goal -> Collection ---
                    dist_c = self.matrix.get(g, {}).get(c, float('inf'))
                    if dist_c == float('inf'): continue

                    # Calculate Turn Cost (Transition at Goal)
                    # Robot arrives at goal with heading = angles_g[1]
                    path_end_heading_at_goal = angles_g[1]
                    
                    angles_c = self.heading_matrix.get(g, {}).get(c, (0.0, 0.0))
                    path_start_heading_coll = angles_c[0]

                    diff = path_start_heading_coll - path_end_heading_at_goal
                    diff = (diff + math.pi) % (2 * math.pi) - math.pi
                    turn_cost_c = abs(diff) / self.w_max

                    # --- Total Time Calculation ---
                    # Previous Time + Turn1 + Travel1 + Turn2 + Travel2
                    new_finish_time = robot_completion_times[r] + turn_cost_g + dist_g + turn_cost_c + dist_c
                    
                    if new_finish_time < min_finish_time:
                        min_finish_time = new_finish_time
                        best_r = r
                        best_g = g
                        best_c = c
                        # Store the final heading (arriving at collection)
                        best_c_end_heading = angles_c[1]

            if best_r:
                schedules[best_r].tasks.append(DeliveryTask(best_g, best_c))
                robot_completion_times[best_r] = min_finish_time
                robot_locations[best_r] = best_c 
                
                robot_headings[best_r] = best_c_end_heading
                
                unassigned.remove(best_g)
            else:
                break 
                
        return schedules
    
    def _apply_random_mutation(self, solution: Dict[str, RobotSchedule]):
        """Applies random swaps or dropoff changes."""
        r_names = list(solution.keys())
        r1_name = random.choice(r_names)
        sched1 = solution[r1_name]
        
        move_type = random.choice(['swap_owner', 'swap_order', 'change_dropoff'])
        
        if not sched1.tasks and move_type != 'swap_owner':
            move_type = 'swap_owner'

        if move_type == 'swap_owner':
            r2_name = random.choice(r_names)
            sched2 = solution[r2_name]
            
            if sched1.tasks or sched2.tasks:
                # Determine source and dest
                if sched1.tasks and sched2.tasks:
                    source, dest = (sched1, sched2) if random.random() < 0.5 else (sched2, sched1)
                elif sched1.tasks:
                    source, dest = sched1, sched2
                else:
                    source, dest = sched2, sched1
                
                # Pop and Insert
                task = source.tasks.pop(random.randint(0, len(source.tasks)-1))
                insert_idx = random.randint(0, len(dest.tasks)) # Can insert at end
                dest.tasks.insert(insert_idx, task)

        elif move_type == 'swap_order':
            if len(sched1.tasks) >= 2:
                idx1 = random.randint(0, len(sched1.tasks)-1)
                idx2 = random.randint(0, len(sched1.tasks)-1)
                sched1.tasks[idx1], sched1.tasks[idx2] = sched1.tasks[idx2], sched1.tasks[idx1]

        elif move_type == 'change_dropoff':
            if sched1.tasks:
                task = sched1.tasks[random.randint(0, len(sched1.tasks)-1)]
                # Try to find a valid random collection, not just any random string
                possible_cs = [c for c in self.collections if c != task.collection_id]
                if possible_cs:
                    new_c = random.choice(possible_cs)
                    # Only apply if reachable
                    if self.matrix.get(task.goal_id, {}).get(new_c, float('inf')) < float('inf'):
                        task.collection_id = new_c

class TaskAllocatorLNS(TaskAllocatorBase):
    """
    State-of-the-Art LNS with Post-Repair Smoothing.
    """
    def solve(self, time_limit: float = 2.0) -> Dict[str, List[DeliveryTask]]:
        print(f"--- Running LNS Allocator (Time Limit: {time_limit}s) ---")
        start_time = time.time()
        
        # 1. Start Empty
        current_sol = {r: RobotSchedule(r, []) for r in self.robots}
        
        # 2. Initial Construction
        all_tasks = [DeliveryTask(g, self.collections[0]) for g in self.goals]
        current_sol = self._repair_regret(current_sol, all_tasks)
        
        # CRITICAL FIX: Optimize dropoffs immediately after construction
        self._optimize_solution_dropoffs(current_sol)
        
        best_sol = {r: sched.clone() for r, sched in current_sol.items()}
        best_cost = self._evaluate_makespan(best_sol)
        
        iterations = 0
        n_remove = max(1, min(4, int(len(self.goals) * 0.4)))

        while (time.time() - start_time) < time_limit:
            iterations += 1
            
            # A. Clone
            temp_sol = {r: sched.clone() for r, sched in current_sol.items()}
            
            # B. Destroy
            if random.random() < 0.7:
                temp_sol, removed_tasks = self._destroy_random(temp_sol, n_remove)
            else:
                temp_sol, removed_tasks = self._destroy_worst(temp_sol, n_remove)
            
            # C. Repair (Regret Insertion)
            temp_sol = self._repair_regret(temp_sol, removed_tasks)
            
            # D. SMOOTHING PASS (The Missing Link)
            # Now that we have a full sequence, fix the blind spots.
            self._optimize_solution_dropoffs(temp_sol)
            
            # E. Evaluate
            new_cost = self._evaluate_makespan(temp_sol)
            
            # F. Acceptance
            if new_cost <= best_cost:
                best_cost = new_cost
                best_sol = {r: sched.clone() for r, sched in temp_sol.items()}
                current_sol = temp_sol
            elif random.random() < 0.05:
                current_sol = temp_sol

        print(f"LNS Finished: Best Cost {best_cost:.2f} | Iterations: {iterations}")
        return {r: sched.tasks for r, sched in best_sol.items()}

    def _optimize_solution_dropoffs(self, solution):
        """
        Iterates through the fixed sequence and updates Collection Points
        to minimize total turning + travel time based on the FULL path.
        """
        for r_name, sched in solution.items():
            if not sched.tasks: continue
            
            prev_loc = r_name
            prev_heading = self.initial_headings.get(r_name, 0.0)
            
            for i, task in enumerate(sched.tasks):
                # Identify Next Location
                if i + 1 < len(sched.tasks):
                    next_loc = sched.tasks[i+1].goal_id
                else:
                    next_loc = None # End of schedule
                    
                # Find Best C for: Prev -> Goal -> [C?] -> Next
                best_c = task.collection_id # Default to current
                min_cost = float('inf')
                
                # Pre-calc arrival at Goal
                d_pg = self.matrix.get(prev_loc, {}).get(task.goal_id, float('inf'))
                angles_pg = self.heading_matrix.get(prev_loc, {}).get(task.goal_id, (0.0, 0.0))
                
                # Turn to Goal
                diff_pg = (angles_pg[0] - prev_heading + math.pi) % (2 * math.pi) - math.pi
                cost_pg = (abs(diff_pg) / self.w_max) + d_pg
                heading_at_goal = angles_pg[1]
                
                # Iterate all Collections
                for c in self.collections:
                    # Goal -> C
                    d_gc = self.matrix.get(task.goal_id, {}).get(c, float('inf'))
                    if d_gc == float('inf'): continue
                    
                    angles_gc = self.heading_matrix.get(task.goal_id, {}).get(c, (0.0, 0.0))
                    diff_gc = (angles_gc[0] - heading_at_goal + math.pi) % (2 * math.pi) - math.pi
                    cost_gc = (abs(diff_gc) / self.w_max) + d_gc
                    heading_at_c = angles_gc[1]
                    
                    # C -> Next (if exists)
                    cost_cn = 0.0
                    if next_loc:
                        d_cn = self.matrix.get(c, {}).get(next_loc, float('inf'))
                        if d_cn == float('inf'): continue
                        angles_cn = self.heading_matrix.get(c, {}).get(next_loc, (0.0, 0.0))
                        diff_cn = (angles_cn[0] - heading_at_c + math.pi) % (2 * math.pi) - math.pi
                        cost_cn = (abs(diff_cn) / self.w_max) + d_cn
                    
                    total = cost_pg + cost_gc + cost_cn
                    
                    if total < min_cost:
                        min_cost = total
                        best_c = c
                
                # Update Task
                task.collection_id = best_c
                
                # Update pointers for next iteration
                prev_loc = best_c
                # Heading leaving C towards Next (or just sitting at C)
                # If next_loc exists, we leave C. If not, we stay.
                if next_loc:
                    prev_heading = self.heading_matrix.get(best_c, {}).get(next_loc, (0.0, 0.0))[1]
                else:
                    # Heading arriving at C
                    prev_heading = self.heading_matrix.get(task.goal_id, {}).get(best_c, (0.0, 0.0))[1]


    def _destroy_random(self, solution, n):
        all_tasks_flat = []
        for r, sched in solution.items():
            for t in sched.tasks: all_tasks_flat.append((r, t))
        if not all_tasks_flat: return solution, []
        to_remove = random.sample(all_tasks_flat, k=min(n, len(all_tasks_flat)))
        for r_name, task in to_remove:
            solution[r_name].tasks.remove(task)
        return solution, [t for r, t in to_remove]

    def _destroy_worst(self, solution, n):
        candidates = []
        for r_name, sched in solution.items():
            if not sched.tasks: continue
            prev_loc = r_name
            for i, task in enumerate(sched.tasks):
                next_loc = sched.tasks[i+1].goal_id if i+1 < len(sched.tasks) else None
                # Simple distance proxy for removal is usually sufficient and fast
                d1 = self.matrix.get(prev_loc, {}).get(task.goal_id, 0)
                d2 = self.matrix.get(task.goal_id, {}).get(task.collection_id, 0)
                d3 = self.matrix.get(task.collection_id, {}).get(next_loc, 0) if next_loc else 0
                shortcut = self.matrix.get(prev_loc, {}).get(next_loc, 0) if next_loc else 0
                savings = (d1 + d2 + d3) - shortcut
                candidates.append((savings, r_name, task))
                prev_loc = task.collection_id
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, r_name, task in candidates[:n]:
            if task in solution[r_name].tasks:
                solution[r_name].tasks.remove(task)
        return solution, [t for _, _, t in candidates[:n]]

    # def _repair_regret(self, solution, tasks_to_insert):
    #     # Tie-break randomization
    #     random.shuffle(tasks_to_insert)
        
    #     while tasks_to_insert:
    #         regrets = []
    #         for task in tasks_to_insert:
    #             options = []
    #             for r_name, sched in solution.items():
    #                 # Try every index
    #                 for i in range(len(sched.tasks) + 1):
    #                     cost_increase, best_c = self._calc_insertion_cost(sched, task, i)
    #                     options.append((cost_increase, r_name, i, best_c))
                
    #             options.sort(key=lambda x: x[0])
    #             if not options: continue
                
    #             best = options[0]
    #             second = options[1][0] if len(options) > 1 else float('inf')
    #             regret_val = second - best[0]
                
    #             regrets.append({'val': regret_val, 't': task, 'r': best[1], 'i': best[2], 'c': best[3]})
            
    #         if not regrets: break
    #         regrets.sort(key=lambda x: x['val'], reverse=True)
    #         winner = regrets[0]
            
    #         t = winner['t']
    #         t.collection_id = winner['c']
    #         solution[winner['r']].tasks.insert(winner['i'], t)
    #         tasks_to_insert.remove(t)
    #     return solution

    def _repair_regret(self, solution, tasks_to_insert):
        # Randomize order slightly to break ties
        random.shuffle(tasks_to_insert)
        
        # [NEW] Pre-calculate current duration for all robots to ensure Load Balancing
        robot_finish_times = {
            r: self._calculate_schedule_duration(r, sched.tasks) 
            for r, sched in solution.items()
        }

        while tasks_to_insert:
            regrets = []
            
            for task in tasks_to_insert:
                options = []
                
                for r_name, sched in solution.items():
                    current_r_time = robot_finish_times[r_name]
                    
                    # Try every index
                    for i in range(len(sched.tasks) + 1):
                        cost_increase, best_c = self._calc_insertion_cost(sched, task, i)
                        
                        # [CRITICAL FIX]
                        # The "Cost" is not just the added time, but the NEW FINISH TIME.
                        # This forces the algorithm to give tasks to idle robots.
                        new_finish_time = current_r_time + cost_increase
                        
                        options.append((new_finish_time, cost_increase, r_name, i, best_c))
                
                # [Old] Sort by LOWEST FINISH TIME (Min-Max Objective)
                # options.sort(key=lambda x: x[0])

                # [New] Weighted Objective (Balance Makespan vs Efficiency)
                # x[0] is new_finish_time (Makespan impact)
                # x[1] is cost_increase (Efficiency impact)
                alpha = 0.7 # Tuning parameter: 70% importance on Makespan, 30% on Efficiency
                options.sort(key=lambda x: (alpha * x[0]) + ((1 - alpha) * x[1]))
                
                if not options: continue
                
                best = options[0]
                second_best_time = options[1][0] if len(options) > 1 else float('inf')
                
                # REGRET: How much does the MAKESPAN increase if we don't pick the best?
                regret_val = second_best_time - best[0]
                
                regrets.append({
                    'regret': regret_val,
                    'task': task,
                    'robot': best[2],
                    'idx': best[3],
                    'coll': best[4],
                    'cost_increase': best[1] # Keep track to update cache
                })
            
            # Sort tasks by Regret (Highest first)
            if not regrets: break
            regrets.sort(key=lambda x: x['regret'], reverse=True)
            
            winner = regrets[0]
            
            # Execute Insertion
            t = winner['task']
            t.collection_id = winner['coll']
            r_name = winner['robot']
            
            solution[r_name].tasks.insert(winner['idx'], t)
            
            # [NEW] Update the cache for the specific robot
            robot_finish_times[r_name] += winner['cost_increase']
            
            tasks_to_insert.remove(t)
            
        return solution

    def _calc_insertion_cost(self, schedule, task, idx):
        # 1. Identify Context
        if idx == 0:
            prev_loc = schedule.robot_name
            prev_heading = self.initial_headings.get(prev_loc, 0.0)
        else:
            prev_task = schedule.tasks[idx-1]
            prev_loc = prev_task.collection_id
            # Estimate heading arriving at Prev Collection
            pg = prev_task.goal_id
            pc = prev_task.collection_id
            prev_heading = self.heading_matrix.get(pg, {}).get(pc, (0.0, 0.0))[1]

        next_loc = schedule.tasks[idx].goal_id if idx < len(schedule.tasks) else None
        
        # 2. Base Cost: Prev -> Goal (Fixed)
        d_pg = self.matrix.get(prev_loc, {}).get(task.goal_id, float('inf'))
        if d_pg == float('inf'): return float('inf'), None
        
        angles_pg = self.heading_matrix.get(prev_loc, {}).get(task.goal_id, (0.0, 0.0))
        turn_pg = abs((angles_pg[0] - prev_heading + math.pi) % (2*math.pi) - math.pi) / self.w_max
        cost_pg = turn_pg + d_pg
        heading_at_goal = angles_pg[1]
        
        # 3. Optimize C
        best_c = None
        min_seg = float('inf')
        
        for c in self.collections:
            # Goal -> C
            d_gc = self.matrix.get(task.goal_id, {}).get(c, float('inf'))
            if d_gc == float('inf'): continue
            
            angles_gc = self.heading_matrix.get(task.goal_id, {}).get(c, (0.0, 0.0))
            turn_gc = abs((angles_gc[0] - heading_at_goal + math.pi) % (2*math.pi) - math.pi) / self.w_max
            cost_gc = turn_gc + d_gc
            heading_at_c = angles_gc[1]
            
            # C -> Next (Approximate if next exists)
            cost_cn = 0.0
            if next_loc:
                d_cn = self.matrix.get(c, {}).get(next_loc, float('inf'))
                if d_cn == float('inf'): continue
                angles_cn = self.heading_matrix.get(c, {}).get(next_loc, (0.0, 0.0))
                turn_cn = abs((angles_cn[0] - heading_at_c + math.pi) % (2*math.pi) - math.pi) / self.w_max
                cost_cn = turn_cn + d_cn
            
            if (cost_gc + cost_cn) < min_seg:
                min_seg = cost_gc + cost_cn
                best_c = c
                
        if best_c is None: return float('inf'), None
        
        # 4. Marginal Delta
        added = cost_pg + min_seg
        removed = 0.0
        if next_loc:
            d_pn = self.matrix.get(prev_loc, {}).get(next_loc, float('inf'))
            if d_pn < float('inf'):
                angles_pn = self.heading_matrix.get(prev_loc, {}).get(next_loc, (0.0, 0.0))
                turn_pn = abs((angles_pn[0] - prev_heading + math.pi) % (2*math.pi) - math.pi) / self.w_max
                removed = turn_pn + d_pn
                
        return added - removed, best_c


# class TaskAllocator:
#     """
#     Solves the Multi-Depot Vehicle Routing Problem.
#     Uses Simulated Annealing with Reheating to prevent premature convergence.
#     """
#     def __init__(self,
#                  cost_matrix: Dict[str, Dict[str, float]],
#                  heading_matrix: Dict[str, Dict[str, Tuple[float, float]]],
#                  initial_headings: Dict[str, float],
#                  w_max: float,
#                  robots: List[str],
#                  goals: List[str],
#                  collections: List[str]):
#         self.matrix = cost_matrix
#         self.heading_matrix = heading_matrix
#         self.initial_headings = initial_headings
#         self.w_max = w_max if w_max > 0.01 else 0.1
#         self.robots = robots
#         self.goals = goals
#         self.collections = collections
        
#         # Cache best collections for greedy init
#         self.best_collections = {} 
#         for g in self.goals:
#             best_c = None
#             min_c_cost = float('inf')
#             if g in self.matrix:
#                 for c in self.collections:
#                     dist = self.matrix[g].get(c, float('inf'))
#                     if dist < min_c_cost:
#                         min_c_cost = dist
#                         best_c = c
#             self.best_collections[g] = best_c

#     def solve(self, time_limit: float = 2.0) -> Dict[str, List[DeliveryTask]]:
#         """Runs the Simulated Annealing optimization with Reheating."""
#         start_time = time.time()
        
#         # 1. Initial Solution (Greedy)
#         current_solution = self._generate_greedy_solution()
#         current_cost = self._evaluate_makespan(current_solution)
        
#         # Keep track of the absolute best found across all "reheats"
#         best_solution_global = {r: sched.clone() for r, sched in current_solution.items()}
#         best_cost_global = current_cost
        
#         # Annealing Parameters
#         initial_temp = 100.0
#         temperature = initial_temp
#         cooling_rate = 0.95  # Slightly slower cooling
#         min_temp = 0.5       # Threshold to trigger reheat
        
#         iterations = 0

#         while (time.time() - start_time) < time_limit:
#             iterations += 1
            
#             # 2. Create Neighbor (Fast Clone)
#             # We copy the dict structure, but we only need to deep clone the schedules 
#             # that we are about to modify. However, for simplicity/safety, we clone all.
#             neighbor_solution = {r: sched.clone() for r, sched in current_solution.items()}
            
#             # 3. Mutate
#             self._apply_random_mutation(neighbor_solution)
            
#             # 4. Evaluate
#             neighbor_cost = self._evaluate_makespan(neighbor_solution)
            
#             # 5. Acceptance Probability
#             delta = neighbor_cost - current_cost
            
#             # If better, or lucky roll
#             if delta < 0 or random.random() < math.exp(-delta / temperature):
#                 current_solution = neighbor_solution
#                 current_cost = neighbor_cost
                
#                 # Update Global Best
#                 if current_cost < best_cost_global:
#                     best_solution_global = {r: sched.clone() for r, sched in current_solution.items()}
#                     best_cost_global = current_cost
#                     # Optional: Print improvement
#                     # print(f"New Best: {best_cost_global:.2f} (Iter {iterations})")
            
#             # 6. Cooling & Reheating
#             temperature *= cooling_rate
#             if temperature < min_temp:
#                 temperature = initial_temp # REHEAT!
#                 # Optional: Reset search to the best known location?
#                 # current_solution = {r: sched.clone() for r, sched in best_solution_global.items()}
#                 # current_cost = best_cost_global
            
#         print(f"Allocator: Best Makespan: {best_cost_global:.2f}s | Iterations: {iterations}")
#         return {r: sched.tasks for r, sched in best_solution_global.items()}

#     def _generate_greedy_solution(self) -> Dict[str, RobotSchedule]:
#         """Assigns tasks to the robot that can finish it soonest, accounting for TURN COSTS."""
#         schedules = {r: RobotSchedule(r, []) for r in self.robots}
#         robot_completion_times = {r: 0.0 for r in self.robots}
#         robot_locations = {r: r for r in self.robots} 
        
#         robot_headings = {r: self.initial_headings.get(r, 0.0) for r in self.robots}

#         unassigned = list(self.goals)
        
#         while unassigned:
#             best_r = None
#             best_g = None
#             best_c = None
            
#             # We look for the assignment that results in the lowest TOTAL finish time
#             min_finish_time = float('inf') 
            
#             # Temp vars to update state after selection
#             best_g_end_heading = 0.0
#             best_c_end_heading = 0.0

#             for g in unassigned:
#                 c = self.best_collections.get(g)
#                 if not c: continue
                
#                 for r in self.robots:
#                     curr_loc = robot_locations[r]
#                     curr_heading = robot_headings[r] # Get current heading
                    
#                     # --- 1. Path: Current -> Goal ---
#                     dist_g = self.matrix.get(curr_loc, {}).get(g, float('inf'))
#                     if dist_g == float('inf'): continue

#                     # Calculate Turn Cost (Transition to Path)
#                     angles_g = self.heading_matrix.get(curr_loc, {}).get(g, (0.0, 0.0))
#                     path_start_heading = angles_g[0]
                    
#                     diff = path_start_heading - curr_heading
#                     diff = (diff + math.pi) % (2 * math.pi) - math.pi
#                     turn_cost_g = abs(diff) / self.w_max
                    
#                     # --- 2. Path: Goal -> Collection ---
#                     dist_c = self.matrix.get(g, {}).get(c, float('inf'))
#                     if dist_c == float('inf'): continue

#                     # Calculate Turn Cost (Transition at Goal)
#                     # Robot arrives at goal with heading = angles_g[1]
#                     path_end_heading_at_goal = angles_g[1]
                    
#                     angles_c = self.heading_matrix.get(g, {}).get(c, (0.0, 0.0))
#                     path_start_heading_coll = angles_c[0]

#                     diff = path_start_heading_coll - path_end_heading_at_goal
#                     diff = (diff + math.pi) % (2 * math.pi) - math.pi
#                     turn_cost_c = abs(diff) / self.w_max

#                     # --- Total Time Calculation ---
#                     # Previous Time + Turn1 + Travel1 + Turn2 + Travel2
#                     new_finish_time = robot_completion_times[r] + turn_cost_g + dist_g + turn_cost_c + dist_c
                    
#                     if new_finish_time < min_finish_time:
#                         min_finish_time = new_finish_time
#                         best_r = r
#                         best_g = g
#                         best_c = c
#                         # Store the final heading (arriving at collection)
#                         best_c_end_heading = angles_c[1]

#             if best_r:
#                 schedules[best_r].tasks.append(DeliveryTask(best_g, best_c))
#                 robot_completion_times[best_r] = min_finish_time
#                 robot_locations[best_r] = best_c 
                
#                 robot_headings[best_r] = best_c_end_heading
                
#                 unassigned.remove(best_g)
#             else:
#                 break 
                
#         return schedules

#     def _evaluate_makespan(self, solution: Dict[str, RobotSchedule]) -> float:
#         max_time = 0.0
        
#         for r_name, schedule in solution.items():
#             current_node = r_name 
            
#             # 1. Get Robot's starting orientation
#             current_heading = self.initial_headings.get(r_name, 0.0)
            
#             total_time = 0.0
            
#             for task in schedule.tasks:
#                 # --- A. MOVE TO GOAL ---
#                 d1 = self.matrix.get(current_node, {}).get(task.goal_id, float('inf'))
#                 if d1 == float('inf'): return float('inf')
                
#                 # Get angles for this segment (Start, End)
#                 angles_1 = self.heading_matrix.get(current_node, {}).get(task.goal_id, (0.0, 0.0))
#                 path_start_heading = angles_1[0]
#                 path_end_heading = angles_1[1]

#                 # CALC TURN COST (Transition)
#                 diff = path_start_heading - current_heading
#                 diff = (diff + math.pi) % (2 * math.pi) - math.pi # Normalize
#                 turn_cost = abs(diff) / self.w_max
                
#                 total_time += turn_cost + d1
#                 current_heading = path_end_heading # Update robot heading
                
#                 # --- B. MOVE TO COLLECTION ---
#                 d2 = self.matrix.get(task.goal_id, {}).get(task.collection_id, float('inf'))
#                 if d2 == float('inf'): return float('inf')
                
#                 angles_2 = self.heading_matrix.get(task.goal_id, {}).get(task.collection_id, (0.0, 0.0))
#                 path_start_heading = angles_2[0]
#                 path_end_heading = angles_2[1]

#                 # CALC TURN COST (Transition at Goal)
#                 diff = path_start_heading - current_heading
#                 diff = (diff + math.pi) % (2 * math.pi) - math.pi
#                 turn_cost = abs(diff) / self.w_max
                
#                 total_time += turn_cost + d2
#                 current_heading = path_end_heading # Update robot heading
                
#                 current_node = task.collection_id
            
#             if total_time > max_time:
#                 max_time = total_time
                
#         return max_time

    # def _apply_random_mutation(self, solution: Dict[str, RobotSchedule]):
    #     """Applies random swaps or dropoff changes."""
    #     r_names = list(solution.keys())
    #     r1_name = random.choice(r_names)
    #     sched1 = solution[r1_name]
        
    #     move_type = random.choice(['swap_owner', 'swap_order', 'change_dropoff'])
        
    #     if not sched1.tasks and move_type != 'swap_owner':
    #         move_type = 'swap_owner'

    #     if move_type == 'swap_owner':
    #         r2_name = random.choice(r_names)
    #         sched2 = solution[r2_name]
            
    #         if sched1.tasks or sched2.tasks:
    #             # Determine source and dest
    #             if sched1.tasks and sched2.tasks:
    #                 source, dest = (sched1, sched2) if random.random() < 0.5 else (sched2, sched1)
    #             elif sched1.tasks:
    #                 source, dest = sched1, sched2
    #             else:
    #                 source, dest = sched2, sched1
                
    #             # Pop and Insert
    #             task = source.tasks.pop(random.randint(0, len(source.tasks)-1))
    #             insert_idx = random.randint(0, len(dest.tasks)) # Can insert at end
    #             dest.tasks.insert(insert_idx, task)

    #     elif move_type == 'swap_order':
    #         if len(sched1.tasks) >= 2:
    #             idx1 = random.randint(0, len(sched1.tasks)-1)
    #             idx2 = random.randint(0, len(sched1.tasks)-1)
    #             sched1.tasks[idx1], sched1.tasks[idx2] = sched1.tasks[idx2], sched1.tasks[idx1]

    #     elif move_type == 'change_dropoff':
    #         if sched1.tasks:
    #             task = sched1.tasks[random.randint(0, len(sched1.tasks)-1)]
    #             # Try to find a valid random collection, not just any random string
    #             possible_cs = [c for c in self.collections if c != task.collection_id]
    #             if possible_cs:
    #                 new_c = random.choice(possible_cs)
    #                 # Only apply if reachable
    #                 if self.matrix.get(task.goal_id, {}).get(new_c, float('inf')) < float('inf'):
    #                     task.collection_id = new_c


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: DiffDriveGeometry
    sp: DiffDriveParameters

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        # self.my_global_path: List[Tuple[float, float]] = []

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        pass

    def on_receive_global_plan(
        self,
        serialized_msg: str,
    ):
        # TODO: process here the received global plan
        global_plan = GlobalPlanMessage.model_validate_json(serialized_msg)
        # if self.name in global_plan.paths:
        #     self.my_global_path = list(global_plan.paths[self.name])
        # else:
        #     self.my_global_path = []

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: DiffDriveState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """

        # TODO: implement here your planning stack
        omega1 = random.random() * self.params.param1
        omega2 = random.random() * self.params.param1

        return DiffDriveCommands(omega_l=omega1, omega_r=omega2)


class Pdm4arGlobalPlanner(GlobalPlanner):
    """
    This is the Global Planner for PDM4AR
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task
    """

    def __init__(self):
        # Parameters
        self.num_samples = 2000         # Can handle more samples now
        self.target_degree = 20         # We WANT this many connections per node
        self.max_candidates = 50        # We CHECK this many to find the valid ones (handles deleted vertices)
        self.robot_radius = 0.6 + 0.1   # Buffer size (robot width/2 + margin)
        self.connection_radius = 10.0   # Max length of an edge
        self.min_sample_dist = 0.3      # Minimum distance between nodes
        self.turn_penalty = 0.0         # Heuristic cost for "stopping and turning" (meters equivalent)

        self.time_limit = 20.0          # Time limit for task allocation

        self.seed = 42

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
        random.seed(self.seed)
        np.random.seed(self.seed)

        # --- 1. EXTRACT OBSTACLES & BOUNDS (Done once) ---
        obs_polygons = []
        if init_sim_obs.dg_scenario and init_sim_obs.dg_scenario.static_obstacles:
            for obs in init_sim_obs.dg_scenario.static_obstacles:
                if hasattr(obs, 'shape'):
                    obs_polygons.append(obs.shape)
        
        # Buffer obstacles
        inflated_obstacles = [o.buffer(self.robot_radius) for o in obs_polygons]

        # Calculate Bounds ONCE (Used for Sampling and Plotting)
        if obs_polygons:
            raw_combined = unary_union(obs_polygons)
            bounds = raw_combined.bounds 
        else:
            bounds = (-12.0, -12.0, 12.0, 12.0)
        
        # --- 2. PREPARE NODES (Iterate once) ---
        special_nodes_plot = {"starts": [], "goals": [], "collections": []}
        initial_nodes_data = [] # List of (x, y, type, label)

        # Starts
        robots_list = []
        for name, state in init_sim_obs.initial_states.items():
            special_nodes_plot["starts"].append((state.x, state.y))
            initial_nodes_data.append((state.x, state.y, "start", name))
            robots_list.append(name)

        # Shared Goals
        goals_list = []
        if init_sim_obs.shared_goals:
            for gid, sgoal in init_sim_obs.shared_goals.items():
                if hasattr(sgoal, 'polygon'):
                    c = sgoal.polygon.centroid
                    special_nodes_plot["goals"].append((c.x, c.y))
                    initial_nodes_data.append((c.x, c.y, "goal", gid))
                    goals_list.append(gid)

        # Collection Points
        collections_list = []
        if init_sim_obs.collection_points:
            for cid, cpoint in init_sim_obs.collection_points.items():
                if hasattr(cpoint, 'polygon'):
                    c = cpoint.polygon.centroid
                    special_nodes_plot["collections"].append((c.x, c.y))
                    initial_nodes_data.append((c.x, c.y, "collection", cid))
                    collections_list.append(cid)

        # --- 3. BUILD PRM ---
        G = self._build_prm(inflated_obstacles, initial_nodes_data, bounds)

        # 1. Get Initial Headings
        initial_headings = {}
        for name, state in init_sim_obs.initial_states.items():
            initial_headings[name] = state.psi

        # 2. Get Kinematics for Allocator
        sg = DiffDriveGeometry.default()
        sp = DiffDriveParameters.default()
        _, w_max = self._get_kinematic_limits(sg, sp)

        # --- 4. COMPUTE ROUTING DATA (COSTS & PATHS) ---
        cost_matrix, path_data, heading_matrix = self._compute_routing_data(G)
        print(f"Computed Cost Matrix for {len(cost_matrix)} nodes.")

        # # 4. Initialize Allocator with new args
        # allocator = TaskAllocator(
        #     cost_matrix=cost_matrix,
        #     heading_matrix=heading_matrix,
        #     initial_headings=initial_headings,
        #     w_max=w_max,
        #     robots=robots_list,
        #     goals=goals_list,
        #     collections=collections_list
        # )
        # assignments = allocator.solve(self.time_limit)

        # 4. Initialize Allocator ARGS
        alloc_args = {
            'cost_matrix': cost_matrix,
            'heading_matrix': heading_matrix,
            'initial_headings': initial_headings,
            'w_max': w_max,
            'robots': robots_list,
            'goals': goals_list,
            'collections': collections_list
        }

        # 5. Run BOTH Algorithms (Comparison Mode)
        
        # A. Run SA
        allocator_sa = TaskAllocatorSA(**alloc_args)
        # Give SA 30% of time or a fixed amount
        sa_assignments = allocator_sa.solve(time_limit=self.time_limit * 0.5) 
        sa_cost = allocator_sa._evaluate_makespan({r: RobotSchedule(r, t) for r, t in sa_assignments.items()})

        # B. Run LNS
        allocator_lns = TaskAllocatorLNS(**alloc_args)
        # Give LNS more time as it's the primary target
        lns_assignments = allocator_lns.solve(time_limit=self.time_limit * 0.5)
        lns_cost = allocator_lns._evaluate_makespan({r: RobotSchedule(r, t) for r, t in lns_assignments.items()})

        print(f"--- RESULT COMPARISON ---")
        print(f"SA Cost:  {sa_cost:.2f}")
        print(f"LNS Cost: {lns_cost:.2f}")

        # --- NEW: Call Debug Printer ---
        self._print_debug_comparison(
            sa_assignments, 
            lns_assignments, 
            cost_matrix, 
            heading_matrix
        )
        # -------------------------------

        # Pick the winner
        if lns_cost <= sa_cost:
            print(">> Using LNS Plan")
            assignments = lns_assignments
        else:
            print(">> Using SA Plan")
            assignments = sa_assignments
        
        # --- 6. CONSTRUCT FINAL PATHS ---
        final_planned_paths = {}
        for r_name, tasks in assignments.items():
            full_coords = []
            current_node = r_name 
            
            for task in tasks:
                # 1. Path: Current -> Goal
                seg1 = self._find_path_coords(path_data, current_node, task.goal_id)
                # 2. Path: Goal -> Collection
                seg2 = self._find_path_coords(path_data, task.goal_id, task.collection_id)
                
                if seg1: full_coords.extend(seg1)
                if seg2: full_coords.extend(seg2)
                
                current_node = task.collection_id 
            
            final_planned_paths[r_name] = full_coords

        # --- 7. DEBUG PLOT ---
        out_dir = Path("out/ex14/debug_plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = out_dir / f"prm_debug_{timestamp}.png"
        
        self._plot_prm(G, obs_polygons, special_nodes_plot, str(filename), bounds, path_data, final_planned_paths)

        # --- 6. RETURN EMPTY PLANS ---
        planned_paths = {p: [] for p in init_sim_obs.players_obs.keys()}
        global_plan_message = GlobalPlanMessage(
            # paths=final_planned_paths
            paths=planned_paths
        )
        return global_plan_message.model_dump_json(round_trip=True)

    def _find_path_coords(self, path_data, src, dst):
        """Helper to find path coordinates from any bucket"""
        for cat in path_data.values():
            if src in cat and dst in cat[src]:
                return cat[src][dst]['coords']
        return []


    def _compute_routing_data(self, G) -> Tuple[dict, dict]:
        """
        Computes APSP for POIs and extracts ALL paths.
        Returns:
            1. cost_matrix: {SourceLabel: {TargetLabel: Cost}}
            2. path_data: { "starts": ..., "goals": ..., "collections": ... }
        """
        cost_matrix = {}
        heading_matrix = {}
        path_data = {"starts": {}, "goals": {}, "collections": {}}
        pos = nx.get_node_attributes(G, 'pos')

        # --- Initialize Robot Model for Cost Calculation ---
        sg_default = DiffDriveGeometry.default()
        sp_default = DiffDriveParameters.default()
        
        # --- [DEBUG PRINT HERE] ---
        v_debug, w_debug = self._get_kinematic_limits(sg_default, sp_default)
        print(f"DEBUG KINEMATICS: V_max = {v_debug:.3f} m/s | Omega_max = {w_debug:.3f} rad/s")
        # --------------------------
        
        starts = [(n, d.get('label')) for n, d in G.nodes(data=True) if d.get('type') == 'start']
        goals = [(n, d.get('label')) for n, d in G.nodes(data=True) if d.get('type') == 'goal']
        collections = [(n, d.get('label')) for n, d in G.nodes(data=True) if d.get('type') == 'collection']

        def process_group(source_list, target_list, category_key):
            for src_idx, src_label in source_list:
                if src_label not in cost_matrix: cost_matrix[src_label] = {}
                if src_label not in heading_matrix: heading_matrix[src_label] = {}
                if src_label not in path_data[category_key]: path_data[category_key][src_label] = {}
                
                # Temp storage to find best target based on TIME, not DISTANCE
                candidates = []

                for tgt_idx, tgt_label in target_list:
                    try:
                        # 1. Get Shortest Path by DISTANCE (Geometric Path)
                        path_nodes = nx.shortest_path(G, src_idx, tgt_idx, weight='weight')
                        coords = [pos[n] for n in path_nodes]
                        
                        # 2. Calculate Cost by DURATION (Time)
                        duration = self._calculate_path_duration(coords)

                        # --- NEW: CALCULATE HEADINGS ---
                        s_angle = 0.0
                        e_angle = 0.0
                        if len(coords) >= 2:
                            # Heading of the first segment
                            s_angle = math.atan2(coords[1][1] - coords[0][1], coords[1][0] - coords[0][0])
                            # Heading of the last segment
                            e_angle = math.atan2(coords[-1][1] - coords[-2][1], coords[-1][0] - coords[-2][0])
                        # -------------------------------
                        
                        candidates.append((duration, tgt_label, coords))
                        
                        # Store in matrix
                        cost_matrix[src_label][tgt_label] = duration
                        heading_matrix[src_label][tgt_label] = (s_angle, e_angle)
                        
                    except nx.NetworkXNoPath:
                        cost_matrix[src_label][tgt_label] = float('inf')
                        heading_matrix[src_label][tgt_label] = (0.0, 0.0)

                # 3. Find which target was the "best" (fastest) and mark it
                if candidates:
                    candidates.sort(key=lambda x: x[0]) # Sort by duration
                    best_label = candidates[0][1]
                    
                    for cost, label, coords in candidates:
                        path_data[category_key][src_label][label] = {
                            'coords': coords,
                            'is_best': (label == best_label)
                        }

        # 2. Compute Robot -> Goals
        process_group(starts, goals, "starts")
        # 3. Compute Goal -> Collections
        process_group(goals, collections, "goals")
        # 4. Compute Collection -> Goals (For multi-step missions)
        process_group(collections, goals, "collections")

        return cost_matrix, path_data, heading_matrix

    def _get_kinematic_limits(self, sg: DiffDriveGeometry, sp: DiffDriveParameters) -> Tuple[float, float]:
        """Derives v_max [m/s] and omega_max [rad/s] from robot structures."""
        # Max wheel rotation (rad/s)
        w_wheel_max = max(abs(sp.omega_limits[0]), abs(sp.omega_limits[1]))
        
        # V_max = r * omega_wheel
        v_max = sg.wheelradius * w_wheel_max
        
        # Omega_max = (2 * r * omega_wheel) / L
        omega_max = (2 * sg.wheelradius * w_wheel_max) / sg.wheelbase
        return v_max, omega_max

    def _calculate_path_duration(self, coords: List[Tuple[float, float]]) -> float:
        """Calculates accurate duration using robot kinematics."""
        if not coords or len(coords) < 2:
            return 0.0

        # Use defaults since we are in GlobalPlanner (or pass specific ones if you have them)
        sg = DiffDriveGeometry.default()
        sp = DiffDriveParameters.default()
        
        v_max, w_max = self._get_kinematic_limits(sg, sp)
        
        # Safety clamp
        if v_max < 1e-4: v_max = 0.1
        if w_max < 1e-4: w_max = 0.1

        total_time = 0.0
        current_heading = None

        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i+1]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.sqrt(dx**2 + dy**2)
            
            # 1. Linear Time
            total_time += (dist / v_max)

            # 2. Angular Time
            target_heading = math.atan2(dy, dx)
            if current_heading is not None:
                angle_diff = target_heading - current_heading
                # Normalize to [-pi, pi]
                angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                total_time += abs(angle_diff) / w_max
            
            current_heading = target_heading

        return total_time

    def _build_prm(self, inflated_obstacles, initial_nodes_data, bounds) -> nx.Graph:
        """
        Pure logic method. 
        Changes: Removed obs_polygons arg, used passed-in bounds.
        """
        G = nx.Graph()
        node_coords = [] 
        node_indices = []
        occupied_grids = set()

        # Spatial Acceleration
        obstacle_tree = STRtree(inflated_obstacles)
        
        # Boundary for OB-PRM
        combined_obstacles = unary_union(inflated_obstacles)
        boundary_geom = combined_obstacles.boundary

        # --- A. ADD INITIAL NODES ---
        for (x, y, n_type, label) in initial_nodes_data:
            idx = len(G.nodes)
            G.add_node(idx, pos=(x, y), type=n_type, label=label)
            node_coords.append([x, y])
            node_indices.append(idx)
            gx, gy = int(x / self.min_sample_dist), int(y / self.min_sample_dist)
            occupied_grids.add((gx, gy))

        # --- B. SAMPLE REMAINING NODES ---
        min_x, min_y, max_x, max_y = bounds
        width, height = max_x - min_x, max_y - min_y
        
        sampler = Halton(d=2, scramble=True, seed=self.seed)
        raw_samples = sampler.random(n=self.num_samples * 3) 
        samples_x = raw_samples[:, 0] * width + min_x
        samples_y = raw_samples[:, 1] * height + min_y
        
        count = 0

        for x, y in zip(samples_x, samples_y):
            if count >= self.num_samples: break

            # Grid Check
            gx = int(x / self.min_sample_dist)
            gy = int(y / self.min_sample_dist)
            if (gx, gy) in occupied_grids: continue
                
            p = Point(x, y)
            
            # Collision Check
            possible_obs_indices = obstacle_tree.query(p)
            is_valid = True
            for obs_idx in possible_obs_indices:
                if inflated_obstacles[obs_idx].contains(p):
                    is_valid = False
                    break
            
            final_point = None
            if is_valid:
                final_point = p
            else:
                # OB-PRM Projection
                try:
                    nearest = nearest_points(p, boundary_geom)[1]
                    dx, dy = nearest.x - p.x, nearest.y - p.y
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 1e-6:
                        nudge = 0.1 
                        final_point = Point(nearest.x + (dx/dist)*nudge, nearest.y + (dy/dist)*nudge)
                    else:
                        final_point = nearest
                except Exception:
                    continue

            if final_point:
                fgx = int(final_point.x / self.min_sample_dist)
                fgy = int(final_point.y / self.min_sample_dist)
                if (fgx, fgy) in occupied_grids: continue 
                
                occupied_grids.add((fgx, fgy))
                
                idx = len(G.nodes)
                G.add_node(idx, pos=(final_point.x, final_point.y), type="sample")
                node_coords.append([final_point.x, final_point.y])
                node_indices.append(idx)
                count += 1

        # --- C. CONNECT NODES ---
        if len(node_coords) > 1:
            data_np = np.array(node_coords)
            tree = cKDTree(data_np)
            dists_all, indices_all = tree.query(data_np, k=self.max_candidates)
            
            for i, (nbr_dists, nbr_indices) in enumerate(zip(dists_all, indices_all)):
                u = node_indices[i]
                u_pos = Point(node_coords[i])
                edges_added = 0
                
                for d, j_idx in zip(nbr_dists, nbr_indices):
                    if i == j_idx: continue
                    if edges_added >= self.target_degree: break
                    if d > self.connection_radius: break
                    
                    v = node_indices[j_idx]
                    if G.has_edge(u, v): 
                        edges_added += 1
                        continue
                    
                    v_pos = Point(node_coords[j_idx])
                    line = LineString([u_pos, v_pos])
                    
                    candidates_idx = obstacle_tree.query(line)
                    is_colliding = False
                    for idx in candidates_idx:
                        if inflated_obstacles[idx].intersects(line):
                            is_colliding = True
                            break
                            
                    if not is_colliding:
                        # Apply Turn Penalty Heuristic here
                        G.add_edge(u, v, weight=d + self.turn_penalty)
                        edges_added += 1
        
        return G

    def _plot_prm(self, G, obstacles, special_nodes, filename, bounds=None, path_data=None, final_paths=None):
        plt.figure(figsize=(12, 12))
        
        # --- 1. Plot Buffered Obstacles (Inflated Boundaries) ---
        added_buffer_label = False
        for poly in obstacles:
            buffered = poly.buffer(self.robot_radius)
            def plot_poly_outline(geom, label=None):
                x, y = geom.exterior.xy
                plt.plot(x, y, 'k--', linewidth=1, alpha=0.5, label=label)
                for interior in geom.interiors:
                    x, y = interior.xy
                    plt.plot(x, y, 'k--', linewidth=1, alpha=0.5)

            if buffered.geom_type == 'Polygon':
                label = "Buffered (C-Space)" if not added_buffer_label else None
                plot_poly_outline(buffered, label)
                if label: added_buffer_label = True
            elif buffered.geom_type == 'MultiPolygon':
                for i, geom in enumerate(buffered.geoms):
                    label = "Buffered (C-Space)" if (not added_buffer_label and i == 0) else None
                    plot_poly_outline(geom, label)
                    if label: added_buffer_label = True

        # --- 2. Plot Real Obstacles ---
        added_obs_label = False
        for poly in obstacles:
            def fill_poly(geom, label=None):
                x, y = geom.exterior.xy
                plt.fill(x, y, color='gray', alpha=0.5, label=label)

            if poly.geom_type == 'Polygon':
                label = "Static Obstacle" if not added_obs_label else None
                fill_poly(poly, label)
                if label: added_obs_label = True
            elif poly.geom_type == 'MultiPolygon':
                for i, geom in enumerate(poly.geoms):
                    label = "Static Obstacle" if (not added_obs_label and i == 0) else None
                    fill_poly(geom, label)
                    if label: added_obs_label = True

        # --- 3. Plot Edges ---
        pos = nx.get_node_attributes(G, 'pos')
        if pos:
            lines = [[pos[u], pos[v]] for u, v in G.edges()]
            from matplotlib.collections import LineCollection
            lc = LineCollection(lines, colors='green', linewidths=0.5, alpha=0.2)
            plt.gca().add_collection(lc)
            plt.plot([], [], color='green', linewidth=0.5, label='PRM Edges')

            # --- 4. Plot Nodes (Samples) ---
            sample_x = [pos[n][0] for n in G.nodes if G.nodes[n].get('type') == 'sample']
            sample_y = [pos[n][1] for n in G.nodes if G.nodes[n].get('type') == 'sample']
            plt.plot(sample_x, sample_y, 'k.', markersize=1, alpha=0.5, label='Samples')

        # --- 5. Plot Special Nodes ---
        for key, color, marker, label_text in [
            ('starts', 'b', 'o', 'Start'), 
            ('goals', 'r', 'x', 'Goal'), 
            ('collections', 'orange', 'd', 'Collection')
        ]:
            if special_nodes[key]:
                sx, sy = zip(*special_nodes[key])
                plt.plot(sx, sy, color=color, marker=marker, linestyle='None', markersize=10, label=label_text, zorder=20)

        # --- 6. Plot Final Paths (If Available) ---
        if final_paths:
            colors = ['cyan', 'magenta', 'yellow', 'lime', 'blue']
            for i, (robot_name, coords) in enumerate(final_paths.items()):
                if not coords: continue
                c = colors[i % len(colors)]
                plt.plot(*zip(*coords), color=c, linewidth=4, alpha=0.8, label=f'Plan {robot_name}', zorder=30)
        
        # Fallback to plotting fragments if no final path
        elif path_data:
            def plot_category_paths(category, color_code):
                if category not in path_data: return
                for src_label, targets in path_data[category].items():
                    for tgt_label, info in targets.items():
                        path = info['coords']
                        is_best = info['is_best']
                        if is_best:
                            plt.plot(*zip(*path), color=color_code, linestyle='-', linewidth=2.5, alpha=0.4, zorder=5)

            plot_category_paths("starts", 'darkviolet') 
            plot_category_paths("goals", 'brown')

        # --- 7. Final Setup ---
        plt.legend(loc="upper right", fontsize='small', framealpha=0.9)
        plt.title(f"Plan (N={len(G.nodes)}, Edges={len(G.edges)})")
        plt.axis('equal')
        if bounds:
            plt.xlim(bounds[0], bounds[2])
            plt.ylim(bounds[1], bounds[3])
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _print_debug_comparison(self, sa_assignments, lns_assignments, cost_matrix, heading_matrix):
        """
        Prints a detailed step-by-step breakdown of SA vs LNS plans.
        """
        print("\n" + "="*80)
        print("DEBUG: DETAILED PLAN COMPARISON")
        print("="*80)

        # Helper to print one schedule
        def analyze_schedule(name, assignments):
            print(f"\n>>> ANALYSIS FOR: {name}")
            total_makespan = 0.0

            for r_name, tasks in assignments.items():
                if not tasks:
                    print(f"  [Robot {r_name}] IDLE")
                    continue

                print(f"  [Robot {r_name}]")
                
                # Get Start State
                curr_node = r_name
                # We need to find the robot's initial heading from the InitSimGlobalObservations
                # But here we just assume the allocator passed it correctly. 
                # For debug, let's grab it from the heading matrix if possible or assume 0
                # (Ideally, pass initial_headings dict to this function, but we'll infer).
                curr_heading = 0.0 # Placeholder, logic below fixes this relative to path
                
                robot_time = 0.0
                
                for i, task in enumerate(tasks):
                    # 1. Move to Goal
                    try:
                        # Data for Current -> Goal
                        d_g = cost_matrix.get(curr_node, {}).get(task.goal_id, 0.0)
                        angles_g = heading_matrix.get(curr_node, {}).get(task.goal_id, (0.0, 0.0))
                        
                        # Calculate Turn
                        # (We can't get exact start heading easily without passing more data, 
                        # but we can show the path heading)
                        path_h_start = angles_g[0]
                        path_h_end = angles_g[1]
                        
                        print(f"    {i+1}. {curr_node} -> {task.goal_id}")
                        print(f"       Dist: {d_g:.2f}s | Path Headings: {path_h_start:.2f} -> {path_h_end:.2f}")
                        
                        robot_time += d_g # (Plus turning time which is calculated in allocator)

                        # 2. Move to Collection
                        d_c = cost_matrix.get(task.goal_id, {}).get(task.collection_id, 0.0)
                        angles_c = heading_matrix.get(task.goal_id, {}).get(task.collection_id, (0.0, 0.0))
                        path_h_start_c = angles_c[0]
                        path_h_end_c = angles_c[1]

                        print(f"       {task.goal_id} -> {task.collection_id} (Collection)")
                        print(f"       Dist: {d_c:.2f}s | Path Headings: {path_h_start_c:.2f} -> {path_h_end_c:.2f}")
                        
                        robot_time += d_c

                        curr_node = task.collection_id
                    except Exception as e:
                        print(f"       ERROR analyzing task: {e}")

                print(f"    Total Approx Travel Time (excluding turns): {robot_time:.2f}s")
                total_makespan = max(total_makespan, robot_time)
            
            return total_makespan

        analyze_schedule("SA SOLUTION", sa_assignments)
        analyze_schedule("LNS SOLUTION", lns_assignments)
        print("="*80 + "\n")