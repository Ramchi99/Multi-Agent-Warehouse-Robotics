import random
import math
import time
import copy
import itertools # Added top-level import
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, List, Optional, Dict, Any

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
        self.eval_count = 0  # Track number of path evaluations
        self.history = [] # (time, cost) tuples

    def _evaluate_makespan(self, solution: Dict[str, RobotSchedule]) -> float:
        self.eval_count += len(solution)  # Count evaluation for each robot
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
                # Calculate turn cost considering REVERSE motion
                target_h = angles_1[0]
                diff = (target_h - current_heading + math.pi) % (2 * math.pi) - math.pi
                # Option 1: Forward (turn 'diff')
                c_fwd = abs(diff)
                # Option 2: Reverse (turn 'diff' + 180)
                c_rev = abs(c_fwd - math.pi)
                
                # Pick best
                if c_rev < c_fwd:
                    turn_cost = c_rev / self.w_max
                    # If we reverse, we arrive facing the opposite of the standard arrival heading
                    arrival_heading = (angles_1[1] + math.pi) % (2 * math.pi) - math.pi
                else:
                    turn_cost = c_fwd / self.w_max
                    arrival_heading = angles_1[1]

                total_time += turn_cost + d1
                current_heading = arrival_heading
                
                # --- B. MOVE TO COLLECTION ---
                d2 = self.matrix.get(task.goal_id, {}).get(task.collection_id, float('inf'))
                if d2 == float('inf'): return float('inf')
                
                angles_2 = self.heading_matrix.get(task.goal_id, {}).get(task.collection_id, (0.0, 0.0))
                
                target_h = angles_2[0]
                diff = (target_h - current_heading + math.pi) % (2 * math.pi) - math.pi
                c_fwd = abs(diff)
                c_rev = abs(c_fwd - math.pi)
                
                if c_rev < c_fwd:
                    turn_cost = c_rev / self.w_max
                    arrival_heading = (angles_2[1] + math.pi) % (2 * math.pi) - math.pi
                else:
                    turn_cost = c_fwd / self.w_max
                    arrival_heading = angles_2[1]
                
                total_time += turn_cost + d2
                current_heading = arrival_heading
                
                current_node = task.collection_id
            
            if total_time > max_time:
                max_time = total_time
        return max_time

    def _calculate_schedule_duration(self, r_name, tasks) -> float:
        """Calculates total time for a specific task list including turns."""
        self.eval_count += 1
        current_node = r_name
        current_heading = self.initial_headings.get(r_name, 0.0)
        total_time = 0.0
        
        for task in tasks:
            # 1. To Goal
            d1 = self.matrix.get(current_node, {}).get(task.goal_id, 0.0)
            angles_1 = self.heading_matrix.get(current_node, {}).get(task.goal_id, (0.0, 0.0))
            
            target_h = angles_1[0]
            diff = (target_h - current_heading + math.pi) % (2 * math.pi) - math.pi
            c_fwd = abs(diff)
            c_rev = abs(c_fwd - math.pi)
            
            if c_rev < c_fwd:
                turn_cost = c_rev / self.w_max
                arrival_heading = (angles_1[1] + math.pi) % (2 * math.pi) - math.pi
            else:
                turn_cost = c_fwd / self.w_max
                arrival_heading = angles_1[1]
                
            total_time += turn_cost + d1
            current_heading = arrival_heading
            
            # 2. To Collection
            d2 = self.matrix.get(task.goal_id, {}).get(task.collection_id, 0.0)
            angles_2 = self.heading_matrix.get(task.goal_id, {}).get(task.collection_id, (0.0, 0.0))
            
            target_h = angles_2[0]
            diff = (target_h - current_heading + math.pi) % (2 * math.pi) - math.pi
            c_fwd = abs(diff)
            c_rev = abs(c_fwd - math.pi)
            
            if c_rev < c_fwd:
                turn_cost = c_rev / self.w_max
                arrival_heading = (angles_2[1] + math.pi) % (2 * math.pi) - math.pi
            else:
                turn_cost = c_fwd / self.w_max
                arrival_heading = angles_2[1]
                
            total_time += turn_cost + d2
            current_heading = arrival_heading
            
            current_node = task.collection_id
            
        return total_time
    
    def _print_schedule_debug(self, solution: Dict[str, RobotSchedule]):
        pass
        # print("\n" + "="*60)
        # print("DEBUG: DETAILED SCHEDULE KINEMATICS (Reverse-Aware)")
        # print("="*60)
        
        # for r_name, sched in solution.items():
        #     if not sched.tasks:
        #         print(f"[Robot {r_name}] IDLE")
        #         continue
                
        #     print(f"[Robot {r_name}] {len(sched.tasks)} Tasks")
        #     curr_node = r_name
        #     curr_heading = self.initial_headings.get(r_name, 0.0)
            
        #     total_time = 0.0
            
        #     for i, task in enumerate(sched.tasks):
        #         print(f"  Task {i+1}: {curr_node} -> {task.goal_id} -> {task.collection_id}")
                
        #         # --- Segment 1: To Goal ---
        #         d1 = self.matrix.get(curr_node, {}).get(task.goal_id, 0.0)
        #         angles_1 = self.heading_matrix.get(curr_node, {}).get(task.goal_id, (0.0, 0.0))
        #         target_h = angles_1[0]
                
        #         # Turn Logic
        #         diff = (target_h - curr_heading + math.pi) % (2 * math.pi) - math.pi
        #         c_fwd = abs(diff)
        #         c_rev = abs(c_fwd - math.pi)
                
        #         is_reverse = c_rev < c_fwd
        #         turn_rad = c_rev if is_reverse else c_fwd
        #         turn_time = turn_rad / self.w_max
                
        #         if is_reverse:
        #             arr_heading = (angles_1[1] + math.pi) % (2 * math.pi) - math.pi
        #             move_type = "REVERSE"
        #         else:
        #             arr_heading = angles_1[1]
        #             move_type = "FORWARD"
                
        #         print(f"    1. To Goal ({task.goal_id}):")
        #         print(f"       Start Head: {curr_heading:.2f} | Path Vector: {target_h:.2f}")
        #         print(f"       Action: {move_type} | Turn: {turn_rad:.2f} rad ({turn_time:.2f}s)")
        #         print(f"       Travel: {d1:.2f}s | Total Seg: {turn_time + d1:.2f}s")
                
        #         total_time += turn_time + d1
        #         curr_heading = arr_heading
                
        #         # --- Segment 2: To Collection ---
        #         d2 = self.matrix.get(task.goal_id, {}).get(task.collection_id, 0.0)
        #         angles_2 = self.heading_matrix.get(task.goal_id, {}).get(task.collection_id, (0.0, 0.0))
        #         target_h = angles_2[0]
                
        #         diff = (target_h - curr_heading + math.pi) % (2 * math.pi) - math.pi
        #         c_fwd = abs(diff)
        #         c_rev = abs(c_fwd - math.pi)
                
        #         is_reverse = c_rev < c_fwd
        #         turn_rad = c_rev if is_reverse else c_fwd
        #         turn_time = turn_rad / self.w_max
                
        #         if is_reverse:
        #             arr_heading = (angles_2[1] + math.pi) % (2 * math.pi) - math.pi
        #             move_type = "REVERSE"
        #         else:
        #             arr_heading = angles_2[1]
        #             move_type = "FORWARD"
                
        #         print(f"    2. To Coll ({task.collection_id}):")
        #         print(f"       Start Head: {curr_heading:.2f} | Path Vector: {target_h:.2f}")
        #         print(f"       Action: {move_type} | Turn: {turn_rad:.2f} rad ({turn_time:.2f}s)")
        #         print(f"       Travel: {d2:.2f}s | Total Seg: {turn_time + d2:.2f}s")
                
        #         total_time += turn_time + d2
        #         curr_heading = arr_heading
                
        #         curr_node = task.collection_id
                
        #     print(f"  >> Total Robot Time: {total_time:.2f}s")
            
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

    def solve(self, time_limit: float = 2.0) -> Tuple[Dict[str, List[DeliveryTask]], List[Tuple[float, float]]]:
        print(f"--- Running SA Allocator (Time Limit: {time_limit}s) ---")
        start_time = time.time()
        
        current_solution = self._generate_greedy_solution()
        current_cost = self._evaluate_makespan(current_solution)
        
        best_solution_global = {r: sched.clone() for r, sched in current_solution.items()}
        best_cost_global = current_cost
        
        self.history.append((0.0, best_cost_global))
        
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
                    self.history.append((time.time() - start_time, best_cost_global))
            
            temperature *= cooling_rate
            if temperature < 0.5: temperature = 100.0 # Reheat

        print(f"SA Finished: Best Cost {best_cost_global:.2f} | Iterations: {iterations} | Evals: {self.eval_count}")
        return {r: sched.tasks for r, sched in best_solution_global.items()}, self.history
    
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
    def solve(self, time_limit: float = 2.0) -> Tuple[Dict[str, List[DeliveryTask]], List[Tuple[float, float]]]:
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
        self.history.append((0.0, best_cost))
        
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
                self.history.append((time.time() - start_time, best_cost))
            elif random.random() < 0.05:
                current_sol = temp_sol

        print(f"LNS Finished: Best Cost {best_cost:.2f} | Iterations: {iterations} | Evals: {self.eval_count}")
        
        # --- Print Optimal Path (User Requested Format) ---
        print("\n" + "="*60)
        print("OPTIMAL PATH SUMMARY (LNS)")
        print("="*60)
        path_lines = []
        for r_name in sorted(best_sol.keys()):
            sched = best_sol[r_name]
            segments = [f"robot {r_name}"]
            for task in sched.tasks:
                segments.append(f"--> goal {task.goal_id} -- collection {task.collection_id}")
            path_lines.append(" ".join(segments))
        print(", ".join(path_lines))
        print("="*60 + "\n")

        return {r: sched.tasks for r, sched in best_sol.items()}, self.history

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
                
                # Turn to Goal (Bidirectional)
                target_h = angles_pg[0]
                diff = (target_h - prev_heading + math.pi) % (2 * math.pi) - math.pi
                c_fwd = abs(diff)
                c_rev = abs(c_fwd - math.pi)
                
                if c_rev < c_fwd:
                    cost_pg = (c_rev / self.w_max) + d_pg
                    heading_at_goal = (angles_pg[1] + math.pi) % (2 * math.pi) - math.pi
                else:
                    cost_pg = (c_fwd / self.w_max) + d_pg
                    heading_at_goal = angles_pg[1]
                
                # Iterate all Collections
                for c in self.collections:
                    # Goal -> C
                    d_gc = self.matrix.get(task.goal_id, {}).get(c, float('inf'))
                    if d_gc == float('inf'): continue
                    
                    angles_gc = self.heading_matrix.get(task.goal_id, {}).get(c, (0.0, 0.0))
                    
                    target_h = angles_gc[0]
                    diff = (target_h - heading_at_goal + math.pi) % (2 * math.pi) - math.pi
                    c_fwd = abs(diff)
                    c_rev = abs((diff + math.pi) % (2 * math.pi) - math.pi)
                    
                    if c_rev < c_fwd:
                        cost_gc = (c_rev / self.w_max) + d_gc
                        heading_at_c = (angles_gc[1] + math.pi) % (2 * math.pi) - math.pi
                    else:
                        cost_gc = (c_fwd / self.w_max) + d_gc
                        heading_at_c = angles_gc[1]
                    
                    # C -> Next (if exists)
                    cost_cn = 0.0
                    if next_loc:
                        d_cn = self.matrix.get(c, {}).get(next_loc, float('inf'))
                        if d_cn == float('inf'): continue
                        angles_cn = self.heading_matrix.get(c, {}).get(next_loc, (0.0, 0.0))
                        
                        target_h = angles_cn[0]
                        diff = (target_h - heading_at_c + math.pi) % (2 * math.pi) - math.pi
                        c_fwd = abs(diff)
                        c_rev = abs((diff + math.pi) % (2 * math.pi) - math.pi)
                        
                        cost_cn = (min(c_fwd, c_rev) / self.w_max) + d_cn
                    
                    total = cost_pg + cost_gc + cost_cn
                    
                    if total < min_cost:
                        min_cost = total
                        best_c = c
                
                # Update Task
                task.collection_id = best_c
                
                # Update pointers for next iteration
                prev_loc = best_c
                
                # We need to correctly propagate heading based on the "best_c" choice
                # Re-calculate the specific exit heading for the chosen C
                if next_loc:
                    angles_next = self.heading_matrix.get(best_c, {}).get(next_loc, (0.0, 0.0))
                    # Check arriving at C (re-calc)
                    angles_gc = self.heading_matrix.get(task.goal_id, {}).get(best_c, (0.0, 0.0))
                    # We need to know if we arrived at C "reversed" to know start heading for Next?
                    # Actually, heading_matrix[c][next][0] is the start heading.
                    # The turn cost calculation handles the diff.
                    # But we need "heading_at_c" (arrival) to feed into the NEXT loop as "prev_heading".
                    
                    # Re-run logic for best_c to get exit heading
                    target_h_gc = angles_gc[0]
                    diff_gc = (target_h_gc - heading_at_goal + math.pi) % (2 * math.pi) - math.pi
                    if abs((diff_gc + math.pi) % (2 * math.pi) - math.pi) < abs(diff_gc):
                         heading_at_c = (angles_gc[1] + math.pi) % (2 * math.pi) - math.pi
                    else:
                         heading_at_c = angles_gc[1]

                    prev_heading = heading_at_c # Arriving at C, ready to turn to Next
                else:
                    # Same logic if it's the last one, just to update state
                    angles_gc = self.heading_matrix.get(task.goal_id, {}).get(best_c, (0.0, 0.0))
                    target_h_gc = angles_gc[0]
                    diff_gc = (target_h_gc - heading_at_goal + math.pi) % (2 * math.pi) - math.pi
                    if abs((diff_gc + math.pi) % (2 * math.pi) - math.pi) < abs(diff_gc):
                         prev_heading = (angles_gc[1] + math.pi) % (2 * math.pi) - math.pi
                    else:
                         prev_heading = angles_gc[1]


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
    
class TaskAllocatorLNS2(TaskAllocatorLNS):
    """
    Hybrid LNS with Micro-Exact Optimization.
    Guarantees optimal sequencing for small clusters using Permutations + Viterbi DP.
    """
    
    def solve(self, time_limit: float = 2.0) -> Tuple[Dict[str, List[DeliveryTask]], List[Tuple[float, float]]]:
        print(f"--- Running Hybrid LNS2 (Time Limit: {time_limit}s) ---")
        start_time = time.time()
        
        # 1. Initialization
        current_sol = {r: RobotSchedule(r, []) for r in self.robots}
        all_tasks = [DeliveryTask(g, self.collections[0]) for g in self.goals]
        
        # Construct initial solution with Noise
        current_sol = self._repair_regret_noise(current_sol, all_tasks, noise_level=0.1)
        self._intensify_solution(current_sol)
        
        best_sol = {r: sched.clone() for r, sched in current_sol.items()}
        best_cost = self._evaluate_makespan(best_sol)
        current_cost = best_cost
        self.history.append((0.0, best_cost))
        
        # SA Parameters
        temperature = 50.0 
        cooling_rate = 0.98 
        iterations = 0
        
        n_tasks = len(self.goals)
        min_rem = 1
        max_rem = max(1, min(4, int(n_tasks * 0.4)))
        
        while (time.time() - start_time) < time_limit:
            iterations += 1
            
            # A. Clone
            temp_sol = {r: sched.clone() for r, sched in current_sol.items()}
            
            # B. Destroy
            n_remove = random.randint(min_rem, max_rem)
            if random.random() < 0.6:
                temp_sol, removed_tasks = self._destroy_random(temp_sol, n_remove)
            else:
                temp_sol, removed_tasks = self._destroy_worst(temp_sol, n_remove)
            
            # C. Repair
            temp_sol = self._repair_regret_noise(temp_sol, removed_tasks, noise_level=0.2)
            
            # D. INTENSIFICATION (Exact Solver)
            self._intensify_solution(temp_sol)
            
            # E. Evaluate
            new_cost = self._evaluate_makespan(temp_sol)
            
            # F. Acceptance
            delta = new_cost - current_cost
            accepted = False
            
            if delta < 0:
                accepted = True
            elif random.random() < math.exp(-delta / max(temperature, 1e-5)):
                accepted = True
                
            if accepted:
                current_sol = temp_sol
                current_cost = new_cost
                if current_cost < best_cost:
                    best_sol = {r: sched.clone() for r, sched in current_sol.items()}
                    best_cost = current_cost
                    self.history.append((time.time() - start_time, best_cost))
            
            temperature *= cooling_rate

        print(f"LNS2 Finished: Best Cost {best_cost:.2f} | Iterations: {iterations} | Evals: {self.eval_count}")
        
        # --- Print Optimal Path (User Requested Format) ---
        print("\n" + "="*60)
        print("OPTIMAL PATH SUMMARY (LNS2)")
        print("="*60)
        path_lines = []
        for r_name in sorted(best_sol.keys()):
            sched = best_sol[r_name]
            segments = [f"robot {r_name}"]
            for task in sched.tasks:
                segments.append(f"--> goal {task.goal_id} -- collection {task.collection_id}")
            path_lines.append(" ".join(segments))
        print(", ".join(path_lines))
        print("="*60 + "\n")

        return {r: sched.tasks for r, sched in best_sol.items()}, self.history

    def _intensify_solution(self, solution):
        for r_name, sched in solution.items():
            if not sched.tasks: continue
            
            # If <= 6 tasks, brute-force ALL permutations (720 checks)
            # This GUARANTEES finding the [B, A] sequence if it's better.
            if len(sched.tasks) <= 6:
                self._optimize_route_exact(r_name, sched)
            else:
                self._optimize_route_2opt(r_name, sched)

    def _optimize_route_exact(self, r_name, schedule):
        """
        Brute-force checks ALL permutations of tasks.
        Uses Viterbi DP to optimize collection points for each permutation.
        """
        if len(schedule.tasks) < 1: return
        
        current_min_time = float('inf')
        best_perm_tasks = None
        
        start_heading = self.initial_headings.get(r_name, 0.0)
        
        # Use simple recursion or libraries if itertools is not available (but it is)
        
        for perm in itertools.permutations(schedule.tasks):
            # Clone tasks to act as candidates
            perm_tasks = [t.clone() for t in perm]
            
            # Calculate EXACT cost of this sequence using Viterbi
            # This updates perm_tasks with the optimal collection IDs in-place
            cost = self._optimize_dropoffs_exact_dp(r_name, start_heading, perm_tasks)
            
            if cost < current_min_time:
                current_min_time = cost
                best_perm_tasks = perm_tasks

        if best_perm_tasks:
            schedule.tasks = best_perm_tasks

    def _optimize_route_2opt(self, r_name, schedule):
        """Standard 2-opt with Viterbi cost evaluation."""
        improved = True
        start_heading = self.initial_headings.get(r_name, 0.0)
        
        while improved:
            improved = False
            current_time = self._optimize_dropoffs_exact_dp(r_name, start_heading, schedule.tasks)
            
            for i in range(len(schedule.tasks) - 1):
                for j in range(i + 1, len(schedule.tasks)):
                    schedule.tasks[i], schedule.tasks[j] = schedule.tasks[j], schedule.tasks[i]
                    
                    test_tasks = [t.clone() for t in schedule.tasks]
                    new_time = self._optimize_dropoffs_exact_dp(r_name, start_heading, test_tasks)
                    
                    if new_time < current_time - 1e-4:
                        current_time = new_time
                        schedule.tasks = test_tasks # Adopt optimized dropoffs
                        improved = True
                    else:
                        schedule.tasks[i], schedule.tasks[j] = schedule.tasks[j], schedule.tasks[i]
                    
                    if improved: break
                if improved: break

    def _optimize_dropoffs_exact_dp(self, start_node, start_heading, tasks) -> float:
        """
        Viterbi Algorithm to find optimal collection points.
        Returns the total cost of the path.
        Modifies 'tasks' in-place with the best collection_ids.
        """
        if not tasks: return 0.0
        
        # dp[i][c_label] = (min_cost_to_reach_here, parent_c_label, heading_at_arrival)
        dp = []
        
        # --- Layer 0: Start -> Goal0 -> C ---
        task0 = tasks[0]
        layer0 = {}
        
        d_sg = self.matrix.get(start_node, {}).get(task0.goal_id, float('inf'))
        if d_sg == float('inf'): return float('inf')
        
        angles_sg = self.heading_matrix.get(start_node, {}).get(task0.goal_id, (0.0, 0.0))
        target_h = angles_sg[0]
        diff = (target_h - start_heading + math.pi) % (2 * math.pi) - math.pi
        c_fwd = abs(diff)
        c_rev = abs((diff + math.pi) % (2 * math.pi) - math.pi)
        
        if c_rev < c_fwd:
            turn_sg = c_rev / self.w_max
            heading_arrival_g0 = (angles_sg[1] + math.pi) % (2 * math.pi) - math.pi
        else:
            turn_sg = c_fwd / self.w_max
            heading_arrival_g0 = angles_sg[1]
            
        cost_arrival_g0 = turn_sg + d_sg
        
        for c in self.collections:
            d_gc = self.matrix.get(task0.goal_id, {}).get(c, float('inf'))
            if d_gc == float('inf'): continue
            
            angles_gc = self.heading_matrix.get(task0.goal_id, {}).get(c, (0.0, 0.0))
            target_h = angles_gc[0]
            diff = (target_h - heading_arrival_g0 + math.pi) % (2 * math.pi) - math.pi
            c_fwd = abs(diff)
            c_rev = abs((diff + math.pi) % (2 * math.pi) - math.pi)
            
            if c_rev < c_fwd:
                turn_gc = c_rev / self.w_max
                heading_arrival_c = (angles_gc[1] + math.pi) % (2 * math.pi) - math.pi
            else:
                turn_gc = c_fwd / self.w_max
                heading_arrival_c = angles_gc[1]
            
            total = cost_arrival_g0 + turn_gc + d_gc
            layer0[c] = (total, None, heading_arrival_c)
            
        dp.append(layer0)
        
        # --- Layers 1..N: Prev_C -> Goal_i -> Curr_C ---
        for i in range(1, len(tasks)):
            curr_task = tasks[i]
            prev_layer = dp[-1]
            curr_layer = {}
            
            if not prev_layer: return float('inf')
            
            # We iterate all possible Prev_C to find best path to Curr_C
            for prev_c, (prev_cost, _, prev_heading_arr) in prev_layer.items():
                
                # 1. Prev_C -> Goal_i
                d_pg = self.matrix.get(prev_c, {}).get(curr_task.goal_id, float('inf'))
                if d_pg == float('inf'): continue
                
                angles_pg = self.heading_matrix.get(prev_c, {}).get(curr_task.goal_id, (0.0, 0.0))
                
                target_h = angles_pg[0]
                diff = (target_h - prev_heading_arr + math.pi) % (2 * math.pi) - math.pi
                c_fwd = abs(diff)
                c_rev = abs((diff + math.pi) % (2 * math.pi) - math.pi)
                
                if c_rev < c_fwd:
                    turn_pg = c_rev / self.w_max
                    heading_arr_g = (angles_pg[1] + math.pi) % (2 * math.pi) - math.pi
                else:
                    turn_pg = c_fwd / self.w_max
                    heading_arr_g = angles_pg[1]

                cost_arr_g = prev_cost + turn_pg + d_pg
                
                # 2. Goal_i -> Curr_C
                for curr_c in self.collections:
                    d_gc = self.matrix.get(curr_task.goal_id, {}).get(curr_c, float('inf'))
                    if d_gc == float('inf'): continue
                    
                    angles_gc = self.heading_matrix.get(curr_task.goal_id, {}).get(curr_c, (0.0, 0.0))
                    
                    target_h = angles_gc[0]
                    diff = (target_h - heading_arr_g + math.pi) % (2 * math.pi) - math.pi
                    c_fwd = abs(diff)
                    c_rev = abs(c_fwd - math.pi)
                    
                    if c_rev < c_fwd:
                        turn_gc = c_rev / self.w_max
                        heading_arr_curr_c = (angles_gc[1] + math.pi) % (2 * math.pi) - math.pi
                    else:
                        turn_gc = c_fwd / self.w_max
                        heading_arr_curr_c = angles_gc[1]
                        
                    total_new_cost = cost_arr_g + turn_gc + d_gc
                    
                    # Store if best
                    if curr_c not in curr_layer or total_new_cost < curr_layer[curr_c][0]:
                        curr_layer[curr_c] = (total_new_cost, prev_c, heading_arr_curr_c)
            
            dp.append(curr_layer)
            
        # --- Backtrack ---
        last_layer = dp[-1]
        if not last_layer: return float('inf')
        
        best_end_c = min(last_layer, key=lambda k: last_layer[k][0])
        min_total_cost = last_layer[best_end_c][0]
        
        curr_c = best_end_c
        for i in range(len(tasks) - 1, -1, -1):
            tasks[i].collection_id = curr_c
            curr_c = dp[i][curr_c][1]
            
        return min_total_cost

    def _repair_regret_noise(self, solution, tasks_to_insert, noise_level=0.1):
        """Regret Insertion with Random Noise to Cost."""
        random.shuffle(tasks_to_insert)
        
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
                    for i in range(len(sched.tasks) + 1):
                        cost_increase, best_c = self._calc_insertion_cost(sched, task, i)
                        if best_c is None: continue
                        
                        noise = random.uniform(1.0 - noise_level, 1.0 + noise_level)
                        new_finish_time = (current_r_time + cost_increase) * noise
                        options.append((new_finish_time, cost_increase, r_name, i, best_c))
                
                if not options: continue
                options.sort(key=lambda x: x[0])
                
                best = options[0]
                second = options[1][0] if len(options) > 1 else float('inf')
                
                # Regret with Noisy Values
                regret_val = second - best[0]
                
                regrets.append({
                    'regret': regret_val, 
                    'task': task, 
                    'vals': best
                })
            
            if not regrets: break
            # Sort by highest regret
            regrets.sort(key=lambda x: x['regret'], reverse=True)
            
            winner = regrets[0]
            t = winner['task']
            vals = winner['vals'] 
            
            t.collection_id = vals[4]
            solution[vals[2]].tasks.insert(vals[3], t)
            # Update cache with REAL cost (not noisy)
            robot_finish_times[vals[2]] += vals[1]
            
            tasks_to_insert.remove(t)
            
        return solution

class TaskAllocatorLNS3(TaskAllocatorLNS2):
    """
    Final LNS3:
    1. Viterbi-Exact Solver (Fixes 1-Robot Sequencing)
    2. Spatial Destruction (Fixes Multi-Robot Overlap)
    3. Noise Injection (Prevents Cycles)
    """
    
    def solve(self, time_limit: float = 2.0) -> Tuple[Dict[str, List[DeliveryTask]], List[Tuple[float, float]]]:
        print(f"--- Running Hybrid LNS3 (Time Limit: {time_limit}s) ---")
        start_time = time.time()
        
        current_sol = {r: RobotSchedule(r, []) for r in self.robots}
        all_tasks = [DeliveryTask(g, self.collections[0]) for g in self.goals]
        
        current_sol = self._repair_regret_noise(current_sol, all_tasks, noise_level=0.1)
        self._intensify_solution(current_sol)
        
        best_sol = {r: sched.clone() for r, sched in current_sol.items()}
        best_cost = self._evaluate_makespan(best_sol)
        current_cost = best_cost
        self.history.append((0.0, best_cost))
        
        temperature = 50.0 
        cooling_rate = 0.98 
        iterations = 0
        
        n_tasks = len(self.goals)
        min_rem = 1
        max_rem = max(1, min(4, int(n_tasks * 0.4)))
        
        while (time.time() - start_time) < time_limit:
            iterations += 1
            
            temp_sol = {r: sched.clone() for r, sched in current_sol.items()}
            
            # --- ADAPTIVE DESTRUCTION STRATEGY ---
            n_remove = random.randint(min_rem, max_rem)
            r_val = random.random()
            
            if r_val < 0.4:
                temp_sol, removed_tasks = self._destroy_random(temp_sol, n_remove)
            elif r_val < 0.7:
                temp_sol, removed_tasks = self._destroy_worst(temp_sol, n_remove)
            else:
                # [NEW] SPATIAL DESTRUCTION
                temp_sol, removed_tasks = self._destroy_spatial(temp_sol, n_remove)
            
            temp_sol = self._repair_regret_noise(temp_sol, removed_tasks, noise_level=0.2)
            self._intensify_solution(temp_sol)
            new_cost = self._evaluate_makespan(temp_sol)
            
            delta = new_cost - current_cost
            accepted = False
            
            if delta < 0:
                accepted = True
            elif random.random() < math.exp(-delta / max(temperature, 1e-5)):
                accepted = True
                
            if accepted:
                current_sol = temp_sol
                current_cost = new_cost
                if current_cost < best_cost:
                    best_sol = {r: sched.clone() for r, sched in current_sol.items()}
                    best_cost = current_cost
                    self.history.append((time.time() - start_time, best_cost))
            
            temperature *= cooling_rate

        print(f"LNS3 Finished: Best Cost {best_cost:.2f} | Iterations: {iterations} | Evals: {self.eval_count}")
        
        # --- Print Optimal Path (User Requested Format) ---
        print("\n" + "="*60)
        print("OPTIMAL PATH SUMMARY (LNS3)")
        print("="*60)
        path_lines = []
        for r_name in sorted(best_sol.keys()):
            sched = best_sol[r_name]
            segments = [f"robot {r_name}"]
            for task in sched.tasks:
                segments.append(f"--> goal {task.goal_id} -- collection {task.collection_id}")
            path_lines.append(" ".join(segments))
        print(", ".join(path_lines))
        print("="*60 + "\n")

        return {r: sched.tasks for r, sched in best_sol.items()}, self.history

    def _destroy_spatial(self, solution, n):
        """
        Removes a set of tasks that are geographically close to each other.
        Crucial for multi-robot scenarios (Task 2 & 3) to fix territory overlaps.
        """
        all_tasks = []
        for r, sched in solution.items():
            for t in sched.tasks: all_tasks.append((r, t))
        
        if not all_tasks: return solution, []
        
        # 1. Pick random seed task
        seed_r, seed_t = random.choice(all_tasks)
        seed_goal = seed_t.goal_id
        
        # 2. Calculate distance from seed to ALL other tasks
        # We use the cost matrix (time distance) as a proxy for spatial distance
        dists = []
        for r, t in all_tasks:
            dist = self.matrix.get(seed_goal, {}).get(t.goal_id, float('inf'))
            dists.append((dist, r, t))
            
        # 3. Sort by closeness and remove top N
        dists.sort(key=lambda x: x[0])
        to_remove = dists[:n]
        
        removed_objs = []
        for _, r_name, task in to_remove:
            if task in solution[r_name].tasks:
                solution[r_name].tasks.remove(task)
                removed_objs.append(task)
                
        return solution, removed_objs

class TaskAllocatorALNS(TaskAllocatorBase):
    """
    Adaptive Large Neighborhood Search (ALNS) - Standalone Implementation.
    
    This class implements the State-of-the-Art ALNS metaheuristic.
    It adaptively selects between different 'Destroy' operators based on their 
    past performance, allowing it to perform like LNS2 (Random/Worst) when 
    time is short or the problem is simple, and like LNS3 (Spatial) when 
    escaping deep local optima is required.
    
    It is fully self-contained (except for the Base data holder) to allow 
    deprecation of previous classes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # --- ALNS Parameters ---
        # [IMPROVEMENT] Conservative Spatial Start:
        # Start 'spatial' lower so we behave like the fast LNS2 initially.
        # If spatial proves useful (long runs), it will earn its way up.
        # [IMPROVEMENT] Exploration Bias:
        # Start with 1.5/1.0 (~60/40 split) to match LNS2's optimal strategy.
        self.scores = {
            'random': 1.0,   # Was 1.5
            'worst': 2.0,    # Was 1.0
            'spatial': 0.1,  # Was 0.1
            'critical': 3.0  # Was 2.0
        }
        self.usage_counts = {k: 0 for k in self.scores}
        
        # Rewards
        self.sigma_1 = 2.0  # New Global Best
        self.sigma_2 = 1.0  # Improved Current
        self.sigma_3 = 0.1  # Accepted (worse but accepted)
        self.sigma_4 = 0.0  # Rejected
        
        self.decay = 0.95   # Smoothing factor for weights
        
    def _get_sol_hash(self, solution):
        # Canonical string representation for uniqueness check
        s = []
        for r in sorted(solution.keys()):
            tasks = tuple((t.goal_id, t.collection_id) for t in solution[r].tasks)
            s.append((r, tasks))
        return tuple(s)

    def solve(self, time_limit: float = 2.0) -> Tuple[Dict[str, List[DeliveryTask]], List[Dict[str, Any]]]:
        print(f"--- Running ALNS (Time Limit: {time_limit}s) ---")
        start_time = time.time()
        
        # 1. Initialization
        current_sol = {r: RobotSchedule(r, []) for r in self.robots}
        all_tasks = [DeliveryTask(g, self.collections[0]) for g in self.goals]
        
        # Construct initial solution
        current_sol = self._repair_regret_noise(current_sol, all_tasks, noise_level=0.1)
        self._intensify_solution(current_sol)
        
        best_sol = {r: sched.clone() for r, sched in current_sol.items()}
        best_cost = self._evaluate_makespan(best_sol)
        current_cost = best_cost
        
        telemetry_log = []
        
        # [NEW] Top K Tracking
        top_k_solutions = [] 
        seen_hashes = set()
        
        def try_add_top_k(sol, cost):
            h = self._get_sol_hash(sol)
            if h in seen_hashes: return
            
            # Deep Copy for storage
            sol_copy = {r: sched.clone() for r, sched in sol.items()}
            
            top_k_solutions.append((cost, sol_copy))
            seen_hashes.add(h)
            
            # Sort by cost (ascending)
            top_k_solutions.sort(key=lambda x: x[0])
            
            if len(top_k_solutions) > 10:
                top_k_solutions.pop() # Remove worst
        
        try_add_top_k(best_sol, best_cost)
        
        # [IMPROVEMENT] Dynamic Temperature
        temperature = max(50.0, current_cost * 0.5)
        start_temp = temperature
        
        cooling_rate = 0.98 
        iterations = 0
        stagnation_threshold = 300 # Was 100 
        
        n_tasks = len(self.goals)
        min_rem = 1
        max_rem = max(1, min(4, int(n_tasks * 0.4)))
        
        # [IMPROVEMENT] Complexity Matching
        if n_tasks < 10:
            self.scores['spatial'] = 0.0
            
        iterations_since_improvement = 0
        force_kick = False # For massive destroy
        consecutive_failed_kicks = 0
        reheat_count = 0
        
        # NEW: Unique Solution Tracking (Initialization)
        best_solutions_count = 1 # Start with 1 for the initial best_sol
        unique_best_solutions = {self._get_sol_hash(best_sol)}
        
        # ALNS Loop
        while (time.time() - start_time) < time_limit:
            iterations += 1
            
            # [IMPROVEMENT] Variable Neighborhood Size (VND)
            if force_kick:
                # 1. ANALYZE CONTEXT (Imbalance)
                times = [self._calculate_schedule_duration(r, current_sol[r].tasks) for r in self.robots]
                if not times:
                    max_dur = 1.0
                    imbalance = 0.0
                else:
                    max_dur = max(times)
                    imbalance = max(times) - min(times)
                
                # Ratio: e.g., 0.15 means imbalance is 15% of the total time
                ratio = imbalance / max(max_dur, 1e-5)

                # 2. DECIDE KICK STRATEGY
                # CASE A: Imbalance is TOO HIGH (> 15%) -> Force fix bottleneck
                if ratio > 0.15:
                    op_name = 'critical'
                    n_remove = int(n_tasks * 0.5)
                    # print(f"  >> CONTEXT KICK: Imbalance High ({ratio:.1%}) -> SNIPER")
                    
                # CASE B: Imbalance is TOO LOW (< 5%) -> Force efficiency scramble
                # (You are over-optimizing balance at the cost of turning efficiency)
                elif ratio < 0.05:
                    op_name = 'random'
                    n_remove = int(n_tasks * 0.5)
                    # print(f"  >> CONTEXT KICK: Imbalance Low ({ratio:.1%}) -> SCRAMBLE")

                # CASE C: Imbalance is GOOD (Goldilocks) -> Use Cyclic Logic
                else:
                    kick_phase = consecutive_failed_kicks % 3
                    if kick_phase == 0:
                        # Strategy 1: Standard Random
                        op_name = 'random'
                        n_remove = int(n_tasks * 0.5)
                    elif kick_phase == 1:
                        # Strategy 2: Nuclear (Deep Escape)
                        op_name = 'random'
                        n_remove = int(n_tasks * 0.7)
                    else:
                        # Strategy 3: Critical (Just in case)
                        op_name = 'critical'
                        n_remove = int(n_tasks * 0.5)
                    # print(f"  >> CYCLIC KICK: Phase {kick_phase}")

                force_kick = False
                
                # Boost temp: Higher boost for Nuclear or High-Context fixes
                kick_strength = 0.8 if (op_name == 'random' and n_remove > n_tasks*0.6) else 0.6
                temperature = max(temperature, start_temp * kick_strength)
                
            else:
                # Normal Scalpel Mode
                op_name = self._select_operator()
                self.usage_counts[op_name] += 1
                base_n = 1 + int(iterations_since_improvement / 50)  # was 20!
                # n_remove = min(max_rem, base_n)
                n_remove = min(4, 1 + base_n)
            
            # B. Clone
            temp_sol = {r: sched.clone() for r, sched in current_sol.items()}
            
            # C. Destroy
            removed_tasks = []
            
            if op_name == 'random':
                temp_sol, removed_tasks = self._destroy_random(temp_sol, n_remove)
            elif op_name == 'worst':
                temp_sol, removed_tasks = self._destroy_worst(temp_sol, n_remove)
            elif op_name == 'spatial':
                temp_sol, removed_tasks = self._destroy_spatial(temp_sol, n_remove)
            elif op_name == 'critical':
                temp_sol, removed_tasks = self._destroy_critical(temp_sol, n_remove)
            
            # D. Repair
            current_noise = 0.2 * (temperature / 50.0)
            temp_sol = self._repair_regret_noise(temp_sol, removed_tasks, noise_level=current_noise)
            
            # E. Intensify (Micro-Optimization)
            self._intensify_solution(temp_sol)
            
            # F. Evaluate
            new_cost = self._evaluate_makespan(temp_sol)
            
            # [NEW] Update Top K
            try_add_top_k(temp_sol, new_cost)
            
            # G. Acceptance & Scoring
            delta = new_cost - current_cost
            accepted = False
            reward = self.sigma_4
            outcome_code = 0 # 0=Reject
            
            if delta < 0:
                accepted = True
                # 1. New Global Best
                if new_cost < best_cost - 1e-6:
                    outcome_code = 3 # Global Best
                    reward = self.sigma_1
                    best_sol = {r: sched.clone() for r, sched in temp_sol.items()}
                    best_cost = new_cost
                    iterations_since_improvement = 0 
                    reheat_count = 0 
                    consecutive_failed_kicks = 0
                    
                    # Reset Unique Count
                    best_solutions_count = 1
                    unique_best_solutions = {self._get_sol_hash(best_sol)}
                
                # 2. Alternative Optimal
                elif abs(new_cost - best_cost) < 1e-6:
                    outcome_code = 2
                    reward = self.sigma_2
                    iterations_since_improvement += 1
                    
                    sol_hash = self._get_sol_hash(temp_sol)
                    if sol_hash not in unique_best_solutions:
                        unique_best_solutions.add(sol_hash)
                        best_solutions_count += 1

                else:
                    outcome_code = 2 # Improved Local
                    reward = self.sigma_2
                    iterations_since_improvement += 1
            elif random.random() < math.exp(-delta / max(temperature, 1e-5)):
                accepted = True
                outcome_code = 1 # Accepted Worse
                reward = self.sigma_3
                iterations_since_improvement += 1
            else:
                iterations_since_improvement += 1
            
            if accepted:
                current_sol = temp_sol
                current_cost = new_cost
            
            # --- LOGGING ---
            times_list = [self._calculate_schedule_duration(r, current_sol[r].tasks) for r in self.robots]
            imbalance = max(times_list) - min(times_list) if times_list else 0
            
            reheat_flag = 0
            # --- THE FIX: REHEATING LOGIC ---
            if iterations_since_improvement > stagnation_threshold:
                # print(f"  >> STAGNATION DETECTED (Iter {iterations}) >> REHEATING!")
                temperature = start_temp * 0.75 # Was 0.5
                current_sol = {r: sched.clone() for r, sched in best_sol.items()}
                current_cost = best_cost
                iterations_since_improvement = 0
                reheat_flag = 1
                reheat_count += 1
                
                if reheat_count > 1:
                    force_nuclear = True
                else:
                    force_kick = True # Trigger massive destroy next iter
            
            log_entry = {
                "time": time.time() - start_time,
                "iter": iterations,
                "temp": temperature,
                "best_cost": best_cost,
                "curr_cost": current_cost,
                "operator": op_name,
                "outcome": outcome_code,
                "imbalance": imbalance,
                "reheat": reheat_flag
            }
            telemetry_log.append(log_entry)
            
            # H. Update Weights
            self._update_weight(op_name, reward)
            
            temperature *= cooling_rate
            
            # Safety floor
            if temperature < 0.01: 
                temperature = start_temp * 0.1
            # temperature *= cooling_rate
            # if temperature < 0.5: temperature = 0.5 # Keep it simmering

        # Print Statistics
        print(f"ALNS Finished: Best Cost {best_cost:.2f} | Iterations: {iterations} | Unique Optimal Solutions: {best_solutions_count}")
        print(f"Operator Scores: {self.scores}")
        print(f"Operator Usage: {self.usage_counts}")
        
        # [DEBUG] Print Detailed Kinematics
        self._print_schedule_debug(best_sol)

        # --- Print Optimal Path (User Requested Format) ---
        print("\n" + "="*60)
        print("OPTIMAL PATH SUMMARY")
        print("="*60)
        path_lines = []
        for r_name in sorted(best_sol.keys()):
            sched = best_sol[r_name]
            segments = [f"robot {r_name}"]
            for task in sched.tasks:
                segments.append(f"--> goal {task.goal_id} -- collection {task.collection_id}")
            path_lines.append(" ".join(segments))
        print(", ".join(path_lines))
        print("="*60 + "\n")
        
        # [NEW] Return Top K (Formatted)
        top_k_formatted = [{r: sched.tasks for r, sched in sol[1].items()} for sol in top_k_solutions]
        return {r: sched.tasks for r, sched in best_sol.items()}, telemetry_log, top_k_formatted

    def _select_operator(self):
        """Roulette Wheel Selection based on weights."""
        total = sum(self.scores.values())
        r = random.uniform(0, total)
        curr = 0
        for name, score in self.scores.items():
            curr += score
            if r <= curr:
                return name
        return 'random' # Fallback

    def _update_weight(self, op_name, reward):
        old_score = self.scores[op_name]
        new_score = self.decay * old_score + (1 - self.decay) * reward
        
        # Clamp to avoid starvation
        self.scores[op_name] = max(0.1, new_score)
        
        # [IMPROVEMENT] Guaranteed Exploration
        # Never let 'random' drop below a safe floor.
        # This prevents the algorithm from becoming too greedy/deterministic.
        if op_name == 'random':
            self.scores['random'] = max(0.6, self.scores['random'])
        elif op_name == 'worst':
            self.scores['worst'] = max(0.4, self.scores['worst'])

    def _destroy_critical(self, solution, n):
        """Removes tasks specifically from the bottleneck robot."""
        # 1. Find the bottleneck robot
        max_time = -1
        worst_robot = None
        
        for r, sched in solution.items():
            t = self._calculate_schedule_duration(r, sched.tasks)
            if t > max_time:
                max_time = t
                worst_robot = r
        
        # Safety check
        if not worst_robot or not solution[worst_robot].tasks:
            return self._destroy_random(solution, n) 
            
        # 2. Remove N random tasks ONLY from this robot
        sched = solution[worst_robot]
        k = min(n, len(sched.tasks))
        removed_tasks = []
        
        for _ in range(k):
            t = sched.tasks.pop(random.randint(0, len(sched.tasks)-1))
            removed_tasks.append(t)
            
        return solution, removed_tasks

    # --- 1. DESTROY OPERATORS ---
    
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
                
                d1 = self.matrix.get(prev_loc, {}).get(task.goal_id, 0)
                d2 = self.matrix.get(task.goal_id, {}).get(task.collection_id, 0)
                d3 = self.matrix.get(task.collection_id, {}).get(next_loc, 0) if next_loc else 0
                shortcut = self.matrix.get(prev_loc, {}).get(next_loc, 0) if next_loc else 0
                
                savings = (d1 + d2 + d3) - shortcut
                candidates.append((savings, r_name, task))
                prev_loc = task.collection_id
                
        candidates.sort(key=lambda x: x[0], reverse=True)
        removed_objs = []
        for _, r_name, task in candidates[:n]:
            if task in solution[r_name].tasks:
                solution[r_name].tasks.remove(task)
                removed_objs.append(task)
                
        return solution, removed_objs

    def _destroy_spatial(self, solution, n):
        all_tasks = []
        for r, sched in solution.items():
            for t in sched.tasks: all_tasks.append((r, t))
        if not all_tasks: return solution, []
        
        seed_r, seed_t = random.choice(all_tasks)
        seed_goal = seed_t.goal_id
        
        dists = []
        for r, t in all_tasks:
            dist = self.matrix.get(seed_goal, {}).get(t.goal_id, float('inf'))
            dists.append((dist, r, t))
            
        dists.sort(key=lambda x: x[0])
        to_remove = dists[:n]
        
        removed_objs = []
        for _, r_name, task in to_remove:
            if task in solution[r_name].tasks:
                solution[r_name].tasks.remove(task)
                removed_objs.append(task)
                
        return solution, removed_objs

    # --- 2. REPAIR OPERATORS ---
    
    def _repair_regret_noise(self, solution, tasks_to_insert, noise_level=0.1):
        random.shuffle(tasks_to_insert)
        
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
                    for i in range(len(sched.tasks) + 1):
                        cost_increase, best_c = self._calc_insertion_cost(sched, task, i)
                        if best_c is None: continue
                        
                        noise = random.uniform(1.0 - noise_level, 1.0 + noise_level)
                        new_finish_time = (current_r_time + cost_increase) * noise
                        options.append((new_finish_time, cost_increase, r_name, i, best_c))
                
                if not options: continue
                # Sort by Weighted Objective (Makespan priority)
                # alpha = 0.7
                alpha = 0.7
                options.sort(key=lambda x: (alpha * x[0]) + ((1 - alpha) * x[1]))
                # Simple Min-Finish time works well for Makespan
                # options.sort(key=lambda x: x[0])
                
                best = options[0]
                second = options[1][0] if len(options) > 1 else float('inf')
                
                regret_val = second - best[0]
                
                regrets.append({
                    'regret': regret_val, 
                    'task': task, 
                    'vals': best
                })
            
            if not regrets: break
            regrets.sort(key=lambda x: x['regret'], reverse=True)
            
            winner = regrets[0]
            t = winner['task']
            vals = winner['vals'] 
            
            t.collection_id = vals[4]
            solution[vals[2]].tasks.insert(vals[3], t)
            robot_finish_times[vals[2]] += vals[1]
            
            tasks_to_insert.remove(t)
            
        return solution
    
    def _calc_insertion_cost(self, schedule, task, idx):
        if idx == 0:
            prev_loc = schedule.robot_name
            prev_heading = self.initial_headings.get(prev_loc, 0.0)
        else:
            prev_task = schedule.tasks[idx-1]
            prev_loc = prev_task.collection_id
            # Approx Heading
            pg = prev_task.goal_id
            pc = prev_task.collection_id
            prev_heading = self.heading_matrix.get(pg, {}).get(pc, (0.0, 0.0))[1]

        next_loc = schedule.tasks[idx].goal_id if idx < len(schedule.tasks) else None
        
        # Base Cost: Prev -> Goal
        d_pg = self.matrix.get(prev_loc, {}).get(task.goal_id, float('inf'))
        if d_pg == float('inf'): return float('inf'), None
        
        angles_pg = self.heading_matrix.get(prev_loc, {}).get(task.goal_id, (0.0, 0.0))
        target_h = angles_pg[0]
        diff = (target_h - prev_heading + math.pi) % (2 * math.pi) - math.pi
        c_fwd = abs(diff)
        c_rev = abs((diff + math.pi) % (2 * math.pi) - math.pi)
        
        if c_rev < c_fwd:
            turn_pg = c_rev / self.w_max
            heading_at_goal = (angles_pg[1] + math.pi) % (2 * math.pi) - math.pi
        else:
            turn_pg = c_fwd / self.w_max
            heading_at_goal = angles_pg[1]

        cost_pg = turn_pg + d_pg
        
        # Optimize C
        best_c = None
        min_seg = float('inf')
        
        for c in self.collections:
            d_gc = self.matrix.get(task.goal_id, {}).get(c, float('inf'))
            if d_gc == float('inf'): continue
            
            angles_gc = self.heading_matrix.get(task.goal_id, {}).get(c, (0.0, 0.0))
            
            target_h = angles_gc[0]
            diff = (target_h - heading_at_goal + math.pi) % (2 * math.pi) - math.pi
            c_fwd = abs(diff)
            c_rev = abs(c_fwd - math.pi)
            
            if c_rev < c_fwd:
                turn_gc = c_rev / self.w_max
                heading_at_c = (angles_gc[1] + math.pi) % (2 * math.pi) - math.pi
            else:
                turn_gc = c_fwd / self.w_max
                heading_at_c = angles_gc[1]

            cost_gc = turn_gc + d_gc
            
            cost_cn = 0.0
            if next_loc:
                d_cn = self.matrix.get(c, {}).get(next_loc, float('inf'))
                if d_cn == float('inf'): continue
                angles_cn = self.heading_matrix.get(c, {}).get(next_loc, (0.0, 0.0))
                
                target_h = angles_cn[0]
                diff = (target_h - heading_at_c + math.pi) % (2 * math.pi) - math.pi
                c_fwd = abs(diff)
                c_rev = abs(c_fwd - math.pi)
                
                # We don't track heading after this, just cost
                cost_cn = (min(c_fwd, c_rev) / self.w_max) + d_cn
            
            if (cost_gc + cost_cn) < min_seg:
                min_seg = cost_gc + cost_cn
                best_c = c
                
        if best_c is None: return float('inf'), None
        
        added = cost_pg + min_seg
        removed = 0.0
        if next_loc:
            d_pn = self.matrix.get(prev_loc, {}).get(next_loc, float('inf'))
            if d_pn < float('inf'):
                angles_pn = self.heading_matrix.get(prev_loc, {}).get(next_loc, (0.0, 0.0))
                target_h = angles_pn[0]
                diff = (target_h - prev_heading + math.pi) % (2 * math.pi) - math.pi
                c_fwd = abs(diff)
                c_rev = abs(c_fwd - math.pi)
                
                removed = (min(c_fwd, c_rev) / self.w_max) + d_pn
                
        return added - removed, best_c

    # --- 3. INTENSIFICATION (Exact Viterbi) ---
    
    def _intensify_solution(self, solution):
        for r_name, sched in solution.items():
            if not sched.tasks: continue
            if len(sched.tasks) <= 6:
                self._optimize_route_exact(r_name, sched)
            else:
                self._optimize_route_2opt(r_name, sched)

    def _optimize_route_exact(self, r_name, schedule):
        if len(schedule.tasks) < 1: return
        current_min_time = float('inf')
        best_perm_tasks = None
        start_heading = self.initial_headings.get(r_name, 0.0)
        
        for perm in itertools.permutations(schedule.tasks):
            perm_tasks = [t.clone() for t in perm]
            cost = self._optimize_dropoffs_exact_dp(r_name, start_heading, perm_tasks)
            if cost < current_min_time:
                current_min_time = cost
                best_perm_tasks = perm_tasks

        if best_perm_tasks:
            schedule.tasks = best_perm_tasks

    def _optimize_route_2opt(self, r_name, schedule):
        improved = True
        start_heading = self.initial_headings.get(r_name, 0.0)
        while improved:
            improved = False
            current_time = self._optimize_dropoffs_exact_dp(r_name, start_heading, schedule.tasks)
            for i in range(len(schedule.tasks) - 1):
                for j in range(i + 1, len(schedule.tasks)):
                    schedule.tasks[i], schedule.tasks[j] = schedule.tasks[j], schedule.tasks[i]
                    test_tasks = [t.clone() for t in schedule.tasks]
                    new_time = self._optimize_dropoffs_exact_dp(r_name, start_heading, test_tasks)
                    if new_time < current_time - 1e-4:
                        current_time = new_time
                        schedule.tasks = test_tasks
                        improved = True
                    else:
                        schedule.tasks[i], schedule.tasks[j] = schedule.tasks[j], schedule.tasks[i]
                    if improved: break
                if improved: break

    def _optimize_dropoffs_exact_dp(self, start_node, start_heading, tasks) -> float:
        if not tasks: return 0.0
        dp = []
        
        # --- Layer 0: Start -> Goal0 -> C ---
        task0 = tasks[0]
        layer0 = {}
        
        d_sg = self.matrix.get(start_node, {}).get(task0.goal_id, float('inf'))
        if d_sg == float('inf'): return float('inf')
        
        angles_sg = self.heading_matrix.get(start_node, {}).get(task0.goal_id, (0.0, 0.0))
        target_h = angles_sg[0]
        diff = (target_h - start_heading + math.pi) % (2 * math.pi) - math.pi
        c_fwd = abs(diff)
        c_rev = abs((diff + math.pi) % (2 * math.pi) - math.pi)
        
        if c_rev < c_fwd:
            turn_sg = c_rev / self.w_max
            heading_arrival_g0 = (angles_sg[1] + math.pi) % (2 * math.pi) - math.pi
        else:
            turn_sg = c_fwd / self.w_max
            heading_arrival_g0 = angles_sg[1]
            
        cost_arrival_g0 = turn_sg + d_sg
        
        for c in self.collections:
            d_gc = self.matrix.get(task0.goal_id, {}).get(c, float('inf'))
            if d_gc == float('inf'): continue
            
            angles_gc = self.heading_matrix.get(task0.goal_id, {}).get(c, (0.0, 0.0))
            target_h = angles_gc[0]
            diff = (target_h - heading_arrival_g0 + math.pi) % (2 * math.pi) - math.pi
            c_fwd = abs(diff)
            c_rev = abs(c_fwd - math.pi)
            
            if c_rev < c_fwd:
                turn_gc = c_rev / self.w_max
                heading_arrival_c = (angles_gc[1] + math.pi) % (2 * math.pi) - math.pi
            else:
                turn_gc = c_fwd / self.w_max
                heading_arrival_c = angles_gc[1]
            
            total = cost_arrival_g0 + turn_gc + d_gc
            layer0[c] = (total, None, heading_arrival_c)
            
        dp.append(layer0)
        
        # --- Layers 1..N: Prev_C -> Goal_i -> Curr_C ---
        for i in range(1, len(tasks)):
            curr_task = tasks[i]
            prev_layer = dp[-1]
            curr_layer = {}
            
            if not prev_layer: return float('inf')
            
            # We iterate all possible Prev_C to find best path to Curr_C
            for prev_c, (prev_cost, _, prev_heading_arr) in prev_layer.items():
                
                # 1. Prev_C -> Goal_i
                d_pg = self.matrix.get(prev_c, {}).get(curr_task.goal_id, float('inf'))
                if d_pg == float('inf'): continue
                
                angles_pg = self.heading_matrix.get(prev_c, {}).get(curr_task.goal_id, (0.0, 0.0))
                target_h = angles_pg[0]
                diff = (target_h - prev_heading_arr + math.pi) % (2 * math.pi) - math.pi
                c_fwd = abs(diff)
                c_rev = abs(c_fwd - math.pi)
                
                if c_rev < c_fwd:
                    turn_pg = c_rev / self.w_max
                    heading_arr_g = (angles_pg[1] + math.pi) % (2 * math.pi) - math.pi
                else:
                    turn_pg = c_fwd / self.w_max
                    heading_arr_g = angles_pg[1]
                
                cost_arr_g = prev_cost + turn_pg + d_pg
                
                # 2. Goal_i -> Curr_C
                for curr_c in self.collections:
                    d_gc = self.matrix.get(curr_task.goal_id, {}).get(curr_c, float('inf'))
                    if d_gc == float('inf'): continue
                    
                    angles_gc = self.heading_matrix.get(curr_task.goal_id, {}).get(curr_c, (0.0, 0.0))
                    target_h = angles_gc[0]
                    diff = (target_h - heading_arr_g + math.pi) % (2 * math.pi) - math.pi
                    c_fwd = abs(diff)
                    c_rev = abs(c_fwd - math.pi)
                    
                    if c_rev < c_fwd:
                        turn_gc = c_rev / self.w_max
                        heading_arr_curr_c = (angles_gc[1] + math.pi) % (2 * math.pi) - math.pi
                    else:
                        turn_gc = c_fwd / self.w_max
                        heading_arr_curr_c = angles_gc[1]
                        
                    total_new_cost = cost_arr_g + turn_gc + d_gc
                    
                    # Store if best
                    if curr_c not in curr_layer or total_new_cost < curr_layer[curr_c][0]:
                        curr_layer[curr_c] = (total_new_cost, prev_c, heading_arr_curr_c)
            
            dp.append(curr_layer)
            
        # --- Backtrack ---
        last_layer = dp[-1]
        if not last_layer: return float('inf')
        
        best_end_c = min(last_layer, key=lambda k: last_layer[k][0])
        min_total_cost = last_layer[best_end_c][0]
        
        curr_c = best_end_c
        for i in range(len(tasks) - 1, -1, -1):
            tasks[i].collection_id = curr_c
            curr_c = dp[i][curr_c][1]
            
        return min_total_cost