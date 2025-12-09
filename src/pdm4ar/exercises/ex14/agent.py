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

class TaskAllocator:
    """
    Solves the Multi-Depot Vehicle Routing Problem.
    Uses Simulated Annealing with Reheating to prevent premature convergence.
    """
    def __init__(self, 
                 cost_matrix: Dict[str, Dict[str, float]], 
                 robots: List[str], 
                 goals: List[str], 
                 collections: List[str]):
        self.matrix = cost_matrix
        self.robots = robots
        self.goals = goals
        self.collections = collections
        
        # Cache best collections for greedy init
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
        """Runs the Simulated Annealing optimization with Reheating."""
        start_time = time.time()
        
        # 1. Initial Solution (Greedy)
        current_solution = self._generate_greedy_solution()
        current_cost = self._evaluate_makespan(current_solution)
        
        # Keep track of the absolute best found across all "reheats"
        best_solution_global = {r: sched.clone() for r, sched in current_solution.items()}
        best_cost_global = current_cost
        
        # Annealing Parameters
        initial_temp = 100.0
        temperature = initial_temp
        cooling_rate = 0.95  # Slightly slower cooling
        min_temp = 0.5       # Threshold to trigger reheat
        
        iterations = 0

        while (time.time() - start_time) < time_limit:
            iterations += 1
            
            # 2. Create Neighbor (Fast Clone)
            # We copy the dict structure, but we only need to deep clone the schedules 
            # that we are about to modify. However, for simplicity/safety, we clone all.
            neighbor_solution = {r: sched.clone() for r, sched in current_solution.items()}
            
            # 3. Mutate
            self._apply_random_mutation(neighbor_solution)
            
            # 4. Evaluate
            neighbor_cost = self._evaluate_makespan(neighbor_solution)
            
            # 5. Acceptance Probability
            delta = neighbor_cost - current_cost
            
            # If better, or lucky roll
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                
                # Update Global Best
                if current_cost < best_cost_global:
                    best_solution_global = {r: sched.clone() for r, sched in current_solution.items()}
                    best_cost_global = current_cost
                    # Optional: Print improvement
                    # print(f"New Best: {best_cost_global:.2f} (Iter {iterations})")
            
            # 6. Cooling & Reheating
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = initial_temp # REHEAT!
                # Optional: Reset search to the best known location?
                # current_solution = {r: sched.clone() for r, sched in best_solution_global.items()}
                # current_cost = best_cost_global
            
        print(f"Allocator: Best Makespan: {best_cost_global:.2f}s | Iterations: {iterations}")
        return {r: sched.tasks for r, sched in best_solution_global.items()}

    def _generate_greedy_solution(self) -> Dict[str, RobotSchedule]:
        """Assigns tasks to the robot that can finish it soonest."""
        schedules = {r: RobotSchedule(r, []) for r in self.robots}
        robot_completion_times = {r: 0.0 for r in self.robots}
        robot_locations = {r: r for r in self.robots} 

        unassigned = list(self.goals)
        
        while unassigned:
            best_r = None
            best_g = None
            best_c = None
            min_added_cost = float('inf')
            
            for g in unassigned:
                c = self.best_collections.get(g)
                if not c: continue
                
                for r in self.robots:
                    curr_loc = robot_locations[r]
                    dist_to_goal = self.matrix.get(curr_loc, {}).get(g, float('inf'))
                    dist_to_coll = self.matrix.get(g, {}).get(c, float('inf'))
                    
                    if dist_to_goal == float('inf') or dist_to_coll == float('inf'):
                        continue

                    new_finish_time = robot_completion_times[r] + dist_to_goal + dist_to_coll
                    
                    if new_finish_time < min_added_cost:
                        min_added_cost = new_finish_time
                        best_r = r
                        best_g = g
                        best_c = c

            if best_r:
                schedules[best_r].tasks.append(DeliveryTask(best_g, best_c))
                robot_completion_times[best_r] = min_added_cost
                robot_locations[best_r] = best_c 
                unassigned.remove(best_g)
            else:
                break 
                
        return schedules

    def _evaluate_makespan(self, solution: Dict[str, RobotSchedule]) -> float:
        """Calculates the time the LAST robot finishes its tasks."""
        max_time = 0.0
        for r_name, schedule in solution.items():
            current_node = r_name 
            total_time = 0.0
            
            for task in schedule.tasks:
                d1 = self.matrix.get(current_node, {}).get(task.goal_id, float('inf'))
                d2 = self.matrix.get(task.goal_id, {}).get(task.collection_id, float('inf'))
                
                # If path is broken, return infinity so this solution is rejected
                if d1 == float('inf') or d2 == float('inf'):
                    return float('inf')
                    
                total_time += (d1 + d2)
                current_node = task.collection_id
            
            if total_time > max_time:
                max_time = total_time
        return max_time

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
        self.robot_radius = 0.8         # Buffer size (robot width/2 + margin)
        self.connection_radius = 10.0   # Max length of an edge
        self.min_sample_dist = 0.3      # Minimum distance between nodes
        self.turn_penalty = 0.0         # Heuristic cost for "stopping and turning" (meters equivalent)

        self.time_limit = 40.0          # Time limit for task allocation

        self.seed = 41

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

        # --- 4. COMPUTE ROUTING DATA (COSTS & PATHS) ---
        # Returns cost matrix AND a structured dictionary of paths for plotting
        cost_matrix, path_data = self._compute_routing_data(G)
        print(f"Computed Cost Matrix for {len(cost_matrix)} nodes.")

        # --- 5. TASK ALLOCATION ---
        allocator = TaskAllocator(cost_matrix, robots_list, goals_list, collections_list)
        assignments = allocator.solve(self.time_limit)
        
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

    # def _compute_cost_matrix(self, G):
    #     """
    #     Computes APSP for POIs and returns matrix + one debug path.
    #     """
    #     poi_indices = []
    #     starts = []
    #     goals = []
        
    #     for n, data in G.nodes(data=True):
    #         if data.get('type') in ['start', 'goal', 'collection']:
    #             poi_indices.append(n)
    #             if data.get('type') == 'start': starts.append(n)
    #             if data.get('type') == 'goal': goals.append(n)
        
    #     matrix = {}
    #     debug_path = []
        
    #     # Calculate Matrix
    #     for source in poi_indices:
    #         matrix[source] = {}
    #         try:
    #             lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
    #             for target in poi_indices:
    #                 matrix[source][target] = lengths.get(target, float('inf'))
    #         except Exception:
    #             pass

    #     # Calculate One Debug Path (Start[0] -> Goal[0])
    #     if starts and goals:
    #         try:
    #             # Use dijkstra_path to get the actual nodes
    #             path_nodes = nx.dijkstra_path(G, starts[0], goals[0], weight='weight')
    #             pos = nx.get_node_attributes(G, 'pos')
    #             debug_path = [pos[n] for n in path_nodes]
    #         except nx.NetworkXNoPath:
    #             pass
                
    #     return matrix, debug_path

    def _compute_routing_data(self, G) -> Tuple[dict, dict]:
        """
        Computes APSP for POIs and extracts ALL paths.
        Returns:
            1. cost_matrix: {SourceLabel: {TargetLabel: Cost}}
            2. path_data: { "starts": ..., "goals": ..., "collections": ... }
        """
        cost_matrix = {}
        path_data = {"starts": {}, "goals": {}, "collections": {}}
        
        pos = nx.get_node_attributes(G, 'pos')
        
        # 1. Group Nodes
        starts = [(n, d.get('label')) for n, d in G.nodes(data=True) if d.get('type') == 'start']
        goals = [(n, d.get('label')) for n, d in G.nodes(data=True) if d.get('type') == 'goal']
        collections = [(n, d.get('label')) for n, d in G.nodes(data=True) if d.get('type') == 'collection']

        # Helper to process a group
        def process_group(source_list, target_list, category_key):
            for src_idx, src_label in source_list:
                if src_label not in cost_matrix: cost_matrix[src_label] = {}
                if src_label not in path_data[category_key]: path_data[category_key][src_label] = {}
                
                best_target_label = None
                min_cost = float('inf')
                temp_results = {}

                for tgt_idx, tgt_label in target_list:
                    try:
                        cost = nx.shortest_path_length(G, src_idx, tgt_idx, weight='weight')
                        path_nodes = nx.shortest_path(G, src_idx, tgt_idx, weight='weight')
                        coords = [pos[n] for n in path_nodes]
                        
                        temp_results[tgt_label] = {'cost': cost, 'coords': coords}
                        
                        if cost < min_cost:
                            min_cost = cost
                            best_target_label = tgt_label
                            
                    except nx.NetworkXNoPath:
                        temp_results[tgt_label] = {'cost': float('inf'), 'coords': None}

                for tgt_label, res in temp_results.items():
                    cost_matrix[src_label][tgt_label] = res['cost']
                    if res['coords']:
                        is_best = (tgt_label == best_target_label)
                        path_data[category_key][src_label][tgt_label] = {
                            'coords': res['coords'],
                            'is_best': is_best
                        }

        # 2. Compute Robot -> Goals
        process_group(starts, goals, "starts")

        # 3. Compute Goal -> Collections
        process_group(goals, collections, "goals")
        
        # 4. Compute Collection -> Goals (For multi-step missions)
        process_group(collections, goals, "collections")

        return cost_matrix, path_data

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