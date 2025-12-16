import random
import datetime
import copy
import math
import csv # [NEW]
from re import A
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
from shapely.strtree import STRtree  # For fast collision detection

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

from .task_allocator import (
    DeliveryTask,
    RobotSchedule,
    TaskAllocatorSA,
    TaskAllocatorLNS,
    TaskAllocatorLNS2,
    TaskAllocatorLNS3,
    TaskAllocatorALNS,
)
from .spacetime_planner import SpaceTimeRoadmapPlanner
from .exact_spacetime_planner import ExactSpaceTimePlanner, PlanPoint
from dg_commons.sim.models.diff_drive import DiffDriveState
from .tournament_viz import TournamentVisualizer 
from .planner_viz import PlannerVisualizer # [NEW]


class GlobalPlanMessage(BaseModel):
    # TODO: modify/add here the fields you need to send your global plan
    # fake_id: int
    # fake_name: str
    # fake_np_data: NDArray # If you need to send numpy arrays, annotate them with NDArray
    # paths: Mapping[str, Sequence[Tuple[float, float]]]
    # [NEW] Updated to support 6D trajectory: (x, y, theta, t, v, w)
    paths: Mapping[str, Sequence[Tuple[float, float, float, float, float, float]]]


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 10
    # [NEW] Gains
    k_x: float = 1.0
    k_theta: float = 2.0


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
        # [NEW] 6D Trajectory
        self.my_global_path: List[Tuple[float, float, float, float, float, float]] = []
        self.current_path_idx = 0
        self._pending_global_plan_msg = None
        
        # Initialize defaults (will be overwritten in on_episode_init)
        self.sg = DiffDriveGeometry.default()
        self.sp = DiffDriveParameters.default()

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        # [FIX] Get our identity and correct physics from the simulator
        self.name = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        
        self.current_path_idx = 0
        
        # [NEW] Setup Logging
        # self.log_file = Path(f"out/ex14/debug_plots/cmd_log_{self.name}.csv")
        # self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.prev_state = None
        self.prev_time = None
        
        # with open(self.log_file, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["t", "omega_l", "omega_r", "v_ref", "w_ref", "v_cmd", "w_cmd", "x", "y", "theta", "rx", "ry", "rtheta", "v_act", "w_act"])
        
        # Process the plan now that we know who we are
        if self._pending_global_plan_msg:
            self._process_global_plan(self._pending_global_plan_msg)
            self._pending_global_plan_msg = None

    def _process_global_plan(self, serialized_msg: str):
        global_plan = GlobalPlanMessage.model_validate_json(serialized_msg)
        if hasattr(self, 'name') and self.name in global_plan.paths:
             self.my_global_path = list(global_plan.paths[self.name])
        else:
             self.my_global_path = []
        self.current_path_idx = 0

    def on_receive_global_plan(
        self,
        serialized_msg: str,
    ):
        # If we already know our name, process immediately
        if hasattr(self, 'name'):
            self._process_global_plan(serialized_msg)
        else:
            # Otherwise, wait until on_episode_init
            self._pending_global_plan_msg = serialized_msg

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        """
        Pure Feedforward Controller (Player Piano).
        Executes the exact FF plan step-by-step based on grid time.
        """
        # 1. Safety Check: If no plan, do nothing
        if not self.my_global_path:
            return DiffDriveCommands(omega_l=0.0, omega_r=0.0)

        current_time = float(sim_obs.time)
        dt = 0.1
        
        # 2. Determine Time Step (THE FIX)
        # We use INT (floor) to stay in the current step for the full 0.1s.
        # We add a tiny epsilon (1e-5) to handle floating point imprecision 
        # (e.g., if t=0.1999999, we want index 1).
        step_idx = int((current_time + 1e-5) / dt)
        
        # 3. Look Ahead
        # The plan at index 'i' tells us the state at t=(i+1)*dt and the command used to GET there
        # (i.e., the command for the interval [i*dt, (i+1)*dt]).
        # So for the current time t (falling in step_idx), we need the command stored at index 'step_idx'.
        lookahead_idx = step_idx

        v_cmd = 0.0
        w_cmd = 0.0
        
        # Debugging variables for logging
        rx, ry, rtheta = 0.0, 0.0, 0.0

        if lookahead_idx < len(self.my_global_path):
            next_pt = self.my_global_path[lookahead_idx]
            
            # Tuple structure: (x, y, theta, t, v, w)
            rx, ry, rtheta = next_pt[0], next_pt[1], next_pt[2]
            v_cmd = next_pt[4]
            w_cmd = next_pt[5]
        else:
            # End of path reached or exceeded
            if self.my_global_path:
                last = self.my_global_path[-1]
                rx, ry, rtheta = last[0], last[1], last[2]
            v_cmd = 0.0
            w_cmd = 0.0

        # 4. Inverse Kinematics (Convert v, w -> omega_l, omega_r)
        r = self.sg.wheelradius
        L = self.sg.wheelbase
        
        # Standard differential drive equations:
        # v = r/2 * (wr + wl)
        # w = r/L * (wr - wl)
        omega_r = (v_cmd + (L / 2) * w_cmd) / r
        omega_l = (v_cmd - (L / 2) * w_cmd) / r
        
        # --- 5. LOGGING & METRICS (Optional but recommended) ---
        current_state = sim_obs.players[self.name].state
        x, y, theta = current_state.x, current_state.y, current_state.psi
        
        # Calculate Actual Velocities (Numerical Differentiation)
        v_act, w_act = 0.0, 0.0
        if self.prev_state is not None:
            dt_sim = current_time - self.prev_time
            if dt_sim > 1e-6:
                dx_act = x - self.prev_state[0]
                dy_act = y - self.prev_state[1]
                dtheta_act = (theta - self.prev_state[2] + math.pi) % (2*math.pi) - math.pi
                
                v_act = math.sqrt(dx_act**2 + dy_act**2) / dt_sim
                move_angle = math.atan2(dy_act, dx_act)
                
                # Check for reverse motion to sign v_act correctly
                if abs(v_act) > 0.01:
                    heading_diff = (move_angle - self.prev_state[2] + math.pi) % (2*math.pi) - math.pi
                    if abs(heading_diff) > math.pi/2:
                        v_act = -v_act
                w_act = dtheta_act / dt_sim

        self.prev_state = (x, y, theta)
        self.prev_time = current_time
        
        # Write to CSV
        # try:
        #     with open(self.log_file, 'a', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow([
        #             current_time, 
        #             omega_l, omega_r, 
        #             v_cmd, w_cmd,       # Reference commands
        #             v_cmd, w_cmd,       # (Duplicate for compatibility with your header)
        #             x, y, theta,        # Actual State
        #             rx, ry, rtheta,     # Reference State
        #             v_act, w_act        # Actual Velocities
        #         ])
        # except Exception:
        #     pass 

        return DiffDriveCommands(omega_l=omega_l, omega_r=omega_r)


class Pdm4arGlobalPlanner(GlobalPlanner):
    """
    This is the Global Planner for PDM4AR
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task
    """

    def __init__(self):
        # Parameters
        self.num_samples = 2000  # Can handle more samples now # 2000
        self.target_degree = 20  # We WANT this many connections per node
        self.max_candidates = 50  # We CHECK this many to find the valid ones (handles deleted vertices)
        self.robot_radius = 0.6 + 0.1  # Buffer size (robot width/2 + margin)
        self.connection_radius = 10.0  # Max length of an edge
        self.min_sample_dist = 0.3  # Minimum distance between nodes # 0.3
        self.turn_penalty = 0.0  # Heuristic cost for "stopping and turning" (meters equivalent)

        self.time_limit = 5.0  # Time limit for task allocation # 10.0

        self.seed = 42

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # [DEBUG] Inspect available global observations
        # print(f"DEBUG: GlobalPlanner init_sim_obs dir: {dir(init_sim_obs)}")

        # --- 1. EXTRACT OBSTACLES & BOUNDS (Done once) ---
        obs_polygons = []
        if init_sim_obs.dg_scenario and init_sim_obs.dg_scenario.static_obstacles:
            for obs in init_sim_obs.dg_scenario.static_obstacles:
                if hasattr(obs, "shape"):
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
        initial_nodes_data = []  # List of (x, y, type, label)

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
                if hasattr(sgoal, "polygon"):
                    c = sgoal.polygon.centroid
                    special_nodes_plot["goals"].append((c.x, c.y))
                    initial_nodes_data.append((c.x, c.y, "goal", gid))
                    goals_list.append(gid)

        # Collection Points
        collections_list = []
        if init_sim_obs.collection_points:
            for cid, cpoint in init_sim_obs.collection_points.items():
                if hasattr(cpoint, "polygon"):
                    c = cpoint.polygon.centroid
                    special_nodes_plot["collections"].append((c.x, c.y))
                    initial_nodes_data.append((c.x, c.y, "collection", cid))
                    collections_list.append(cid)

        # --- 3. BUILD PRM ---
        G = self._build_prm(inflated_obstacles, initial_nodes_data, bounds)

        # 4. Get Initial Headings
        initial_headings = {}
        for name, state in init_sim_obs.initial_states.items():
            initial_headings[name] = state.psi

        # 5. Get Kinematics for Allocator
        # [FIX] Extract dynamic parameters from the first player observation
        first_player_obs = next(iter(init_sim_obs.players_obs.values()))
        sg = first_player_obs.model_geometry
        sp = first_player_obs.model_params
        _, w_max = self._get_kinematic_limits(sg, sp)

        # --- Create STRtree for Smoothing ---
        obstacle_tree = STRtree(inflated_obstacles)

        # --- 6. COMPUTE ROUTING DATA (COSTS & PATHS) ---
        # [MODIFIED] Now passing obstacle data for smoothing AND physics parameters
        cost_matrix, path_data, heading_matrix = self._compute_routing_data(G, obstacle_tree, inflated_obstacles, sg, sp)
        print(f"Computed Cost Matrix for {len(cost_matrix)} nodes.")

        # 7. Initialize Allocator ARGS
        alloc_args = {
            "cost_matrix": cost_matrix,
            "heading_matrix": heading_matrix,
            "initial_headings": initial_headings,
            "w_max": w_max,
            "robots": robots_list,
            "goals": goals_list,
            "collections": collections_list,
        }

        # Setup Output
        out_dir = Path("out/ex14/debug_plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # A. Run SA
        # allocator_sa = TaskAllocatorSA(**alloc_args)
        # sa_assignments, sa_hist = allocator_sa.solve(time_limit=self.time_limit)
        # sa_cost = allocator_sa._evaluate_makespan({r: RobotSchedule(r, t) for r, t in sa_assignments.items()})

        # B. Run LNS
        # allocator_lns = TaskAllocatorLNS(**alloc_args)
        # lns_assignments, lns_hist = allocator_lns.solve(time_limit=self.time_limit)
        # lns_cost = allocator_lns._evaluate_makespan({r: RobotSchedule(r, t) for r, t in lns_assignments.items()})

        # C. Run LNS2
        # allocator_lns2 = TaskAllocatorLNS2(**alloc_args)
        # lns2_assignments, lns2_hist = allocator_lns2.solve(time_limit=self.time_limit)
        # lns2_cost = allocator_lns2._evaluate_makespan({r: RobotSchedule(r, t) for r, t in lns2_assignments.items()})

        # D. Run LNS3
        # allocator_lns3 = TaskAllocatorLNS3(**alloc_args)
        # lns3_assignments, lns3_hist = allocator_lns3.solve(time_limit=self.time_limit)
        # lns3_cost = allocator_lns3._evaluate_makespan({r: RobotSchedule(r, t) for r, t in lns3_assignments.items()})

        # E. Run ALNS (Adaptive)
        allocator_alns = TaskAllocatorALNS(**alloc_args)
        alns_assignments, alns_telemetry, alns_top_k = allocator_alns.solve(time_limit=self.time_limit)
        alns_cost = allocator_alns._evaluate_makespan({r: RobotSchedule(r, t) for r, t in alns_assignments.items()})

        # Telemetry Processing
        alns_hist = [(entry["time"], entry["best_cost"]) for entry in alns_telemetry]

        import json

        # telemetry_file = out_dir / f"alns_telemetry_{timestamp}.json"
        # with open(telemetry_file, "w") as f:
        #     json.dump(alns_telemetry, f, indent=2)
        
        # [NEW] Plot Convergence
        viz_helper = PlannerVisualizer(robot_radius=self.robot_radius)
        viz_helper.plot_convergence(
            {"ALNS": alns_hist}, 
            str(out_dir / f"convergence_{timestamp}.png")
        )

        # --- ALLOCATOR COMPARISON & SELECTION ---
        print(f"--- RESULT COMPARISON (Theoretical) ---")
        # print(f"SA Cost:  {sa_cost:.2f}")
        # print(f"LNS Cost: {lns_cost:.2f}")
        # print(f"LNS2 Cost: {lns2_cost:.2f}")
        # print(f"LNS3 Cost: {lns3_cost:.2f}")
        print(f"ALNS Cost: {alns_cost:.2f}")

        # Candidates to evaluate
        candidates = [
            # ("SA", sa_assignments, sa_cost),
            # ("LNS", lns_assignments, lns_cost),
            # ("LNS2", lns2_assignments, lns2_cost),
            # ("LNS3", lns3_assignments, lns3_cost),
        ]
        
        # Add Top K ALNS candidates
        print("ALNS TOP Solutions:")
        for i, sol in enumerate(alns_top_k):
            cost = allocator_alns._evaluate_makespan({r: RobotSchedule(r, t) for r, t in sol.items()})
            print(f"  {i+1}. Cost: {cost:.2f}")
            candidates.append((f"ALNS_{i+1}", sol, cost))

        # --- SELECT WINNER (PHYSICS TOURNAMENT) ---
        # Instead of trusting the theoretical cost, we run the physics simulator 
        # on the Top Candidates to see which one ACTUALLY executes fastest.
        
        # 1. Sort candidates by theoretical cost
        candidates.sort(key=lambda x: x[2])
        
        # 2. Pick Top 5 Unique Assignments to test
        # (Deduplication based on cost is a simple proxy, or just take top 5)
        top_candidates = candidates[:10]
        
        print(f"\n>> SELECTED TOP {len(top_candidates)} CANDIDATES FOR PHYSICS EVALUATION:")
        for name, _, cost in top_candidates:
            print(f"   - {name}: {cost:.2f} (theoretical)")

        best_actual_makespan = float('inf')
        best_final_plans_6d = {}
        winner_name = "None"
        best_waypoints = {} # For plotting

        # We need static obstacles for the planner
        static_obs_polys = []
        if init_sim_obs.dg_scenario and init_sim_obs.dg_scenario.static_obstacles:
            static_obs_polys = [o.shape for o in init_sim_obs.dg_scenario.static_obstacles]

        # 3. Tournament Loop
        viz = TournamentVisualizer() # [NEW]
        viz_helper = PlannerVisualizer(robot_radius=self.robot_radius) # [NEW]
        
        for cand_name, cand_assign, cand_theo_cost in top_candidates:
            print(f"\n   [Evaluating {cand_name}] ...")
            
            # --- A. Prepare Waypoints ---
            robot_waypoints = {}
            geometries = {}
            params = {}
            initial_states_obj = {}
            
            # Track where the "real work" ends for each robot
            robot_last_task_idx = {} 

            for r_name, tasks in cand_assign.items():
                # Physics Data
                p_obs = init_sim_obs.players_obs[r_name]
                geometries[r_name] = p_obs.model_geometry
                params[r_name] = p_obs.model_params
                
                # Clean State
                raw_s = init_sim_obs.initial_states[r_name]
                initial_states_obj[r_name] = DiffDriveState(
                    x=float(raw_s.x), y=float(raw_s.y), psi=float(raw_s.psi)
                )

                # Waypoints
                wps = []
                curr_node = r_name 
                for task in tasks:
                    # Path -> Goal
                    seg = self._find_path_coords_raw(path_data, curr_node, task.goal_id)
                    if seg: wps.extend(seg[1:])
                    curr_node = task.goal_id
                    
                    # Path -> Collection
                    seg = self._find_path_coords_raw(path_data, curr_node, task.collection_id)
                    if seg: wps.extend(seg[1:])
                    curr_node = task.collection_id
                
                # [NEW] Mark the end of tasks
                robot_last_task_idx[r_name] = len(wps)
                
                # Return to Start
                seg = self._find_path_coords_raw(path_data, curr_node, r_name)
                if seg: wps.extend(seg[1:])
                
                robot_waypoints[r_name] = wps

            # --- B. Calculate Priority (Longest Delivery First) ---
            robot_priority_list = []
            for r_name, tasks in cand_assign.items():
                delivery_duration = 0.0
                curr_node = r_name
                for task in tasks:
                    t_goal = cost_matrix.get(curr_node, {}).get(task.goal_id, 1000.0)
                    delivery_duration += t_goal
                    t_coll = cost_matrix.get(task.goal_id, {}).get(task.collection_id, 1000.0)
                    delivery_duration += t_coll
                    curr_node = task.collection_id
                robot_priority_list.append((r_name, delivery_duration))
            
            robot_priority_list.sort(key=lambda x: x[1], reverse=True)
            sorted_robots = [r for r, c in robot_priority_list]
            # print(f"     Priority: {sorted_robots}")

            # --- C. Run Exact Planner ---
            # Instantiate fresh planner for each candidate
            exact_planner = ExactSpaceTimePlanner(static_obstacles=static_obs_polys, dt=0.1, use_stagnation_logic=False)
            
            plan_start_t = time.time()
            plans_6d = exact_planner.plan_prioritized(
                robots_sequence=sorted_robots,
                initial_states=initial_states_obj,
                waypoints_dict=robot_waypoints,
                geometries=geometries,
                params=params
            )
            plan_dur = time.time() - plan_start_t
            
            # --- D. Measure Actual Makespan (Delivery Only) & Validity ---
            actual_makespan = 0.0
            is_valid_candidate = True 
            
            # 1. Check Task Completion
            for r_name, r_traj in plans_6d.items():
                if not r_traj: 
                    is_valid_candidate = False
                    continue
                
                limit_idx = robot_last_task_idx.get(r_name, 0)
                delivery_t = 0.0
                found = False
                
                if r_traj[-1].target_idx < limit_idx:
                    is_valid_candidate = False
                
                for pt in r_traj:
                    if pt.target_idx >= limit_idx:
                        delivery_t = pt.t
                        found = True
                        break
                
                if not found:
                    delivery_t = r_traj[-1].t
                    
                actual_makespan = max(actual_makespan, delivery_t)
            
            # 2. Gather Stats & Check MAX_ITERS
            total_backtracks = 0
            total_collisions = 0
            total_iterations = 0 
            per_robot_times = []
            per_robot_iters = []
            
            if exact_planner.debugger and exact_planner.debugger.logs:
                for r_name in sorted_robots:
                    r_log = exact_planner.debugger.logs.get(r_name, {})
                    
                    total_backtracks += len(r_log.get('backtracks', []))
                    total_collisions += len(r_log.get('collisions', []))
                    
                    iters_list = r_log.get('iterations', [])
                    it = iters_list[-1] if iters_list else 0
                    t = max(0.0, r_log.get('wall_time', 0.0))
                    
                    if it >= exact_planner.MAX_ITERS:
                        is_valid_candidate = False
                        print(f"     [Fail] Robot {r_name} hit MAX_ITERS ({it})")
                    
                    total_iterations += it
                    per_robot_times.append(t)
                    per_robot_iters.append(it)
            
            # 3. Record & Print Result (AFTER Validity Update)
            viz.record_result(
                cand_name, cand_theo_cost, actual_makespan, plan_dur, total_backtracks, total_collisions, total_iterations,
                per_robot_times, per_robot_iters, is_valid=is_valid_candidate
            )
            
            status_str = "VALID" if is_valid_candidate else "INVALID"
            print(f"   -> Result {cand_name}: Theo={cand_theo_cost:.2f}s | Actual={actual_makespan:.2f}s | Status={status_str}")
            
            # 4. Update Winner
            if 'best_is_valid' not in locals(): best_is_valid = False
            better_found = False
            
            if is_valid_candidate and not best_is_valid:
                better_found = True
            elif is_valid_candidate and best_is_valid:
                if actual_makespan < best_actual_makespan:
                    better_found = True
            elif not is_valid_candidate and not best_is_valid:
                if actual_makespan < best_actual_makespan:
                    better_found = True
            
            if better_found:
                best_actual_makespan = actual_makespan
                best_final_plans_6d = plans_6d
                winner_name = cand_name
                best_waypoints = robot_waypoints
                best_is_valid = is_valid_candidate
        
        # Plot Summary
        viz.plot_all()

        # ---------------------------------------------------------------------
        # --- 4. Finalize Winner ---
        # ---------------------------------------------------------------------
        print(f"\n>> TOURNAMENT WINNER: {winner_name} (Time: {best_actual_makespan:.2f}s)")
        
        # Convert to Message Format
        paths_output_6d = {}
        for r_name, plan_points in best_final_plans_6d.items():
            paths_output_6d[r_name] = [(p.x, p.y, p.theta, p.t, p.v, p.w) for p in plan_points]

        global_plan_message = GlobalPlanMessage(paths=paths_output_6d)
        
        # Plotting
        paths_output_xy_plot = {r: [(p[0], p[1]) for p in traj] for r, traj in paths_output_6d.items()}
        filename_prm = out_dir / f"prm_debug_{timestamp}_{winner_name}.png"
        viz_helper.plot_prm(
            G, obs_polygons, special_nodes_plot, str(filename_prm), bounds, path_data, final_paths=paths_output_xy_plot
        )
        viz_helper.plot_trajectory_comparison(
            waypoints_dict=best_waypoints,
            final_plans_6d=best_final_plans_6d,
            obstacles=static_obs_polys,
            filename=str(out_dir / f"traj_debug_{timestamp}.png")
        )

        return global_plan_message.model_dump_json(round_trip=True)
    def _find_path_coords(self, path_data, src, dst):
        """Helper to find path coordinates from any bucket"""
        for cat in path_data.values():
            if src in cat and dst in cat[src]:
                # [NEW] Densify just before returning for the final plan
                raw_coords = cat[src][dst]["coords"]
                return self._densify_path(raw_coords, step=0.05)
        return []

    def _densify_path(self, coords, step=0.1):
        """Injects points into sparse path for controller stability."""
        if not coords or len(coords) < 2:
            return coords
        new_coords = [coords[0]]
        for i in range(len(coords) - 1):
            p1 = np.array(coords[i])
            p2 = np.array(coords[i + 1])
            dist = np.linalg.norm(p2 - p1)
            if dist > step:
                num_points = int(dist / step)
                for j in range(1, num_points + 1):
                    new_coords.append(tuple(p1 + (p2 - p1) * (j / (num_points + 1))))
            new_coords.append(coords[i + 1])
        return new_coords

    def _compute_routing_data(self, G, obstacle_tree, inflated_obstacles, sg, sp) -> Tuple[dict, dict]:
        """
        Computes APSP for POIs and extracts ALL paths.
        Now includes POST-PROCESS SMOOTHING and GOAL AVOIDANCE.
        """
        cost_matrix = {}
        heading_matrix = {}
        path_data = {"starts": {}, "goals": {}, "collections": {}}
        pos = nx.get_node_attributes(G, "pos")

        # --- [DEBUG PRINT HERE] ---
        v_debug, w_debug = self._get_kinematic_limits(sg, sp)
        print(f"DEBUG KINEMATICS: V_max = {v_debug:.3f} m/s | Omega_max = {w_debug:.3f} rad/s")
        # --------------------------

        # --- [NEW] PRE-CALCULATE GOAL CONFLICTS ---
        # 1. Identify Goal Positions
        goal_positions = {}
        for n, d in G.nodes(data=True):
            if d.get("type") == "goal":
                goal_positions[d.get("label")] = pos[n]
        
        # 2. Map Edges to Conflicting Goals
        # edge_conflicts[ (u,v) ] = { 'goal_id_1', 'goal_id_2' }
        edge_conflicts = {}
        
        GOAL_AVOID_RADIUS = 0.35 # 0.3m + 0.05m margin

        def point_line_segment_distance(px, py, x1, y1, x2, y2):
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                return math.hypot(px - x1, py - y1)
            t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
            t = max(0, min(1, t))
            nearest_x = x1 + t * dx
            nearest_y = y1 + t * dy
            return math.hypot(px - nearest_x, py - nearest_y)

        print(f"Pre-calculating goal conflicts for {len(G.edges)} edges...")
        for u, v in G.edges():
            p1 = pos[u]
            p2 = pos[v]
            
            conflicts = set()
            for g_label, g_pos in goal_positions.items():
                dist = point_line_segment_distance(g_pos[0], g_pos[1], p1[0], p1[1], p2[0], p2[1])
                if dist < GOAL_AVOID_RADIUS:
                    conflicts.add(g_label)
            
            if conflicts:
                key = tuple(sorted((u, v)))
                edge_conflicts[key] = conflicts
        # ------------------------------------------

        starts = [(n, d.get("label")) for n, d in G.nodes(data=True) if d.get("type") == "start"]
        goals = [(n, d.get("label")) for n, d in G.nodes(data=True) if d.get("type") == "goal"]
        collections = [(n, d.get("label")) for n, d in G.nodes(data=True) if d.get("type") == "collection"]

        def process_group(source_list, target_list, category_key):
            for src_idx, src_label in source_list:
                if src_label not in cost_matrix:
                    cost_matrix[src_label] = {}
                if src_label not in heading_matrix:
                    heading_matrix[src_label] = {}
                if src_label not in path_data[category_key]:
                    path_data[category_key][src_label] = {}

                # Temp storage to find best target based on TIME, not DISTANCE
                candidates = []

                # --- [NEW] Define Forbidden Goals for this Source ---
                # We must avoid all goals EXCEPT the one we are at (Src) or going to (Tgt)
                base_forbidden = set(goal_positions.keys())
                if src_label in base_forbidden:
                    base_forbidden.remove(src_label)
                # Note: Tgt is handled inside the inner loop
                
                for tgt_idx, tgt_label in target_list:
                    try:
                        # Determine Forbidden Set for this specific path
                        current_forbidden = base_forbidden.copy()
                        if tgt_label in current_forbidden:
                            current_forbidden.remove(tgt_label)
                        
                        # Custom Weight Function
                        def weight_fn(u, v, d):
                            key = tuple(sorted((u, v)))
                            conflicts = edge_conflicts.get(key, set())
                            # If edge conflicts with any FORBIDDEN goal, block it (inf cost)
                            if not conflicts.isdisjoint(current_forbidden):
                                return float('inf')
                            return d.get('weight', 1.0)

                        # 1. Get Shortest Path with Goal Avoidance
                        path_nodes = nx.shortest_path(G, src_idx, tgt_idx, weight=weight_fn)
                        raw_coords = [pos[n] for n in path_nodes]

                        # [NEW] SMOOTH PATH
                        smoothed_coords = self._smooth_path(raw_coords, obstacle_tree, inflated_obstacles)

                        # 2. Calculate Cost by DURATION (Time)
                        # We use the SMOOTHED coords for cost calculation!
                        duration = self._calculate_path_duration(smoothed_coords, sg, sp)

                        # --- NEW: CALCULATE HEADINGS ---
                        s_angle = 0.0
                        e_angle = 0.0
                        if len(smoothed_coords) >= 2:
                            # Heading of the first segment
                            s_angle = math.atan2(
                                smoothed_coords[1][1] - smoothed_coords[0][1],
                                smoothed_coords[1][0] - smoothed_coords[0][0],
                            )
                            # Heading of the last segment
                            e_angle = math.atan2(
                                smoothed_coords[-1][1] - smoothed_coords[-2][1],
                                smoothed_coords[-1][0] - smoothed_coords[-2][0],
                            )
                        # -------------------------------

                        candidates.append((duration, tgt_label, smoothed_coords))

                        # Store in matrix
                        cost_matrix[src_label][tgt_label] = duration
                        heading_matrix[src_label][tgt_label] = (s_angle, e_angle)

                    except nx.NetworkXNoPath:
                        cost_matrix[src_label][tgt_label] = float("inf")
                        heading_matrix[src_label][tgt_label] = (0.0, 0.0)

                # 3. Find which target was the "best" (fastest) and mark it
                if candidates:
                    candidates.sort(key=lambda x: x[0])  # Sort by duration
                    best_label = candidates[0][1]

                    for cost, label, coords in candidates:
                        path_data[category_key][src_label][label] = {"coords": coords, "is_best": (label == best_label)}

        # 2. Compute Robot -> Goals
        process_group(starts, goals, "starts")
        # 3. Compute Goal -> Collections
        process_group(goals, collections, "goals")
        # 4. Compute Collection -> Goals (For multi-step missions)
        process_group(collections, goals, "collections")
        # 5. Compute Collection -> Starts (For Return-to-Base)
        process_group(collections, starts, "collections")

        return cost_matrix, path_data, heading_matrix

    def _smooth_path(self, coords, obstacle_tree, inflated_obstacles):
        """
        Greedy Shortcutting:
        Iterate from Start. Try to connect to the furthest possible node in the sequence
        that is visible (collision-free).
        """
        if len(coords) < 3:
            return coords

        smoothed = [coords[0]]
        current_idx = 0

        while current_idx < len(coords) - 1:
            # Look ahead from end to current+1
            best_next_idx = current_idx + 1

            # Check indices from End down to Current+2
            # We want the FURTHEST reachable node
            for check_idx in range(len(coords) - 1, current_idx + 1, -1):

                # Check line segment
                p1 = coords[current_idx]
                p2 = coords[check_idx]
                line = LineString([p1, p2])

                # Fast AABB check
                possible_obs = obstacle_tree.query(line)
                is_colliding = False
                for idx in possible_obs:
                    if inflated_obstacles[idx].intersects(line):
                        is_colliding = True
                        break

                if not is_colliding:
                    best_next_idx = check_idx
                    break  # Found the furthest one

            smoothed.append(coords[best_next_idx])
            current_idx = best_next_idx

        return smoothed
    
    def _find_path_coords_raw(self, path_data, src, dst):
        """
        Helper to find the geometric path between two nodes.
        Checks all categories ('starts', 'goals', 'collections') to find the segment.
        """
        for cat in ["starts", "goals", "collections"]:
            # Check if src exists in this category and if dst is a target of src
            if src in path_data[cat] and dst in path_data[cat][src]:
                return path_data[cat][src][dst]["coords"]
        return []

    def _get_kinematic_limits(self, sg: DiffDriveGeometry, sp: DiffDriveParameters) -> Tuple[float, float]:
        """Derives v_max [m/s] and omega_max [rad/s] from robot structures."""
        # Max wheel rotation (rad/s)
        w_wheel_max = max(abs(sp.omega_limits[0]), abs(sp.omega_limits[1]))

        # V_max = r * omega_wheel
        v_max = sg.wheelradius * w_wheel_max

        # Omega_max = (2 * r * omega_wheel) / L
        omega_max = (2 * sg.wheelradius * w_wheel_max) / sg.wheelbase
        return v_max, omega_max

    def _calculate_path_duration(self, coords: List[Tuple[float, float]], sg, sp) -> float:
        """Calculates accurate duration using robot kinematics."""
        if not coords or len(coords) < 2:
            return 0.0

        v_max, w_max = self._get_kinematic_limits(sg, sp)

        # Safety clamp
        if v_max < 1e-4:
            v_max = 0.1
        if w_max < 1e-4:
            w_max = 0.1

        total_time = 0.0
        current_heading = None

        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.sqrt(dx**2 + dy**2)

            # 1. Linear Time
            total_time += dist / v_max

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
        for x, y, n_type, label in initial_nodes_data:
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
            if count >= self.num_samples:
                break

            # Grid Check
            gx = int(x / self.min_sample_dist)
            gy = int(y / self.min_sample_dist)
            if (gx, gy) in occupied_grids:
                continue

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
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 1e-6:
                        nudge = 0.1
                        final_point = Point(nearest.x + (dx / dist) * nudge, nearest.y + (dy / dist) * nudge)
                    else:
                        final_point = nearest
                except Exception:
                    continue

            if final_point:
                fgx = int(final_point.x / self.min_sample_dist)
                fgy = int(final_point.y / self.min_sample_dist)
                if (fgx, fgy) in occupied_grids:
                    continue

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
                    if i == j_idx:
                        continue
                    if edges_added >= self.target_degree:
                        break
                    if d > self.connection_radius:
                        break

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
