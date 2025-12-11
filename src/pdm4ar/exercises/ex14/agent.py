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

from .task_allocator import DeliveryTask, RobotSchedule, TaskAllocatorSA, TaskAllocatorLNS, TaskAllocatorLNS2, TaskAllocatorLNS3


class GlobalPlanMessage(BaseModel):
    # TODO: modify/add here the fields you need to send your global plan
    # fake_id: int
    # fake_name: str
    # fake_np_data: NDArray # If you need to send numpy arrays, annotate them with NDArray
    paths: Mapping[str, Sequence[Tuple[float, float]]]


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 10


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
        self.num_samples = 2000  # Can handle more samples now # 2000
        self.target_degree = 20  # We WANT this many connections per node
        self.max_candidates = 50  # We CHECK this many to find the valid ones (handles deleted vertices)
        self.robot_radius = 0.6 + 0.1  # Buffer size (robot width/2 + margin)
        self.connection_radius = 10.0  # Max length of an edge
        self.min_sample_dist = 0.3  # Minimum distance between nodes # 0.3
        self.turn_penalty = 0.0  # Heuristic cost for "stopping and turning" (meters equivalent)

        self.time_limit = 10.0  # Time limit for task allocation

        self.seed = 42

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
        random.seed(self.seed)
        np.random.seed(self.seed)

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

        # 4. Initialize Allocator ARGS
        alloc_args = {
            "cost_matrix": cost_matrix,
            "heading_matrix": heading_matrix,
            "initial_headings": initial_headings,
            "w_max": w_max,
            "robots": robots_list,
            "goals": goals_list,
            "collections": collections_list,
        }

        # A. Run SA
        allocator_sa = TaskAllocatorSA(**alloc_args)
        # Give SA 30% of time or a fixed amount
        sa_assignments = allocator_sa.solve(time_limit=self.time_limit)
        sa_cost = allocator_sa._evaluate_makespan({r: RobotSchedule(r, t) for r, t in sa_assignments.items()})

        # B. Run LNS
        allocator_lns = TaskAllocatorLNS(**alloc_args)
        # Give LNS more time as it's the primary target
        lns_assignments = allocator_lns.solve(time_limit=self.time_limit)
        lns_cost = allocator_lns._evaluate_makespan({r: RobotSchedule(r, t) for r, t in lns_assignments.items()})

        # B. Run LNS2
        allocator_lns2 = TaskAllocatorLNS2(**alloc_args)
        # Give LNS more time as it's the primary target
        lns2_assignments = allocator_lns2.solve(time_limit=self.time_limit)
        lns2_cost = allocator_lns2._evaluate_makespan({r: RobotSchedule(r, t) for r, t in lns2_assignments.items()})

        # B. Run LNS3
        allocator_lns3 = TaskAllocatorLNS3(**alloc_args)
        # Give LNS more time as it's the primary target
        lns3_assignments = allocator_lns3.solve(time_limit=self.time_limit)
        lns3_cost = allocator_lns3._evaluate_makespan({r: RobotSchedule(r, t) for r, t in lns3_assignments.items()})


        print(f"--- RESULT COMPARISON ---")
        print(f"SA Cost:  {sa_cost:.2f}")
        print(f"LNS Cost: {lns_cost:.2f}")
        print(f"LNS2 Cost: {lns2_cost:.2f}")
        print(f"LNS3 Cost: {lns3_cost:.2f}")

        # --- NEW: Call Debug Printer ---
        # self._print_debug_comparison(sa_assignments, lns_assignments, cost_matrix, heading_matrix)
        # -------------------------------

        # Pick the winner
        if lns3_cost <= sa_cost and lns3_cost <= lns_cost and lns3_cost <= lns2_cost:
            print(">> Using LNS3 Plan")
            assignments = lns3_assignments
        elif lns2_cost <= sa_cost and lns2_cost <= lns_cost:
            print(">> Using LNS2 Plan")
            assignments = lns2_assignments
        elif lns_cost <= sa_cost:
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

                if seg1:
                    full_coords.extend(seg1)
                if seg2:
                    full_coords.extend(seg2)

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
                return cat[src][dst]["coords"]
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
        pos = nx.get_node_attributes(G, "pos")

        # --- Initialize Robot Model for Cost Calculation ---
        sg_default = DiffDriveGeometry.default()
        sp_default = DiffDriveParameters.default()

        # --- [DEBUG PRINT HERE] ---
        v_debug, w_debug = self._get_kinematic_limits(sg_default, sp_default)
        print(f"DEBUG KINEMATICS: V_max = {v_debug:.3f} m/s | Omega_max = {w_debug:.3f} rad/s")
        # --------------------------

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

                for tgt_idx, tgt_label in target_list:
                    try:
                        # 1. Get Shortest Path by DISTANCE (Geometric Path)
                        path_nodes = nx.shortest_path(G, src_idx, tgt_idx, weight="weight")
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

    def _plot_prm(self, G, obstacles, special_nodes, filename, bounds=None, path_data=None, final_paths=None):
        plt.figure(figsize=(12, 12))

        # --- 1. Plot Buffered Obstacles (Inflated Boundaries) ---
        added_buffer_label = False
        for poly in obstacles:
            buffered = poly.buffer(self.robot_radius)

            def plot_poly_outline(geom, label=None):
                x, y = geom.exterior.xy
                plt.plot(x, y, "k--", linewidth=1, alpha=0.5, label=label)
                for interior in geom.interiors:
                    x, y = interior.xy
                    plt.plot(x, y, "k--", linewidth=1, alpha=0.5)

            if buffered.geom_type == "Polygon":
                label = "Buffered (C-Space)" if not added_buffer_label else None
                plot_poly_outline(buffered, label)
                if label:
                    added_buffer_label = True
            elif buffered.geom_type == "MultiPolygon":
                for i, geom in enumerate(buffered.geoms):
                    label = "Buffered (C-Space)" if (not added_buffer_label and i == 0) else None
                    plot_poly_outline(geom, label)
                    if label:
                        added_buffer_label = True

        # --- 2. Plot Real Obstacles ---
        added_obs_label = False
        for poly in obstacles:

            def fill_poly(geom, label=None):
                x, y = geom.exterior.xy
                plt.fill(x, y, color="gray", alpha=0.5, label=label)

            if poly.geom_type == "Polygon":
                label = "Static Obstacle" if not added_obs_label else None
                fill_poly(poly, label)
                if label:
                    added_obs_label = True
            elif poly.geom_type == "MultiPolygon":
                for i, geom in enumerate(poly.geoms):
                    label = "Static Obstacle" if (not added_obs_label and i == 0) else None
                    fill_poly(geom, label)
                    if label:
                        added_obs_label = True

        # --- 3. Plot Edges ---
        pos = nx.get_node_attributes(G, "pos")
        if pos:
            lines = [[pos[u], pos[v]] for u, v in G.edges()]
            from matplotlib.collections import LineCollection

            lc = LineCollection(lines, colors="green", linewidths=0.5, alpha=0.2)
            plt.gca().add_collection(lc)
            plt.plot([], [], color="green", linewidth=0.5, label="PRM Edges")

            # --- 4. Plot Nodes (Samples) ---
            sample_x = [pos[n][0] for n in G.nodes if G.nodes[n].get("type") == "sample"]
            sample_y = [pos[n][1] for n in G.nodes if G.nodes[n].get("type") == "sample"]
            plt.plot(sample_x, sample_y, "k.", markersize=1, alpha=0.5, label="Samples")

        # --- 5. Plot Special Nodes ---
        for key, color, marker, label_text in [
            ("starts", "b", "o", "Start"),
            ("goals", "r", "x", "Goal"),
            ("collections", "orange", "d", "Collection"),
        ]:
            if special_nodes[key]:
                sx, sy = zip(*special_nodes[key])
                plt.plot(
                    sx, sy, color=color, marker=marker, linestyle="None", markersize=10, label=label_text, zorder=20
                )

        # --- 6. Plot Final Paths (If Available) ---
        if final_paths:
            colors = ["cyan", "magenta", "yellow", "lime", "blue"]
            for i, (robot_name, coords) in enumerate(final_paths.items()):
                if not coords:
                    continue
                c = colors[i % len(colors)]
                plt.plot(*zip(*coords), color=c, linewidth=4, alpha=0.8, label=f"Plan {robot_name}", zorder=30)

        # Fallback to plotting fragments if no final path
        elif path_data:

            def plot_category_paths(category, color_code):
                if category not in path_data:
                    return
                for src_label, targets in path_data[category].items():
                    for tgt_label, info in targets.items():
                        path = info["coords"]
                        is_best = info["is_best"]
                        if is_best:
                            plt.plot(*zip(*path), color=color_code, linestyle="-", linewidth=2.5, alpha=0.4, zorder=5)

            plot_category_paths("starts", "darkviolet")
            plot_category_paths("goals", "brown")

        # --- 7. Final Setup ---
        plt.legend(loc="upper right", fontsize="small", framealpha=0.9)
        plt.title(f"Plan (N={len(G.nodes)}, Edges={len(G.edges)})")
        plt.axis("equal")
        if bounds:
            plt.xlim(bounds[0], bounds[2])
            plt.ylim(bounds[1], bounds[3])
        plt.grid(True, which="both", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _print_debug_comparison(self, sa_assignments, lns_assignments, cost_matrix, heading_matrix):
        """
        Prints a detailed step-by-step breakdown of SA vs LNS plans.
        """
        print("\n" + "=" * 80)
        print("DEBUG: DETAILED PLAN COMPARISON")
        print("=" * 80)

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
                curr_heading = 0.0  # Placeholder, logic below fixes this relative to path

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

                        robot_time += d_g  # (Plus turning time which is calculated in allocator)

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
        print("=" * 80 + "\n")
