import random
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, List, Optional

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

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        pass

    def on_receive_global_plan(
        self,
        serialized_msg: str,
    ):
        # TODO: process here the received global plan
        global_plan = GlobalPlanMessage.model_validate_json(serialized_msg)
        # We don't do anything with the plan yet

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
        # ### [OLD]
        # self.num_samples = 1000
        # self.k_neighbors = 20
        
        # ### [NEW] Optimized parameters
        self.num_samples = 2000     # Can handle more samples now
        self.target_degree = 20     # We WANT this many connections per node
        self.max_candidates = 50    # We CHECK this many to find the valid ones (handles deleted vertices)
        self.robot_radius = 0.8     # Buffer size (robot width/2 + margin)
        self.connection_radius = 10.0 # Max length of an edge

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
        # TODO: implement here your global planning stack.

        # --- 1. EXTRACT & INFLATE OBSTACLES ---
        obs_polygons = []
        if init_sim_obs.dg_scenario and init_sim_obs.dg_scenario.static_obstacles:
            for obs in init_sim_obs.dg_scenario.static_obstacles:
                if hasattr(obs, 'shape'):
                    obs_polygons.append(obs.shape)
        
        # Buffer obstacles (Configuration Space)
        inflated_obstacles = [o.buffer(self.robot_radius) for o in obs_polygons]
        combined_obstacles = unary_union(inflated_obstacles)

        # ### [NEW] Build STRtree for O(log N) collision checks (much faster than combined_obstacles)
        obstacle_tree = STRtree(inflated_obstacles)

        # --- 2. GATHER IMPORTANT NODES ---
        G = nx.Graph()
        
        # ### [NEW] Keep track of coordinates for KDTree later
        node_coords = [] # list of [x, y]
        node_indices = [] # list of node IDs

        special_nodes = {
            "starts": [],
            "goals": [],
            "collections": []
        }

        # Starts
        for name, state in init_sim_obs.initial_states.items():
            special_nodes["starts"].append((state.x, state.y))
            # ### [NEW] Add to coord lists
            idx = len(G.nodes)
            G.add_node(idx, pos=(state.x, state.y), type="start", label=name)
            node_coords.append([state.x, state.y])
            node_indices.append(idx)

        # Shared Goals
        if init_sim_obs.shared_goals:
            for gid, sgoal in init_sim_obs.shared_goals.items():
                if hasattr(sgoal, 'polygon'):
                    c = sgoal.polygon.centroid
                    special_nodes["goals"].append((c.x, c.y))
                    # ### [NEW] Add to coord lists
                    idx = len(G.nodes)
                    G.add_node(idx, pos=(c.x, c.y), type="goal", label=gid)
                    node_coords.append([c.x, c.y])
                    node_indices.append(idx)

        # Collection Points
        if init_sim_obs.collection_points:
            for cid, cpoint in init_sim_obs.collection_points.items():
                if hasattr(cpoint, 'polygon'):
                    c = cpoint.polygon.centroid
                    special_nodes["collections"].append((c.x, c.y))
                    # ### [NEW] Add to coord lists
                    idx = len(G.nodes)
                    G.add_node(idx, pos=(c.x, c.y), type="collection", label=cid)
                    node_coords.append([c.x, c.y])
                    node_indices.append(idx)

        # --- 3. SAMPLE REMAINING NODES (HALTON + OB-PRM) ---
        raw_combined = unary_union(obs_polygons)
        if not raw_combined.is_empty:
            bounds = raw_combined.bounds # (minx, miny, maxx, maxy)
            # ### [NEW] Add margin to bounds so we don't sample exactly on edge of map
            #bounds = (bounds[0]-2, bounds[1]-2, bounds[2]+2, bounds[3]+2)
        else:
            bounds = (-12.0, -12.0, 12.0, 12.0)
            
        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y
        
        # Initialize Halton sampler
        sampler = Halton(d=2, scramble=True)
        # Oversample significantly to find enough points
        raw_samples = sampler.random(n=self.num_samples * 3) 
        
        # Scale samples to bounds
        samples_x = raw_samples[:, 0] * width + min_x
        samples_y = raw_samples[:, 1] * height + min_y
        
        count = 0
        boundary_geom = combined_obstacles.boundary

        for x, y in zip(samples_x, samples_y):
            if count >= self.num_samples:
                break
                
            p = Point(x, y)
            
            # ### [OLD] Slow check
            # is_valid = not p.within(combined_obstacles)
            
            # ### [NEW] Fast check using STRtree
            # query returns indices of obstacles that 'might' intersect
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
                # OB-PRM: Project invalid point to valid surface
                try:
                    nearest = nearest_points(p, boundary_geom)[1]
                    dx = nearest.x - p.x
                    dy = nearest.y - p.y
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 1e-6:
                        nudge_dist = 0.1 
                        nx_vec = (dx / dist) * nudge_dist
                        ny_vec = (dy / dist) * nudge_dist
                        final_point = Point(nearest.x + nx_vec, nearest.y + ny_vec)
                    else:
                        final_point = nearest
                        
                except Exception:
                    continue

            if final_point:
                # ### [NEW] Add to coord lists and graph
                idx = len(G.nodes)
                G.add_node(idx, pos=(final_point.x, final_point.y), type="sample")
                node_coords.append([final_point.x, final_point.y])
                node_indices.append(idx)
                count += 1

        # --- 4. CONNECT k-NEAREST NEIGHBORS (OPTIMIZED) ---
        
        # ### [OLD] O(N^2) Approach
        # node_positions = nx.get_node_attributes(G, 'pos')
        # nodes_list = list(G.nodes)
        # for i in nodes_list:
        #     pos_i = Point(node_positions[i])
        #     distances = []
        #     for j in nodes_list:
        #         if i == j: continue
        #         pos_j = Point(node_positions[j])
        #         dist = pos_i.distance(pos_j)
        #         distances.append((dist, j))
        #     distances.sort(key=lambda x: x[0])
        #     neighbors = distances[:self.k_neighbors]
        #     for dist, j in neighbors:
        #         pos_j = Point(node_positions[j])
        #         line = LineString([pos_i, pos_j])
        #         if not line.intersects(combined_obstacles):
        #             G.add_edge(i, j, weight=dist)

        # ### [NEW] O(N log N) Approach with Deleted Vertices handling
        if len(node_coords) > 1:
            data_np = np.array(node_coords)
            tree = cKDTree(data_np)
            
            # Query more neighbors than we need (max_candidates) to account for collisions
            dists_all, indices_all = tree.query(data_np, k=self.max_candidates)
            
            for i, (nbr_dists, nbr_indices) in enumerate(zip(dists_all, indices_all)):
                u = node_indices[i]
                u_pos = Point(node_coords[i])
                edges_added = 0
                
                for d, j_idx in zip(nbr_dists, nbr_indices):
                    # Skip self
                    if i == j_idx: continue
                    
                    # Stop if we have enough connections (handling "deleted vertices")
                    if edges_added >= self.target_degree: break
                    
                    # Stop if neighbor is physically too far
                    if d > self.connection_radius: break
                    
                    v = node_indices[j_idx]
                    
                    # Avoid duplicate edge checks
                    if G.has_edge(u, v):
                        edges_added += 1
                        continue
                    
                    v_pos = Point(node_coords[j_idx])
                    line = LineString([u_pos, v_pos])
                    
                    # Fast Collision Check
                    # 1. Broad Phase: Get obstacles near the line
                    candidates_idx = obstacle_tree.query(line)
                    
                    # 2. Narrow Phase: Check intersection
                    is_colliding = False
                    for idx in candidates_idx:
                        if inflated_obstacles[idx].intersects(line):
                            is_colliding = True
                            break
                            
                    if not is_colliding:
                        G.add_edge(u, v, weight=d)
                        edges_added += 1


        # --- 5. DEBUG PLOT ---
        out_dir = Path("out/ex14/debug_plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = out_dir / f"prm_debug_{timestamp}.png"
        
        self._plot_prm(G, obs_polygons, special_nodes, str(filename), bounds)

        # --- 6. RETURN EMPTY PLANS ---
        # TODO: Run A* on graph G here
        planned_paths = {p: [] for p in init_sim_obs.players_obs.keys()}
        global_plan_message = GlobalPlanMessage(
            paths=planned_paths
        )
        return global_plan_message.model_dump_json(round_trip=True)

    def _plot_prm(self, G, obstacles, special_nodes, filename, bounds=None):
        plt.figure(figsize=(12, 12))
        
        # 0. Plot Buffered Obstacles (C-Space)
        for poly in obstacles:
            buffered = poly.buffer(self.robot_radius)
            if buffered.geom_type == 'Polygon':
                x, y = buffered.exterior.xy
                plt.plot(x, y, 'k--', linewidth=1, alpha=0.5, label='Buffered' if 'Buffered' not in plt.gca().get_legend_handles_labels()[1] else "")
                for interior in buffered.interiors:
                    x, y = interior.xy
                    plt.plot(x, y, 'k--', linewidth=1, alpha=0.5)
            elif buffered.geom_type == 'MultiPolygon':
                for geom in buffered.geoms:
                    x, y = geom.exterior.xy
                    plt.plot(x, y, 'k--', linewidth=1, alpha=0.5)
                    for interior in geom.interiors:
                        x, y = interior.xy
                        plt.plot(x, y, 'k--', linewidth=1, alpha=0.5)

        # 1. Plot Real Obstacles
        for poly in obstacles:
            if poly.geom_type == 'Polygon':
                x, y = poly.exterior.xy
                plt.fill(x, y, color='gray', alpha=0.5, label='Obstacle' if 'Obstacle' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif poly.geom_type == 'MultiPolygon':
                 for geom in poly.geoms:
                    x, y = geom.exterior.xy
                    plt.fill(x, y, color='gray', alpha=0.5)
            elif poly.geom_type in ['LineString', 'LinearRing']:
                x, y = poly.xy
                plt.plot(x, y, color='gray', linewidth=3, alpha=0.7, label='Boundary' if 'Boundary' not in plt.gca().get_legend_handles_labels()[1] else "")

        # 2. Plot Edges
        pos = nx.get_node_attributes(G, 'pos')
        # Collect lines for faster plotting
        lines = []
        for (u, v) in G.edges():
            p1 = pos[u]
            p2 = pos[v]
            lines.append([p1, p2])
        
        # Plot edges as a collection
        from matplotlib.collections import LineCollection
        lc = LineCollection(lines, colors='green', linewidths=0.5, alpha=0.3)
        plt.gca().add_collection(lc)

        # 3. Plot Nodes (Samples)
        sample_x = [pos[n][0] for n in G.nodes if G.nodes[n].get('type') == 'sample']
        sample_y = [pos[n][1] for n in G.nodes if G.nodes[n].get('type') == 'sample']
        plt.plot(sample_x, sample_y, 'k.', markersize=1, alpha=0.5, label='Sample')

        # 4. Plot Special Nodes
        if special_nodes['starts']:
            sx, sy = zip(*special_nodes['starts'])
            plt.plot(sx, sy, 'bo', markersize=8, label='Start')
        
        if special_nodes['goals']:
            gx, gy = zip(*special_nodes['goals'])
            plt.plot(gx, gy, 'rx', markersize=8, markeredgewidth=2, label='Goal')

        if special_nodes['collections']:
            cx, cy = zip(*special_nodes['collections'])
            plt.plot(cx, cy, 'bd', markersize=8, color='orange', label='Collection')

        plt.legend()
        # ### [NEW] Updated title
        plt.title(f"k-NN PRM (N={len(G.nodes)}, Edges={len(G.edges)})")
        plt.axis('equal')
        if bounds:
            plt.xlim(bounds[0], bounds[2])
            plt.ylim(bounds[1], bounds[3])
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()