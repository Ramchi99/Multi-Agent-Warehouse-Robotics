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
        # Parameters
        self.num_samples = 2000         # Can handle more samples now
        self.target_degree = 20         # We WANT this many connections per node
        self.max_candidates = 50        # We CHECK this many to find the valid ones (handles deleted vertices)
        self.robot_radius = 0.8         # Buffer size (robot width/2 + margin)
        self.connection_radius = 10.0   # Max length of an edge
        self.min_sample_dist = 0.3      # Minimum distance between nodes

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
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
            # Optional: Add margin to bounds
            # bounds = (bounds[0]-2, bounds[1]-2, bounds[2]+2, bounds[3]+2)
        else:
            bounds = (-12.0, -12.0, 12.0, 12.0)
        
        # --- 2. PREPARE NODES (Iterate once) ---
        # We build two structures simultaneously to avoid double looping
        special_nodes_plot = {"starts": [], "goals": [], "collections": []}
        initial_nodes_data = [] # List of (x, y, type, label)

        # Starts
        for name, state in init_sim_obs.initial_states.items():
            special_nodes_plot["starts"].append((state.x, state.y))
            initial_nodes_data.append((state.x, state.y, "start", name))

        # Shared Goals
        if init_sim_obs.shared_goals:
            for gid, sgoal in init_sim_obs.shared_goals.items():
                if hasattr(sgoal, 'polygon'):
                    c = sgoal.polygon.centroid
                    special_nodes_plot["goals"].append((c.x, c.y))
                    initial_nodes_data.append((c.x, c.y, "goal", gid))

        # Collection Points
        if init_sim_obs.collection_points:
            for cid, cpoint in init_sim_obs.collection_points.items():
                if hasattr(cpoint, 'polygon'):
                    c = cpoint.polygon.centroid
                    special_nodes_plot["collections"].append((c.x, c.y))
                    initial_nodes_data.append((c.x, c.y, "collection", cid))

        # --- 3. BUILD PRM ---
        # Optimization: removed 'obs_polygons' from arguments. 
        # The builder only needs bounds and inflated obstacles.
        G = self._build_prm(inflated_obstacles, initial_nodes_data, bounds)

        # --- 4. DEBUG PLOT ---
        out_dir = Path("out/ex14/debug_plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = out_dir / f"prm_debug_{timestamp}.png"
        
        self._plot_prm(G, obs_polygons, special_nodes_plot, str(filename), bounds)

        # --- 5. RETURN EMPTY PLANS ---
        # TODO: Run A* on graph G here
        planned_paths = {p: [] for p in init_sim_obs.players_obs.keys()}
        global_plan_message = GlobalPlanMessage(
            paths=planned_paths
        )
        return global_plan_message.model_dump_json(round_trip=True)

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
        
        sampler = Halton(d=2, scramble=True)
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
                        G.add_edge(u, v, weight=d)
                        edges_added += 1
        
        return G

    def _plot_prm(self, G, obstacles, special_nodes, filename, bounds=None):
        plt.figure(figsize=(12, 12))
        
        # --- 1. Plot Buffered Obstacles (Inflated Boundaries) ---
        # We re-calculate the buffer here just for visualization so we can see the C-space
        added_buffer_label = False
        for poly in obstacles:
            buffered = poly.buffer(self.robot_radius)
            
            # Helper to plot a single polygon geometry
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
            # Helper to fill polygon
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
            # We don't label every edge, it clutters the legend. We add a proxy artist later if needed.
            lc = LineCollection(lines, colors='green', linewidths=0.5, alpha=0.3)
            plt.gca().add_collection(lc)
            
            # Hack to add "Edges" to legend without plotting a dummy line
            plt.plot([], [], color='green', linewidth=0.5, label='PRM Edges')

            # --- 4. Plot Nodes (Samples) ---
            sample_x = [pos[n][0] for n in G.nodes if G.nodes[n].get('type') == 'sample']
            sample_y = [pos[n][1] for n in G.nodes if G.nodes[n].get('type') == 'sample']
            plt.plot(sample_x, sample_y, 'k.', markersize=1, label='Samples')

        # --- 5. Plot Special Nodes ---
        for key, color, marker, label_text in [
            ('starts', 'b', 'o', 'Start'), 
            ('goals', 'r', 'x', 'Goal'), 
            ('collections', 'orange', 'd', 'Collection')
        ]:
            if special_nodes[key]:
                sx, sy = zip(*special_nodes[key])
                plt.plot(sx, sy, color=color, marker=marker, linestyle='None', markersize=8, label=label_text)

        # --- 6. Final Setup ---
        plt.legend(loc="upper right", fontsize='small', framealpha=0.9)
        plt.title(f"k-NN PRM (N={len(G.nodes)}, Edges={len(G.edges)})")
        plt.axis('equal')
        if bounds:
            plt.xlim(bounds[0], bounds[2])
            plt.ylim(bounds[1], bounds[3])
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()