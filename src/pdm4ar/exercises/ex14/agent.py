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
        self.num_samples = 1000
        self.k_neighbors = 20
        self.robot_radius = 0.8 # Buffer size (robot width/2 + margin)

    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
        # TODO: implement here your global planning stack.
        # global_plan_message = GlobalPlanMessage(
        #     fake_id=1,
        #     fake_name="agent_1",
        #     fake_np_data=np.array([[1, 2, 3], [4, 5, 6]]),
        # )

        # --- 1. EXTRACT & INFLATE OBSTACLES ---
        obs_polygons = []
        if init_sim_obs.dg_scenario and init_sim_obs.dg_scenario.static_obstacles:
            for obs in init_sim_obs.dg_scenario.static_obstacles:
                if hasattr(obs, 'shape'):
                    obs_polygons.append(obs.shape)
        
        # Buffer obstacles (Configuration Space)
        inflated_obstacles = [o.buffer(self.robot_radius) for o in obs_polygons]
        combined_obstacles = unary_union(inflated_obstacles)

        # --- 2. GATHER IMPORTANT NODES ---
        G = nx.Graph()
        
        special_nodes = {
            "starts": [],
            "goals": [],
            "collections": []
        }

        # Starts
        for name, state in init_sim_obs.initial_states.items():
            special_nodes["starts"].append((state.x, state.y))
            G.add_node(len(G.nodes), pos=(state.x, state.y), type="start", label=name)

        # Shared Goals
        if init_sim_obs.shared_goals:
            for gid, sgoal in init_sim_obs.shared_goals.items():
                if hasattr(sgoal, 'polygon'):
                    c = sgoal.polygon.centroid
                    special_nodes["goals"].append((c.x, c.y))
                    G.add_node(len(G.nodes), pos=(c.x, c.y), type="goal", label=gid)

        # Collection Points
        if init_sim_obs.collection_points:
            for cid, cpoint in init_sim_obs.collection_points.items():
                # Assuming cpoint is a CollectionPoint object which might have a polygon
                if hasattr(cpoint, 'polygon'):
                    c = cpoint.polygon.centroid
                    special_nodes["collections"].append((c.x, c.y))
                    G.add_node(len(G.nodes), pos=(c.x, c.y), type="collection", label=cid)

        # --- 3. SAMPLE REMAINING NODES (HALTON + OB-PRM) ---
        # Dynamic bounds from obstacles (use raw obstacles for tighter sampling/plotting limits)
        raw_combined = unary_union(obs_polygons)
        if not raw_combined.is_empty:
            bounds = raw_combined.bounds # (minx, miny, maxx, maxy)
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
            is_valid = not p.within(combined_obstacles)
            
            final_point = None
            if is_valid:
                # Valid free space point
                final_point = p
            else:
                # OB-PRM: Project invalid point to valid surface
                try:
                    # nearest_points(p, boundary) returns (p, point_on_boundary)
                    nearest = nearest_points(p, boundary_geom)[1]
                    
                    # Nudge slightly into free space
                    # Vector from p (inside) to nearest (surface) points OUTWARD
                    dx = nearest.x - p.x
                    dy = nearest.y - p.y
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 1e-6:
                        nudge_dist = 0.1 # 5cm safety margin
                        nx_vec = (dx / dist) * nudge_dist
                        ny_vec = (dy / dist) * nudge_dist
                        final_point = Point(nearest.x + nx_vec, nearest.y + ny_vec)
                    else:
                        # If p is exactly on boundary (unlikely), just take it
                        final_point = nearest
                        
                except Exception:
                    continue

            if final_point:
                G.add_node(len(G.nodes), pos=(final_point.x, final_point.y), type="sample")
                count += 1

        # --- 4. CONNECT k-NEAREST NEIGHBORS ---
        node_positions = nx.get_node_attributes(G, 'pos')
        nodes_list = list(G.nodes)
        
        for i in nodes_list:
            pos_i = Point(node_positions[i])
            
            # Calculate distances to all other nodes
            distances = []
            for j in nodes_list:
                if i == j: continue
                pos_j = Point(node_positions[j])
                dist = pos_i.distance(pos_j)
                distances.append((dist, j))
            
            # Sort and take top k
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.k_neighbors]
            
            for dist, j in neighbors:
                pos_j = Point(node_positions[j])
                line = LineString([pos_i, pos_j])
                
                if not line.intersects(combined_obstacles):
                    G.add_edge(i, j, weight=dist)

        # --- 5. DEBUG PLOT ---
        out_dir = Path("out/ex14/debug_plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = out_dir / f"prm_debug_{timestamp}.png"
        
        self._plot_prm(G, obs_polygons, special_nodes, str(filename), bounds)

        # --- 6. RETURN EMPTY PLANS ---
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
        plt.title(f"k-NN PRM (N={len(G.nodes)}, k={self.k_neighbors})")
        plt.axis('equal')
        if bounds:
            plt.xlim(bounds[0], bounds[2])
            plt.ylim(bounds[1], bounds[3])
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
