import math
import itertools
import heapq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import bisect
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from scipy.spatial import cKDTree

# --- 1. DATA STRUCTURES ---

@dataclass
class TrajectoryPoint:
    x: float
    y: float
    theta: float
    t: float
    v: float = 0.0
    w: float = 0.0
    mode: str = "idle"  # 'turn', 'drive', 'wait', 'idle'

class SpatialTimeHash:
    """
    A grid-based Space-Time reservation system.
    Stores 'occupied' bubbles in (grid_x, grid_y) buckets.
    """
    def __init__(self, cell_size=2.0):
        self.cell_size = cell_size
        # Map: (gx, gy) -> List of (x, y, t_start, t_end, radius)
        self.grid = defaultdict(list)
        # Map: (gx, gy) -> List of (x, y, t_start, radius)  <-- Permanent
        self.static_grid = defaultdict(list)
        # Helper to track if a bucket needs sorting
        self.dirty_buckets = set()

    def add_dense_trajectory(self, trajectory: List[TrajectoryPoint], radius: float):
        """
        Samples a dense trajectory and reserves space-time volumes.
        """
        modified_keys = set()
        for pt in trajectory:
            gx = int(pt.x / self.cell_size)
            gy = int(pt.y / self.cell_size)
            
            # Add to bucket. 
            entry = (pt.x, pt.y, pt.t - 0.05, pt.t + 0.05, radius)
            key = (gx, gy)
            self.grid[key].append(entry)
            modified_keys.add(key)
            
        # Sort modified buckets by t_start (index 2)
        for key in modified_keys:
            self.grid[key].sort(key=lambda x: x[2])

    def add_permanent_obstacle(self, x, y, t_start, radius):
        """Reserves a position permanently from t_start onwards."""
        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)
        self.static_grid[(gx, gy)].append((x, y, t_start, radius))

    def is_collision(self, x, y, t_start, t_end, radius, safety_margin=0.1):
        """
        Checks if a volume (circle at x,y) collides with anything during [t_start, t_end].
        Uses Binary Search for temporal filtering.
        """
        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)
        
        # Check 3x3 neighbor grids to handle edge cases
        for ddx in [-1, 0, 1]:
            for ddy in [-1, 0, 1]:
                key = (gx + ddx, gy + ddy)
                
                # Check Permanent Obstacles
                if key in self.static_grid:
                    for (ox, oy, ot_start, orad) in self.static_grid[key]:
                        # Temporal: Our end time > obstacle start time
                        if t_end > ot_start:
                            dist_sq = (x - ox)**2 + (y - oy)**2
                            safe_sq = (radius + orad + safety_margin) ** 2
                            if dist_sq < safe_sq:
                                return True

                if key not in self.grid:
                    continue
                
                bucket = self.grid[key]
                if not bucket: continue

                # Binary Search: Find first entry where entry.t_end > t_start
                # Since we sorted by t_start (index 2), and t_end ~= t_start + 0.1,
                # we can approximate or just search for t_start - max_duration.
                # A safer bound: search for entry where entry.t_start >= t_start - 0.2 (approx max bubble dur)
                # Actually, let's just find where entry.t_start >= t_start - 1.0 to be safe
                # then check overlap.
                
                # Optimized: We need max(t_start, ot0) < min(t_end, ot1)
                # This implies ot1 > t_start.
                # Since ot1 = ot0 + dt, ot0 + dt > t_start => ot0 > t_start - dt.
                # Let's find index where ot0 >= t_start - 0.5 (generous buffer)
                
                search_val = t_start - 0.5
                # Create a dummy tuple for comparison (only index 2 matters)
                # We use a key function for bisect in Python 3.10+, but for compatibility:
                # We'll rely on the fact that python compares tuples element-wise.
                # (float('-inf'), ..., search_val, ...)
                
                # Custom bisect for list of tuples based on index 2
                idx = bisect.bisect_left(bucket, search_val, key=lambda e: e[2])
                
                for i in range(idx, len(bucket)):
                    (ox, oy, ot0, ot1, orad) = bucket[i]
                    
                    # Stop condition: if bubble starts after our query ends, no future bubble can overlap
                    # (since list is sorted by start time)
                    if ot0 >= t_end:
                        break
                        
                    # 1. Temporal Overlap (Exact Check)
                    if max(t_start, ot0) < min(t_end, ot1):
                        # 2. Spatial Overlap
                        dist_sq = (x - ox)**2 + (y - oy)**2
                        safe_sq = (radius + orad + safety_margin) ** 2
                        if dist_sq < safe_sq:
                            return True
        return False

# --- 2. THE PLANNER ---

class SpaceTimeRoadmapPlanner:
    def __init__(self, 
                 prm_graph: nx.Graph, 
                 robot_radius: float, 
                 v_max: float,
                 w_max: float,
                 dt_search: float = 0.5,
                 use_prm: bool = True):
        
        self.base_graph = prm_graph
        # REMOVED: nx.relabel_nodes(self.base_graph, str, copy=False)
        
        self.radius = robot_radius
        self.v_max = v_max
        self.w_max = w_max
        self.dt = dt_search
        self.use_prm = use_prm
        
        # Pre-build PRM KDTree for fast "ramp" connections
        self.prm_node_indices = list(self.base_graph.nodes())
        coords = [self.base_graph.nodes[n]['pos'] for n in self.prm_node_indices]
        self.prm_coords = np.array(coords) if coords else np.empty((0, 2))
        self.prm_tree = cKDTree(self.prm_coords) if len(self.prm_coords) > 0 else None

        # Collision System
        self.st_hash = SpatialTimeHash(cell_size=3.0)
        
        # Debug Data
        self.debug_waits = []
        self.debug_paths = {}

    def plan_prioritized(self, 
                         assignments: Dict[str, List[any]], 
                         path_data: dict, 
                         initial_states: Dict[str, Tuple[float, float, float]]) -> Tuple[Dict[str, List[TrajectoryPoint]], Dict[str, float], int]:
        
        robots = list(assignments.keys())
        
        # Heuristic: Sort by longest path first (default)
        heuristic_order = sorted(robots, key=lambda r: len(assignments[r]), reverse=True)
        
        # Decide on Permutations
        # If N <= 4 (24 perms), force brute force.
        # If N >= 5, stick to heuristic to avoid timeouts.
        if len(robots) <= 4:
            permutations = list(itertools.permutations(robots))
        else:
            permutations = [tuple(heuristic_order)]
            
        print(f"--- Hybrid Space-Time Planner: Checking {len(permutations)} permutations ---")
        
        best_result = None
        best_score = (float('inf'), float('inf')) # (Failures, Makespan)
        
        for i, perm in enumerate(permutations):
            res = self._run_priority_pass(list(perm), assignments, path_data, initial_states)
            final_plans, mission_times, d_paths, d_waits, fail_count = res
            
            # Calculate Makespan (Mission End)
            makespan = 0.0
            if mission_times:
                makespan = max(mission_times.values())
            
            # Score: Minimize Failures first, then Makespan
            current_score = (fail_count, makespan)
            
            if current_score < best_score:
                best_score = current_score
                best_result = res
                
        # Unpack Best
        final_plans, mission_times, d_paths, d_waits, fail_count = best_result
        
        # Update Class State for Visualization
        self.debug_paths = d_paths
        self.debug_waits = d_waits
        
        # --- Visualization Truncation ---
        if mission_times:
            global_mission_end = max(mission_times.values()) if mission_times else 0.0
            for r_name in self.debug_paths:
                self.debug_paths[r_name] = [p for p in self.debug_paths[r_name] if p[2] <= global_mission_end + 0.1]
                
        print(f"--- Selected Best Permutation (Failures: {fail_count}, Makespan: {best_score[1]:.2f}s) ---")
        return final_plans, mission_times, fail_count

    def _run_priority_pass(self, 
                           ordered_robots: List[str],
                           assignments: Dict[str, List[any]], 
                           path_data: dict, 
                           initial_states: Dict[str, Tuple[float, float, float]]):
        
        # Local State per pass
        self.st_hash = SpatialTimeHash() 
        local_debug_waits = []
        local_debug_paths = {}
        final_plans = {}
        mission_times = {}
        failure_count = 0
        
        for r_name in ordered_robots:
            start_pose = initial_states[r_name]
            tasks = assignments[r_name]
            
            segments = self._extract_geometric_segments(r_name, tasks, path_data)
            
            full_robot_traj = []
            curr_pose = start_pose
            curr_time = 0.0
            
            mission_end_idx = len(segments) - 3
            recorded_mission_time = 0.0
            
            if not segments:
                mission_times[r_name] = 0.0
                continue

            success = True
            
            for seg_idx, raw_path in enumerate(segments):
                if not raw_path: continue
                
                geometric_path = self._densify_segment(raw_path, spacing=0.5)
                added_nodes, overlay_map = self._add_overlay(geometric_path)
                
                get_node_data = self.base_graph.nodes
                def get_pos(n):
                    if n in overlay_map: return overlay_map[n]
                    return get_node_data[n]['pos']

                try:
                    start_node = -1 
                    goal_node = -1 - (len(geometric_path)-1)
                    start_t_idx = int(math.ceil(curr_time / self.dt))
                    
                    node_sequence = self._spacetime_astar(
                        self.base_graph, get_pos, start_node, goal_node, curr_pose[2], start_t_idx
                    )
                finally:
                    self._remove_overlay(added_nodes)
                
                if node_sequence:
                    seg_start_h = curr_pose[2]
                    seg_traj, waits = self._nodes_to_trajectory(node_sequence, get_pos, seg_start_h)
                    
                    if full_robot_traj and seg_traj:
                        if math.hypot(seg_traj[0].x - full_robot_traj[-1].x, seg_traj[0].y - full_robot_traj[-1].y) < 1e-3:
                             full_robot_traj.extend(seg_traj[1:])
                        else:
                             full_robot_traj.extend(seg_traj)
                    else:
                        full_robot_traj.extend(seg_traj)
                    
                    # Debug removed
                    last_pt = seg_traj[-1]
                    curr_pose = (last_pt.x, last_pt.y, last_pt.theta)
                    curr_time = last_pt.t
                    
                    if seg_idx == mission_end_idx:
                        recorded_mission_time = curr_time
                    
                    for w_pos, w_dur in waits:
                         local_debug_waits.append((w_pos[0], w_pos[1], w_dur, r_name))

                else:
                    # print(f"  [!] {r_name}: Segment {seg_idx} failed. Holding.")
                    dummy = [TrajectoryPoint(curr_pose[0], curr_pose[1], curr_pose[2], curr_time + t, 0, 0, "wait") for t in np.arange(0, 30.0, 0.1)]
                    full_robot_traj.extend(dummy)
                    local_debug_waits.append((curr_pose[0], curr_pose[1], 30.0, f"{r_name} (STUCK)"))
                    success = False
                    failure_count += 1
                    break 

            mission_times[r_name] = recorded_mission_time

            self.st_hash.add_dense_trajectory(full_robot_traj, self.radius)
            if full_robot_traj:
                last = full_robot_traj[-1]
                self.st_hash.add_permanent_obstacle(last.x, last.y, last.t, self.radius)
            
            final_plans[r_name] = full_robot_traj
            local_debug_paths[r_name] = [(p.x, p.y, p.t) for p in full_robot_traj]
            
        return final_plans, mission_times, local_debug_paths, local_debug_waits, failure_count

    def _densify_segment(self, coords, spacing=0.5):
        if not coords or len(coords) < 2: return coords
        new_coords = [coords[0]]
        for i in range(len(coords)-1):
            p1 = np.array(coords[i])
            p2 = np.array(coords[i+1])
            dist = np.linalg.norm(p2 - p1)
            if dist > spacing:
                num_points = int(dist / spacing)
                for j in range(1, num_points + 1):
                    new_coords.append(tuple(p1 + (p2 - p1) * (j / (num_points + 1))))
            new_coords.append(coords[i+1])
        return new_coords

    def _add_overlay(self, geometric_path):
        """
        Adds the geometric path nodes to the base graph IN PLACE using NEGATIVE INTEGER IDs.
        Returns a list of added node IDs for later cleanup.
        """
        added_nodes = []
        overlay = {}
        prev_id = None
        
        for i, (x, y) in enumerate(geometric_path):
            nid = -1 - i  # Integer ID: -1, -2, -3...
            # Add to graph
            if not self.base_graph.has_node(nid):
                self.base_graph.add_node(nid, pos=(x, y))
                added_nodes.append(nid)
            
            overlay[nid] = (x, y)
            
            if prev_id is not None:
                self.base_graph.add_edge(prev_id, nid, weight=1.0)
            
            # Connect to PRM
            if self.use_prm and self.prm_tree:
                dists, indices = self.prm_tree.query([x, y], k=3)
                if not isinstance(indices, (list, np.ndarray)): indices = [indices]
                for idx in indices:
                    prm_id = int(self.prm_node_indices[idx]) # Integer ID
                    prm_pos = self.prm_coords[idx]
                    if math.hypot(x-prm_pos[0], y-prm_pos[1]) < 3.0:
                        self.base_graph.add_edge(nid, prm_id, weight=3.0)
                        self.base_graph.add_edge(prm_id, nid, weight=3.0)
            
            prev_id = nid
            
        return added_nodes, overlay

    def _remove_overlay(self, added_nodes):
        """Removes the temporary overlay nodes from the graph."""
        for n in added_nodes:
            if self.base_graph.has_node(n):
                self.base_graph.remove_node(n)

    def _spacetime_astar(self, graph, get_pos_func, start_node, goal_node, start_h, start_t_idx):
        # PQ State: (f, g, t_idx, curr_node, curr_heading)
        pq = [(0, 0.0, start_t_idx, start_node, start_h)]
        
        visited = {}
        came_from = {}
        
        goal_pos = get_pos_func(goal_node)
        max_time_idx = start_t_idx + int(120.0 / self.dt)

        while pq:
            f, g, t_idx, curr, curr_h = heapq.heappop(pq)
            
            if t_idx > max_time_idx: continue
            
            state = (curr, t_idx)
            if state in visited and visited[state] <= g: continue
            visited[state] = g
            
            current_time = t_idx * self.dt
            
            if curr == goal_node:
                return self._reconstruct(came_from, curr, t_idx)
                
            curr_pos = get_pos_func(curr)

            # --- MOVE ---
            for neighbor in graph.neighbors(curr):
                nbr_pos = get_pos_func(neighbor)
                edge_weight = graph.edges[curr, neighbor].get('weight', 1.0)
                
                dx, dy = nbr_pos[0] - curr_pos[0], nbr_pos[1] - curr_pos[1]
                dist = math.hypot(dx, dy)
                
                # Bi-directional Logic
                target_h_fwd = math.atan2(dy, dx)
                diff_fwd = (target_h_fwd - curr_h + math.pi) % (2*math.pi) - math.pi
                
                if abs(diff_fwd) > math.pi / 2:
                    # Reverse
                    rev_angle = target_h_fwd + math.pi
                    target_h = math.atan2(math.sin(rev_angle), math.cos(rev_angle))
                    
                    diff_raw = target_h - curr_h
                    diff = math.atan2(math.sin(diff_raw), math.cos(diff_raw))
                else:
                    # Forward
                    target_h = target_h_fwd
                    diff = diff_fwd

                duration = (abs(diff) / self.w_max) + (dist / self.v_max)
                arrival_time = current_time + duration
                
                # Check Safety
                safe = True
                if self.st_hash.is_collision(curr_pos[0], curr_pos[1], current_time, current_time + 0.1, self.radius): safe = False
                if safe and self.st_hash.is_collision(nbr_pos[0], nbr_pos[1], arrival_time - 0.1, arrival_time, self.radius): safe = False
                
                if safe:
                    new_g = g + duration * edge_weight
                    h = math.hypot(goal_pos[0]-nbr_pos[0], goal_pos[1]-nbr_pos[1]) / self.v_max
                    new_t_idx = int(math.ceil(arrival_time / self.dt))
                    
                    heapq.heappush(pq, (new_g + h, new_g, new_t_idx, neighbor, target_h))
                    # Ultra-Minimal Memory: (prev_node, prev_t_idx)
                    came_from[(neighbor, new_t_idx)] = (curr, t_idx)

            # --- WAIT ---
            wait_time = current_time + self.dt
            next_t_idx = t_idx + 1
            if not self.st_hash.is_collision(curr_pos[0], curr_pos[1], current_time, wait_time, self.radius):
                wait_cost = g + self.dt * 1.5
                h = math.hypot(goal_pos[0]-curr_pos[0], goal_pos[1]-curr_pos[1]) / self.v_max
                
                heapq.heappush(pq, (wait_cost + h, wait_cost, next_t_idx, curr, curr_h))
                came_from[(curr, next_t_idx)] = (curr, t_idx)

        return None

    def _reconstruct(self, came_from, curr, t_idx):
        # Returns simple list of (node, t_idx)
        path = []
        key = (curr, t_idx)
        while key in came_from:
            path.append(key)
            key = came_from[key]
        path.append(key) # Add start
        path.reverse()
        return path

    def _nodes_to_trajectory(self, node_sequence, get_pos_func, start_heading):
        """
        Re-simulates the path to generate detailed trajectory points.
        Trades CPU for RAM.
        """
        traj = []
        debug_waits = []
        
        if not node_sequence: return [], []

        # Start
        start_node, start_t_idx = node_sequence[0]
        start_pos = get_pos_func(start_node)
        start_t = start_t_idx * self.dt
        
        curr_h = start_heading
        curr_t = start_t
        
        traj.append(TrajectoryPoint(start_pos[0], start_pos[1], curr_h, curr_t, 0, 0, 'idle'))
        
        for i in range(1, len(node_sequence)):
            curr_node, curr_t_idx = node_sequence[i]
            prev_node, prev_t_idx = node_sequence[i-1]
            
            p1 = get_pos_func(prev_node)
            p2 = get_pos_func(curr_node)
            
            # --- ACTION: WAIT ---
            if curr_node == prev_node:
                # Duration based on A* time steps
                # Note: A* steps are discretized by dt.
                # In A*: new_t_idx = prev + 1. So duration = dt.
                duration = (curr_t_idx - prev_t_idx) * self.dt
                steps = int(duration / 0.1)
                for k in range(steps + 1):
                    traj.append(TrajectoryPoint(p1[0], p1[1], curr_h, curr_t + k*0.1, 0, 0, 'wait'))
                
                curr_t += duration
                debug_waits.append((p1, duration))
                
            # --- ACTION: MOVE ---
            else:
                dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                dist = math.hypot(dx, dy)
                
                # Bi-directional Logic
                target_h_fwd = math.atan2(dy, dx)
                diff_fwd = (target_h_fwd - curr_h + math.pi) % (2*math.pi) - math.pi
                
                if abs(diff_fwd) > math.pi / 2:
                    # Reverse
                    rev_angle = target_h_fwd + math.pi
                    target_h = math.atan2(math.sin(rev_angle), math.cos(rev_angle))
                    
                    diff_raw = target_h - curr_h
                    diff = math.atan2(math.sin(diff_raw), math.cos(diff_raw))
                    v_cmd = -self.v_max
                else:
                    # Forward
                    target_h = target_h_fwd
                    diff = diff_fwd
                    v_cmd = self.v_max

                # Re-calculate durations exactly as in A*
                t_turn = abs(diff) / self.w_max
                t_drive = dist / self.v_max
                
                # Turn
                turn_steps = int(math.ceil(t_turn / 0.1))
                omega = self.w_max if diff > 0 else -self.w_max
                for k in range(turn_steps):
                    traj.append(TrajectoryPoint(p1[0], p1[1], curr_h + omega*k*0.1, curr_t + k*0.1, 0, omega, 'turn'))
                
                curr_h = target_h
                curr_t += turn_steps * 0.1 # [FIX] Aligned with discrete steps
                
                # Drive
                drive_steps = int(math.ceil(t_drive / 0.1))
                for k in range(drive_steps + 1):
                    alpha = k / max(1, drive_steps)
                    traj.append(TrajectoryPoint(p1[0] + dx*alpha, p1[1] + dy*alpha, curr_h, curr_t + k*0.1, v_cmd, 0, 'drive'))
                    
                curr_t += drive_steps * 0.1 # [FIX] Aligned with discrete steps
                
        return traj, debug_waits

    def _build_augmented_graph(self, geometric_path):
        G = self.base_graph.copy()
        # Convert all PRM nodes to string keys for safety
        nx.relabel_nodes(G, {n: str(n) for n in G.nodes()}, copy=False)
        
        overlay = {}
        prev_id = None
        
        for i, (x, y) in enumerate(geometric_path):
            nid = f"path_{i}"
            G.add_node(nid, pos=(x, y)) # Ensure pos attribute exists
            overlay[nid] = (x, y)
            
            if prev_id:
                G.add_edge(prev_id, nid, weight=1.0)
            
            # Connect to PRM
            if self.prm_tree:
                dists, indices = self.prm_tree.query([x, y], k=3)
                if not isinstance(indices, (list, np.ndarray)): indices = [indices]
                for idx in indices:
                    prm_id = str(self.prm_node_indices[idx]) # String ID
                    prm_pos = self.prm_coords[idx]
                    if math.hypot(x-prm_pos[0], y-prm_pos[1]) < 3.0:
                        G.add_edge(nid, prm_id, weight=3.0)
                        G.add_edge(prm_id, nid, weight=3.0)
            
            prev_id = nid
            
        # Helper map for positions (merge overlay and graph)
        full_overlay = overlay.copy()
        for n in G.nodes():
            if n not in full_overlay:
                full_overlay[n] = G.nodes[n]['pos']
                
        return G, full_overlay

    def _extract_geometric_segments(self, r_name, tasks, path_data) -> List[List[Tuple[float, float]]]:
        segments = []
        curr = r_name
        
        def get_coords(u, v):
            for cat in path_data.values():
                if u in cat and v in cat[u]: return cat[u][v]['coords']
            return []
            
        for task in tasks:
            # Segment 1: Current -> Goal
            c1 = get_coords(curr, task.goal_id)
            if c1: segments.append(c1)
            else: 
                # Fallback if no path found (should not happen if allocator is correct)
                # But we must maintain connectivity
                segments.append([]) 
            
            curr = task.goal_id
            
            # Segment 2: Goal -> Collection
            c2 = get_coords(curr, task.collection_id)
            if c2: segments.append(c2)
            else: segments.append([])
            
            curr = task.collection_id
            
        return segments

    def plot_execution(self, filename, obstacles, special_nodes=None):
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        for poly in obstacles:
            if hasattr(poly, 'exterior'):
                x, y = poly.exterior.xy
                ax.fill(x, y, color='k', alpha=0.3)
        
        # Plot Special Nodes (Goals/Collections)
        if special_nodes:
            if 'goals' in special_nodes and special_nodes['goals']:
                gx, gy = zip(*special_nodes['goals'])
                ax.plot(gx, gy, 'rx', markersize=10, markeredgewidth=3, label='Goals', zorder=5)
            
            if 'collections' in special_nodes and special_nodes['collections']:
                cx, cy = zip(*special_nodes['collections'])
                ax.plot(cx, cy, 'd', color='orange', markersize=10, markeredgecolor='black', label='Collections', zorder=5)

        colors = plt.cm.get_cmap('tab10', len(self.debug_paths) + 1)
        for i, (r_name, pts) in enumerate(self.debug_paths.items()):
            if not pts: continue
            # Unpack (x, y, t)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ts = [p[2] for p in pts]
            
            # Plot Path
            c = colors(i)
            ax.plot(xs, ys, color=c, linewidth=2, alpha=0.7, label=r_name)
            ax.plot(xs[0], ys[0], 'o', color=c); ax.plot(xs[-1], ys[-1], 'x', color=c)
            
            # Plot Time Ticks every 5s
            next_tick = 5.0
            for j in range(len(pts)):
                if ts[j] >= next_tick:
                    # White dot with robot-colored edge
                    ax.plot(xs[j], ys[j], 'o', color='white', markeredgecolor=c, markersize=5, zorder=20)
                    # Text with robot color
                    ax.text(xs[j], ys[j], f"{int(next_tick)}", fontsize=9, color='black', 
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                            ha='center', va='center', fontweight='bold', zorder=21)
                    next_tick += 5.0

        if self.debug_waits:
            for (x, y, dur, r_name) in self.debug_waits:
                if "(STUCK)" in str(r_name):
                    # Plot Black 'X' for stuck robots
                    ax.plot(x, y, 'kx', markersize=15, markeredgewidth=3, zorder=12, label='STUCK')
                    ax.text(x, y + 0.5, "STUCK", fontsize=9, color='black', ha='center', fontweight='bold', zorder=13)
                else:
                    radius = max(0.3, 0.2 + (dur / 2.0))
                    ax.add_patch(plt.Circle((x, y), radius, color='red', alpha=0.4, zorder=10))
                    if dur > 1.0: ax.text(x, y, f"{dur:.1f}", fontsize=8, color='darkred', ha='center', weight='bold', zorder=11)

        plt.legend(loc='upper right')
        plt.title("Execution Plan")
        plt.axis('equal'); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(filename); plt.close()