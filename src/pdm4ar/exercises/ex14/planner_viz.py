import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import math
import json
import datetime # [NEW]
from pathlib import Path
from typing import List, Dict, Tuple
from shapely.geometry import Polygon
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

class PlannerDebugger:
    def __init__(self, output_dir="out/ex14/debug_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # [NEW]
        
        # Structure: {robot_name: {'iterations': [], 'times': [], 'collisions': [], 'backtracks': [], 'targets': []}}
        self.logs = {}
        self.current_robot = None

    def start_robot(self, robot_name):
        self.current_robot = robot_name
        self.logs[robot_name] = {
            'iterations': [],      
            'sim_times': [],       
            'backtracks': [],      # (iter, t_before, t_after, x, y)
            'collisions': [],      # (x, y)
            'targets': [],          # Target Index at this iteration
            'stagnations': [],      # (iter, time)
            'waits': [],            # (iter, time) [NEW]
            'wall_time': 0.0        # [NEW]
        }

    def record_iteration(self, iter_idx, sim_time, target_idx=0):
        if not self.current_robot: return
        self.logs[self.current_robot]['iterations'].append(int(iter_idx))
        self.logs[self.current_robot]['sim_times'].append(float(sim_time))
        self.logs[self.current_robot]['targets'].append(int(target_idx))
    
    def record_planning_time(self, duration):
        if not self.current_robot: return
        self.logs[self.current_robot]['wall_time'] = float(duration)
        
    def record_wait(self, iter_idx, sim_time):
        if not self.current_robot: return
        self.logs[self.current_robot]['waits'].append((int(iter_idx), float(sim_time)))
        
    def record_stagnation(self, iter_idx, sim_time):
        if not self.current_robot: return
        self.logs[self.current_robot]['stagnations'].append((int(iter_idx), float(sim_time)))

    def record_collision(self, x, y):
        if not self.current_robot: return
        self.logs[self.current_robot]['collisions'].append((float(x), float(y)))

    def record_backtrack(self, iter_idx, sim_time_before, sim_time_after):
        if not self.current_robot: return
        # Note: We don't have x,y here easily, but we can infer it later if needed or add it to args
        # For now, we just track time jumps.
        self.logs[self.current_robot]['backtracks'].append((int(iter_idx), float(sim_time_before), float(sim_time_after)))

    def export_logs_to_json(self):
        """Exports raw logs to JSON for analysis."""
        filepath = self.output_dir / f"planner_logs_{self.timestamp}.json"
        try:
            with open(filepath, "w") as f:
                json.dump(self.logs, f, indent=2)
            print(f"Exported planner logs to {filepath}")
        except Exception as e:
            print(f"Failed to export JSON logs: {e}")

    def plot_summary(self, static_obstacles: List[Polygon]):
        """Generates comprehensive plots."""
        print(f"Generating Debug Plots in {self.output_dir}...")
        
        # self.export_logs_to_json() 
        
        # 1. Search Progress (Time & Targets)
        self._plot_search_progress()
        
        # 2. Conflict Heatmap
        self._plot_conflict_heatmap(static_obstacles)
        
        # 3. Wait Time Statistics
        self._print_wait_statistics()

    def _print_wait_statistics(self):
        print("\n--- Wait Time Statistics ---")
        for r_name, data in self.logs.items():
            times = data['sim_times']
            iters = data['iterations']
            if not times: continue
            
            final_time = times[-1]
            last_iter = iters[-1] if iters else 0
            
            print(f"Robot {r_name}:")
            print(f"  Final Time Reached: {final_time:.2f}s")
            print(f"  Max Iteration:      {last_iter}")
            print(f"  Backtracks:         {len(data['backtracks'])}")
            print(f"  Stagnation Resets:  {len(data['stagnations'])}")

    def _plot_search_progress(self):
        """Plots Simulation Time vs Iterations AND Target Index vs Iterations."""
        
        for r_name, data in self.logs.items():
            if not data['iterations']: continue
            
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            iters = data['iterations']
            times = data['sim_times']
            targets = data['targets']
            stagnations = data.get('stagnations', [])
            waits = data.get('waits', [])
            backtracks = data['backtracks']

            # --- Primary Axis: Simulation Time ---
            color = 'tab:blue'
            ax1.set_xlabel('Planner Iterations')
            ax1.set_ylabel('Simulation Time (s)', color=color)
            ax1.plot(iters, times, color=color, linewidth=1.0, alpha=0.8, label="Sim Time")
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Highlight Waits
            if waits:
                w_iters, w_times = zip(*waits)
                ax1.scatter(w_iters, w_times, color='orange', s=10, marker='.', label="Wait Step", zorder=3)
            
            # Highlight Stagnation Zones
            if stagnations:
                stag_iters, stag_times = zip(*stagnations)
                # Shade the area around stagnation? Or just a line.
                for i, t in stagnations:
                    ax1.axvline(x=i, color='red', linestyle='--', alpha=0.5)
                    ax1.text(i, t, " STAGNATION", rotation=90, verticalalignment='bottom', color='red', fontsize=8)

            # Highlight Backtracks (Color coded segments)
            if backtracks:
                segments = []
                magnitudes = []
                for (i, t_before, t_after) in backtracks:
                    seg = [(i, t_before), (i, t_after)]
                    mag = abs(t_before - t_after)
                    segments.append(seg)
                    magnitudes.append(mag)
                
                if segments:
                    norm = Normalize(vmin=0, vmax=2.0)
                    cmap = cm.get_cmap('autumn_r')
                    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.5, alpha=0.9, zorder=5)
                    lc.set_array(np.array(magnitudes))
                    ax1.add_collection(lc)
                    # cbar = fig.colorbar(lc, ax=ax1, label='Backtrack Size (s)', pad=0.01)

            # --- Secondary Axis: Target Index ---
            ax2 = ax1.twinx()  
            color = 'tab:green'
            ax2.set_ylabel('Target Waypoint Index', color=color)  
            ax2.plot(iters, targets, color=color, linewidth=1.5, linestyle='-', alpha=0.6, label="Target Idx")
            ax2.tick_params(axis='y', labelcolor=color)

            # Final Time Box
            final_t = times[-1] if times else 0
            plt.title(f"Search Progress: {r_name} (Final T = {final_t:.2f}s)")
            
            fig.tight_layout()  
            plt.grid(True, alpha=0.2)
            
            # [NEW] Save to dedicated folder
            prog_dir = self.output_dir / "progress_plots"
            prog_dir.mkdir(exist_ok=True)
            plt.savefig(prog_dir / f"progress_{r_name}_{self.timestamp}.png", dpi=150)
            plt.close()

    def _plot_conflict_heatmap(self, static_obstacles):
        """Plots where collisions occurred most frequently using Density."""
        if not self.logs: return

        # Gather ALL collisions
        all_collisions = []
        for data in self.logs.values():
            all_collisions.extend(data['collisions'])
        
        if not all_collisions:
            print("No collisions recorded to plot.")
            return

        cx_all, cy_all = zip(*all_collisions)

        plt.figure(figsize=(12, 12))
        
        # Calculate Extent from Obstacles to ensure consistent grid size
        from shapely.ops import unary_union
        if static_obstacles:
            # Handle list of polygons or MultiPolygons
            # Note: static_obstacles might be a mix of Polygons and MultiPolygons
            combined = unary_union(static_obstacles)
            minx, miny, maxx, maxy = combined.bounds
            # Add a small buffer to the extent
            extent = (minx - 1, maxx + 1, miny - 1, maxy + 1)
        else:
            extent = (-12, 12, -12, 12)

        # 1. Plot Density (Hexbin)
        # gridsize=20 divides the EXTENT into 20 bins. 
        # Since extent is now fixed to the map size, hexes will be consistent.
        hb = plt.hexbin(cx_all, cy_all, gridsize=30, extent=extent, cmap='inferno_r', mincnt=1, alpha=0.7, zorder=1)
        cb = plt.colorbar(hb, label='Collision Count')

        # 2. Plot Static Obstacles (Overlaid)
        for poly in static_obstacles:
            if hasattr(poly, 'geoms'): geoms = poly.geoms
            else: geoms = [poly]
            for g in geoms:
                try:
                    if g.geom_type == 'Polygon':
                        x, y = g.exterior.xy
                        plt.fill(x, y, color='black', alpha=0.4, zorder=2) # Darker
                        plt.plot(x, y, color='white', linewidth=1, zorder=2) # White outline for contrast
                    elif g.geom_type in ['LinearRing', 'LineString']:
                        x, y = g.xy
                        plt.plot(x, y, color='black', linewidth=3, alpha=0.6, zorder=2)
                except AttributeError:
                    pass

        plt.title("Collision Density Heatmap (All Robots)")
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"debug_conflict_heatmap_{self.timestamp}.png", dpi=150)
        plt.close()

class PlannerVisualizer:
    def __init__(self, robot_radius=0.7):
        self.robot_radius = robot_radius

    def plot_prm(self, G, obstacles, special_nodes, filename, bounds=None, path_data=None, final_paths=None):
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

    def plot_trajectory_comparison(self, waypoints_dict, final_plans_6d, obstacles, filename):
        """
        Plots the Raw Waypoints vs the Calculated Physics Trajectory.
        Robust to different Shapely geometry types (Polygon, LinearRing, etc.)
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(24, 24)) 
        
        # 1. Plot Obstacles (Robustly)
        for poly in obstacles:
            # Handle Multi-geometries (MultiPolygon, GeometryCollection)
            geoms = poly.geoms if hasattr(poly, "geoms") else [poly]
            
            for geom in geoms:
                try:
                    if geom.geom_type == 'Polygon':
                        # Polygons have an exterior
                        x, y = geom.exterior.xy
                        plt.fill(x, y, color="gray", alpha=0.5, label="Obstacle" if "Obstacle" not in plt.gca().get_legend_handles_labels()[1] else "")
                    elif geom.geom_type in ['LinearRing', 'LineString']:
                        # LinearRings are just lines (no exterior attribute)
                        x, y = geom.xy
                        plt.plot(x, y, color="gray", linewidth=2, alpha=0.5, label="Obstacle" if "Obstacle" not in plt.gca().get_legend_handles_labels()[1] else "")
                    else:
                        print(f"Warning: Skipping unsupported geometry type: {geom.geom_type}")
                except Exception as e:
                    print(f"Error plotting geometry: {e}")

        # 2. Plot Paths per Robot
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        
        for i, r_name in enumerate(final_plans_6d.keys()):
            c = colors[i % len(colors)]
            
            # A. Plot Raw Waypoints (The Input)
            if r_name in waypoints_dict:
                wps = waypoints_dict[r_name]
                if wps:
                    wx, wy = zip(*wps)
                    plt.plot(wx, wy, color=c, marker='x', linestyle='--', linewidth=1, markersize=8, alpha=0.5, label=f"{r_name} Raw Input")
            
            # B. Plot Calculated Physics Trajectory (The Output)
            plan = final_plans_6d[r_name]
            if plan:
                px = [p.x for p in plan]
                py = [p.y for p in plan]
                
                # Plot the line
                plt.plot(px, py, color=c, linewidth=2, label=f"{r_name} Physics Plan")
                
                # Plot Orientation Arrows (Subsample every ~1s)
                arrow_step = 10
                if len(plan) > arrow_step:
                    quiver_x = px[::arrow_step]
                    quiver_y = py[::arrow_step]
                    quiver_u = [math.cos(p.theta) for p in plan[::arrow_step]]
                    quiver_v = [math.sin(p.theta) for p in plan[::arrow_step]]
                    
                    plt.quiver(quiver_x, quiver_y, quiver_u, quiver_v, color=c, scale=20, width=0.003, alpha=0.8)

        plt.title(f"Trajectory Debug: Raw Input vs Physics Plan")
        plt.legend(loc="upper right")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Generated trajectory comparison plot: {filename}")

    def print_debug_comparison(self, sa_assignments, lns_assignments, cost_matrix, heading_matrix):
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

    def plot_convergence(self, histories, filename):
        plt.figure(figsize=(12, 8))

        # Determine global max time to extend plots to the right edge
        global_max_t = 0.0
        for h in histories.values():
            if h:
                global_max_t = max(global_max_t, h[-1][0])
        # Add a small buffer or assume time_limit was ~global_max_t
        global_max_t = max(global_max_t, 0.1)

        for name, history in histories.items():
            if not history:
                continue
            history.sort(key=lambda x: x[0])

            times = [t for t, c in history]
            costs = [c for t, c in history]

            # Extend the line to the global max time for visual comparison
            if times[-1] < global_max_t:
                times.append(global_max_t)
                costs.append(costs[-1])

            # Plot
            final_c = costs[-1]
            plt.step(times, costs, where="post", label=f"{name} ({final_c:.2f})", linewidth=2.5, alpha=0.8)
            plt.plot(times, costs, "o", markersize=5, alpha=0.6)  # Mark improvements

        plt.xlabel("Computation Time (s)", fontsize=12)
        plt.ylabel("Theoretical Cost", fontsize=12)
        plt.title("Optimization Convergence Profile", fontsize=14)
        plt.legend(fontsize=10, loc="best")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
