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
            'dists': [],           # [NEW] Cumulative Distance
            'vs': [],              # [NEW] Linear Velocity
            'ws': [],              # [NEW] Angular Velocity
            'backtracks': [],      # (iter, t_before, t_after, x, y)
            'collisions': [],      # (x, y)
            'targets': [],          # Target Index at this iteration
            'stagnations': [],      # (iter, time)
            'waits': [],            # (iter, time) [NEW]
            'wall_time': 0.0        # [NEW]
        }

    def record_iteration(self, iter_idx, sim_time, target_idx=0, dist=0.0, v=0.0, w=0.0):
        if not self.current_robot: return
        self.logs[self.current_robot]['iterations'].append(int(iter_idx))
        self.logs[self.current_robot]['sim_times'].append(float(sim_time))
        self.logs[self.current_robot]['targets'].append(int(target_idx))
        self.logs[self.current_robot]['dists'].append(float(dist))
        self.logs[self.current_robot]['vs'].append(float(v))
        self.logs[self.current_robot]['ws'].append(float(w))
    
    def record_planning_time(self, duration):
        if not self.current_robot: return
        self.logs[self.current_robot]['wall_time'] = float(duration)
        
    def record_wait(self, iter_idx, sim_time):
        if not self.current_robot: return
        self.logs[self.current_robot]['waits'].append((int(iter_idx), float(sim_time)))
        
    def record_stagnation(self, iter_idx, sim_time):
        if not self.current_robot: return
        self.logs[self.current_robot]['stagnations'].append((int(iter_idx), float(sim_time)))

    def record_collision(self, iter_idx, sim_time, x, y):
        if not self.current_robot: return
        self.logs[self.current_robot]['collisions'].append((int(iter_idx), float(sim_time), float(x), float(y)))

    def record_backtrack(self, iter_idx, sim_time_before, sim_time_after, moves_popped):
        if not self.current_robot: return
        # Store (iter, t_before, t_after, moves_popped)
        self.logs[self.current_robot]['backtracks'].append((int(iter_idx), float(sim_time_before), float(sim_time_after), int(moves_popped)))

    def export_logs_to_json(self):
        """Exports raw logs to JSON for analysis."""
        filepath = self.output_dir / f"planner_logs_{self.timestamp}.json"
        try:
            with open(filepath, "w") as f:
                json.dump(self.logs, f, indent=2)
            print(f"Exported planner logs to {filepath}")
        except Exception as e:
            print(f"Failed to export JSON logs: {e}")

    def _export_interactive_plot(self, r_name, data):
        """Generates a standalone HTML file with interactive Plotly.js visualization."""
        if not data['iterations']: return
        
        # Prepare Data
        iters = list(data['iterations']) # Ensure lists
        times = list(data['sim_times'])
        dists = list(data.get('dists', [0]*len(iters)))
        vs = list(data.get('vs', [0]*len(iters)))
        ws = list(data.get('ws', [0]*len(iters)))
        targets = list(data['targets'])
        collisions = data.get('collisions', [])
        
        # Collision markers
        c_x = [c[0] for c in collisions]
        c_y = [c[1] for c in collisions]
        
        # --- Prepare Colored Segments ---
        # Categories: Move, Wait, Turn, Backtrack1, Backtrack2, Backtrack3, Backtrack4, Backtrack5
        
        traces = {
            'move': {'x': [], 'y': [], 'color': 'green', 'name': 'Move (v>0)', 'width': 1.5},
            'wait': {'x': [], 'y': [], 'color': 'blue', 'name': 'Wait (v=0)', 'width': 2.5},
            'turn': {'x': [], 'y': [], 'color': 'purple', 'name': 'Turn (v=0, w!=0)', 'width': 2.0},
            'backtrack1': {'x': [], 'y': [], 'color': 'orange', 'name': 'Backtrack (Small, 1)', 'width': 1.5},
            'backtrack2': {'x': [], 'y': [], 'color': 'red', 'name': 'Backtrack (Medium, 2)', 'width': 2.0},
            'backtrack3': {'x': [], 'y': [], 'color': 'black', 'name': 'Backtrack (Large, 5)', 'width': 2.5},
            'backtrack4': {'x': [], 'y': [], 'color': 'darkviolet', 'name': 'Backtrack (Huge, 10)', 'width': 3.0},
            'backtrack5': {'x': [], 'y': [], 'color': 'sienna', 'name': 'Backtrack (Massive, 15)', 'width': 3.5},
        }
        
        # Create Lookup for Backtracks: Iter -> Moves Popped
        backtrack_map = {entry[0]: entry[3] for entry in data.get('backtracks', []) if len(entry) >= 4}
        
        import numpy as np
        # Iterate through segments
        for i in range(len(iters) - 1):
            x0, x1 = iters[i], iters[i+1]
            y0, y1 = times[i], times[i+1]
            dt = y1 - y0
            
            # Check for Backtrack Event first
            next_iter = iters[i+1]
            cat = None
            
            if next_iter in backtrack_map:
                moves = backtrack_map[next_iter]
                if moves == 1: cat = 'backtrack1'
                elif moves == 2: cat = 'backtrack2'
                elif moves == 5: cat = 'backtrack3'
                elif moves == 10: cat = 'backtrack4'
                elif moves >= 15: cat = 'backtrack5'
                else: cat = 'backtrack3' # Default fallback
            elif dt < -1e-6:
                # Fallback
                cat = 'backtrack3'
            else:
                # Forward (Time increased or same)
                v_cmd = vs[i+1]
                w_cmd = ws[i+1]
                
                if v_cmd > 1e-6: cat = 'move'
                elif abs(w_cmd) > 1e-6: cat = 'turn'
                else: cat = 'wait'
            
            if cat:
                traces[cat]['x'].extend([x0, x1, None])
                traces[cat]['y'].extend([y0, y1, None])

        # --- Generate Shapes for Target Zones ---
        shapes = []
        annotations = []
        
        targets_np = np.array(targets)
        
        change_indices = np.where(np.diff(targets_np) != 0)[0] + 1
        boundaries = np.concatenate(([0], change_indices, [len(iters)-1]))
        
        # Simple color cycle for zones
        zone_colors = ['rgba(255, 0, 0, 0.1)', 'rgba(0, 255, 0, 0.1)', 'rgba(0, 0, 255, 0.1)', 'rgba(255, 255, 0, 0.1)']
        
        for k in range(len(boundaries)-1):
            start_idx = boundaries[k]
            end_idx = boundaries[k+1]
            t_val = targets[start_idx]
            
            x0_zone = iters[start_idx]
            x1_zone = iters[end_idx]
            
            color = zone_colors[t_val % len(zone_colors)]
            
            shapes.append({
                'type': 'rect',
                'xref': 'x', 'yref': 'paper',
                'x0': x0_zone, 'y0': 0,
                'x1': x1_zone, 'y1': 1,
                'fillcolor': color,
                'line': {'width': 0},
                'layer': 'below'
            })
            
            # Add Label
            annotations.append({
                'x': (x0_zone + x1_zone) / 2,
                'y': 1,
                'xref': 'x', 'yref': 'paper',
                'text': f"WP {t_val}",
                'showarrow': False,
                'font': {'color': 'gray'}
            })

        # --- HTML Template ---
        # Construct JS Traces
        js_traces = []
        
        # 1. Add Segment Traces (Left Axis)
        for cat, info in traces.items():
            if not info['x']: continue
            js_traces.append(f"""
            {{
                x: {json.dumps(info['x'])},
                y: {json.dumps(info['y'])},
                mode: 'lines',
                name: '{info['name']}',
                line: {{color: '{info['color']}', width: {info['width']}}},
                hoverinfo: 'skip'
            }}""")
            
        # 2. Add Path Length Trace (Right Axis)
        if dists:
            js_traces.append(f"""
            {{
                x: {json.dumps(iters)},
                y: {json.dumps(dists)},
                mode: 'lines',
                name: 'Path Length',
                line: {{color: 'gray', width: 1, dash: 'dash'}},
                yaxis: 'y2',
                opacity: 0.5
            }}""")

        # 3. Add Collisions
        js_traces.append(f"""
            {{
                x: {c_x},
                y: {c_y},
                mode: 'markers',
                name: 'Collisions',
                marker: {{color: 'red', symbol: 'x', size: 10}}
            }}""")

        js_traces_str = ",\n".join(js_traces)

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Search Genealogy: {r_name}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>body {{ font-family: sans-serif; margin: 0; padding: 20px; }}</style>
</head>
<body>
    <h2>Search Genealogy: {r_name}</h2>
    <div id="chart" style="width:100%;height:800px;"></div>
    <script>
        var data = [{js_traces_str}];
        
        var layout = {{
            title: 'Search Timeline',
            xaxis: {{title: 'Iteration'}},
            yaxis: {{title: 'Sim Time (s)'}},
            yaxis2: {{
                title: 'Path Length (m)',
                overlaying: 'y',
                side: 'right',
                showgrid: false
            }},
            hovermode: 'closest',
            shapes: {json.dumps(shapes)},
            annotations: {json.dumps(annotations)},
            legend: {{x: 0, y: 1}}
        }};
        
        Plotly.newPlot('chart', data, layout);
    </script>
</body>
</html>
"""
        filepath = self.output_dir / "progress_plots" / f"interactive_{r_name}_{self.timestamp}.html"
        try:
            with open(filepath, "w") as f:
                f.write(html_content)
        except Exception as e:
            print(f"Failed to write HTML: {e}")

    def _export_interactive_physical_plot(self, r_name, data):
        """Generates interactive Plotly.js visualization for Physical Progression."""
        if not data['iterations']: return
        
        # Prepare Data
        iters = list(data['iterations'])
        times = list(data['sim_times'])
        dists = list(data.get('dists', [0]*len(iters)))
        vs = list(data.get('vs', [0]*len(iters)))
        ws = list(data.get('ws', [0]*len(iters)))
        targets = list(data['targets'])
        collisions = data.get('collisions', [])
        
        # Collision markers (Interpolate Dist)
        c_x = [c[0] for c in collisions]
        c_y = []
        for cx in c_x:
            idx = int(cx)
            if 0 <= idx < len(dists):
                c_y.append(dists[idx])
            else:
                c_y.append(0)
        
        # --- Prepare Colored Segments (Based on DISTANCE Change & Backtrack Log) ---
        traces = {
            'move': {'x': [], 'y': [], 'color': 'green', 'name': 'Move (Dist+)', 'width': 1.5},
            'wait': {'x': [], 'y': [], 'color': 'blue', 'name': 'Wait (Dist=)', 'width': 2.5},
            'turn': {'x': [], 'y': [], 'color': 'purple', 'name': 'Turn (Dist=)', 'width': 2.0},
            'backtrack1': {'x': [], 'y': [], 'color': 'orange', 'name': 'Backtrack (Small, 1)', 'width': 1.5},
            'backtrack2': {'x': [], 'y': [], 'color': 'red', 'name': 'Backtrack (Medium, 2)', 'width': 2.0},
            'backtrack3': {'x': [], 'y': [], 'color': 'black', 'name': 'Backtrack (Large, 5)', 'width': 2.5},
            'backtrack4': {'x': [], 'y': [], 'color': 'darkviolet', 'name': 'Backtrack (Huge, 10)', 'width': 3.0},
            'backtrack5': {'x': [], 'y': [], 'color': 'sienna', 'name': 'Backtrack (Massive, 15)', 'width': 3.5},
        }
        
        # Create Lookup for Backtracks: Iter -> Moves Popped
        backtrack_map = {entry[0]: entry[3] for entry in data.get('backtracks', []) if len(entry) >= 4}
        
        for i in range(len(iters) - 1):
            x0, x1 = iters[i], iters[i+1]
            y0, y1 = dists[i], dists[i+1]
            dd = y1 - y0
            
            # Check for Backtrack Event first
            next_iter = iters[i+1]
            cat = None
            
            if next_iter in backtrack_map:
                moves = backtrack_map[next_iter]
                if moves == 1: cat = 'backtrack1'
                elif moves == 2: cat = 'backtrack2'
                elif moves == 5: cat = 'backtrack3'
                elif moves == 10: cat = 'backtrack4'
                elif moves >= 15: cat = 'backtrack5'
                else: cat = 'backtrack3'
            elif dd < -1e-6:
                # Fallback
                cat = 'backtrack3'
            elif dd > 1e-6:
                cat = 'move'
            else:
                w_cmd = ws[i+1]
                if abs(w_cmd) > 1e-6: cat = 'turn'
                else: cat = 'wait'
            
            if cat:
                traces[cat]['x'].extend([x0, x1, None])
                traces[cat]['y'].extend([y0, y1, None])

        # --- Generate Shapes for Target Zones ---
        shapes = []
        annotations = []
        
        targets_np = np.array(targets)
        change_indices = np.where(np.diff(targets_np) != 0)[0] + 1
        boundaries = np.concatenate(([0], change_indices, [len(iters)-1]))
        zone_colors = ['rgba(255, 0, 0, 0.1)', 'rgba(0, 255, 0, 0.1)', 'rgba(0, 0, 255, 0.1)', 'rgba(255, 255, 0, 0.1)']
        
        for k in range(len(boundaries)-1):
            start_idx = boundaries[k]
            end_idx = boundaries[k+1]
            t_val = targets[start_idx]
            x0_zone = iters[start_idx]
            x1_zone = iters[end_idx]
            color = zone_colors[t_val % len(zone_colors)]
            shapes.append({
                'type': 'rect', 'xref': 'x', 'yref': 'paper',
                'x0': x0_zone, 'y0': 0, 'x1': x1_zone, 'y1': 1,
                'fillcolor': color, 'line': {'width': 0}, 'layer': 'below'
            })
            annotations.append({
                'x': (x0_zone + x1_zone) / 2, 'y': 1, 'xref': 'x', 'yref': 'paper',
                'text': f"WP {t_val}", 'showarrow': False, 'font': {'color': 'gray'}
            })

        # --- HTML Template ---
        js_traces = []
        for cat, info in traces.items():
            if not info['x']: continue
            js_traces.append(f"""
            {{
                x: {json.dumps(info['x'])},
                y: {json.dumps(info['y'])},
                mode: 'lines',
                name: '{info['name']}',
                line: {{color: '{info['color']}', width: {info['width']}}},
                hoverinfo: 'skip'
            }}""")
            
        # Add Time Trace (Secondary Axis)
        js_traces.append(f"""
        {{
            x: {json.dumps(iters)},
            y: {json.dumps(times)},
            mode: 'lines',
            name: 'Sim Time',
            line: {{color: 'gray', width: 1, dash: 'dash'}},
            yaxis: 'y2',
            opacity: 0.5
        }}""")

        # Add Collisions
        js_traces.append(f"""
        {{
            x: {c_x},
            y: {c_y},
            mode: 'markers',
            name: 'Collisions',
            marker: {{color: 'red', symbol: 'x', size: 10}}
        }}""")

        js_traces_str = ",\n".join(js_traces)

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Physical Progression: {r_name}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>body {{ font-family: sans-serif; margin: 0; padding: 20px; }}</style>
</head>
<body>
    <h2>Physical Progression: {r_name}</h2>
    <div id="chart" style="width:100%;height:800px;"></div>
    <script>
        var data = [{js_traces_str}];
        var layout = {{
            title: 'Physical Progression (Cumulative Distance)',
            xaxis: {{title: 'Iteration'}},
            yaxis: {{title: 'Cumulative Path Length (m)'}},
            yaxis2: {{
                title: 'Sim Time (s)',
                overlaying: 'y',
                side: 'right',
                showgrid: false
            }},
            hovermode: 'closest',
            shapes: {json.dumps(shapes)},
            annotations: {json.dumps(annotations)},
            legend: {{x: 0, y: 1}}
        }};
        Plotly.newPlot('chart', data, layout);
    </script>
</body>
</html>
"""
        filepath = self.output_dir / "progress_plots" / f"interactive_physical_{r_name}_{self.timestamp}.html"
        try:
            with open(filepath, "w") as f:
                f.write(html_content)
        except Exception as e:
            print(f"Failed to write HTML: {e}")

    def plot_summary(self, static_obstacles: List[Polygon]):
        """Generates comprehensive plots."""
        print(f"Generating Debug Plots in {self.output_dir}...")
        
        # 1. Search Progress (Static & Interactive)
        self._plot_physical_genealogy()
        
        # Generate Interactive HTML for each robot
        for r_name, data in self.logs.items():
            self._export_interactive_plot(r_name, data)
            self._export_interactive_physical_plot(r_name, data)
        
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

    def _plot_physical_genealogy(self):
        """
        Generates the 'Physical Progression' plot: A timeline of Physical Distance covered.
        Y-Axis: Cumulative Distance (m).
        X-Axis: Iteration.
        Colors:
          - Green: Moving (Distance increases).
          - Blue: Waiting (Distance flat, w=0).
          - Purple: Turning (Distance flat, w!=0).
          - Orange/Red/Black: Backtracking (Distance decreases).
        """
        for r_name, data in self.logs.items():
            if not data['iterations']: continue
            
            iters = np.array(data['iterations'])
            times = np.array(data['sim_times'])
            dists = np.array(data.get('dists', [0]*len(iters)))
            vs = np.array(data.get('vs', [0]*len(iters)))
            ws = np.array(data.get('ws', [0]*len(iters)))
            targets = np.array(data['targets'])
            collisions = data.get('collisions', [])
            
            # Setup Plot with Twin Axes
            fig, ax1 = plt.subplots(figsize=(24, 12))
            ax2 = ax1.twinx() # For Time
            
            # 1. Background Progress Zones (Target Index)
            unique_targets = np.unique(targets)
            norm = Normalize(vmin=min(unique_targets), vmax=max(unique_targets))
            cmap = cm.get_cmap('Pastel1') 
            
            change_indices = np.where(np.diff(targets) != 0)[0] + 1
            boundaries = np.concatenate(([0], change_indices, [len(iters)-1]))
            
            for k in range(len(boundaries)-1):
                start_idx = boundaries[k]
                end_idx = boundaries[k+1]
                t_val = targets[start_idx]
                
                x_start = iters[start_idx]
                x_end = iters[end_idx]
                
                color = cmap(norm(t_val))
                ax1.axvspan(x_start, x_end, color=color, alpha=0.3, lw=0)
                
                mid_x = (x_start + x_end) / 2
                if (x_end - x_start) > (iters[-1] * 0.02):
                    ax1.text(mid_x, ax1.get_ylim()[1], f"WP {t_val}", ha='center', va='top', fontsize=8, color='grey')

            # 2. Construct Colored Line Segments for DISTANCE (Left Axis)
            points = np.array([iters, dists]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create Lookup for Backtracks: Iter -> Moves Popped
            backtrack_map = {entry[0]: entry[3] for entry in data.get('backtracks', []) if len(entry) >= 4}
            
            colors = []
            linewidths = []
            
            for i in range(len(iters) - 1):
                d0, d1 = dists[i], dists[i+1]
                dd = d1 - d0
                
                # Check if this segment (leading to iters[i+1]) was a Backtrack
                next_iter = iters[i+1]
                
                if next_iter in backtrack_map:
                    # It IS a backtrack
                    moves = backtrack_map[next_iter]
                    if moves == 1:
                        colors.append('tab:orange')
                        linewidths.append(1.5)
                    elif moves == 2:
                        colors.append('tab:red')
                        linewidths.append(2.0)
                    else: # >= 5 or others
                        colors.append('black')
                        linewidths.append(2.5)
                elif dd < -1e-6:
                    # Fallback for negative distance not in backtrack log (shouldn't happen with correct logging)
                    colors.append('black')
                    linewidths.append(1.0)
                elif dd > 1e-6:
                    # Forward Movement
                    colors.append('tab:green')
                    linewidths.append(1.5)
                else:
                    # Distance Flat (Wait or Turn)
                    # Check Angular Velocity
                    w_cmd = ws[i+1]
                    if abs(w_cmd) > 1e-6:
                        colors.append('tab:purple') # Turn
                        linewidths.append(2.0)
                    else:
                        colors.append('tab:blue') # Wait
                        linewidths.append(2.5)
            
            lc = LineCollection(segments, colors=colors, linewidths=linewidths, alpha=0.9)
            ax1.add_collection(lc)
            
            # 3. Plot Time (Right Axis)
            ax2.plot(iters, times, color='gray', linestyle='--', linewidth=1.0, alpha=0.6, label='Sim Time')

            # 4. Plot Collisions (using dists for Y)
            if collisions:
                c_iters = [c[0] for c in collisions]
                # We need to interpolate Y (Distance) for these X (Iter)
                # Simple lookup since we have dense data now
                c_dists = []
                for cx in c_iters:
                    idx = int(cx)
                    if 0 <= idx < len(dists):
                        c_dists.append(dists[idx])
                    else:
                        c_dists.append(0)
                ax1.scatter(c_iters, c_dists, marker='x', color='red', s=40, zorder=5, label='Collision')

            # Formatting
            ax1.autoscale()
            ax1.set_xlabel('Planner Iteration')
            ax1.set_ylabel('Cumulative Path Length (m)', color='black')
            ax2.set_ylabel('Sim Time (s)', color='gray')
            
            ax1.set_title(f'Physical Progression: {r_name} (Max Dist: {np.max(dists):.2f}m)')
            
            # Custom Legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='tab:green', lw=2, label='Move (Dist+)'),
                Line2D([0], [0], color='tab:blue', lw=2, label='Wait (Dist=)'),
                Line2D([0], [0], color='tab:purple', lw=2, label='Turn (Dist=)'),
                Line2D([0], [0], color='tab:orange', lw=2, label='Backtrack (Small)'),
                Line2D([0], [0], color='tab:red', lw=2, label='Backtrack (Medium)'),
                Line2D([0], [0], color='black', lw=2, label='Backtrack (Large)'),
                Line2D([0], [0], color='gray', linestyle='--', lw=1, label='Sim Time'),
                Line2D([0], [0], marker='x', color='red', lw=0, label='Collision'),
            ]
            ax1.legend(handles=legend_elements, loc='upper left')
            
            ax1.minorticks_on()
            ax1.grid(True, which='major', linestyle='-', alpha=0.4)
            ax1.grid(True, which='minor', linestyle=':', alpha=0.2)
            
            # Save
            prog_dir = self.output_dir / "progress_plots"
            prog_dir.mkdir(exist_ok=True)
            plt.savefig(prog_dir / f"physical_{r_name}_{self.timestamp}.png", dpi=150)
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

        # Unpack (iter, time, x, y)
        _, _, cx_all, cy_all = zip(*all_collisions)

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
