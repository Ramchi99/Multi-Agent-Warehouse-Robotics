import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import datetime
from typing import List, Dict, Any

class TournamentVisualizer:
    def __init__(self, output_dir="out/ex14/tournament_stats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data storage
        self.results = [] # List of dicts

    def record_result(self, 
                      name: str, 
                      theoretical_cost: float, 
                      actual_makespan: float, 
                      planning_time: float,
                      total_backtracks: int,
                      total_collisions: int,
                      total_iterations: int,
                      per_robot_times: List[float] = [],
                      per_robot_iters: List[int] = [],
                      is_valid: bool = True): # [NEW]
        """
        Records the statistics for a single tournament candidate.
        """
        self.results.append({
            "name": name,
            "theo": theoretical_cost,
            "actual": actual_makespan,
            "p_time": planning_time,
            "backtracks": total_backtracks,
            "collisions": total_collisions,
            "iterations": total_iterations,
            "robot_times": per_robot_times,
            "robot_iters": per_robot_iters,
            "valid": is_valid # [NEW]
        })

    def plot_all(self):
        if not self.results:
            print("No tournament results to plot.")
            return

        print(f"Generating Tournament Plots in {self.output_dir}...")
        
        # 1. Comparison: Theoretical vs Actual Makespan
        self._plot_makespan_comparison()
        
        # 2. Planner Performance (Time vs Difficulty)
        self._plot_planner_performance()
        
        # 3. Stability (Backtracks vs Collisions)
        self._plot_stability_metrics()

    def _plot_makespan_comparison(self):
        names = [r['name'] for r in self.results]
        theo = [r['theo'] for r in self.results]
        
        # Handle Actuals: If invalid, use 0 for bar height but label as FAILED
        actual_heights = []
        actual_labels = []
        
        for r in self.results:
            if r['valid']:
                actual_heights.append(r['actual'])
                actual_labels.append(f"{r['actual']:.1f}")
            else:
                actual_heights.append(0) # No bar
                actual_labels.append("FAILED")

        x = np.arange(len(names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width/2, theo, width, label='Theoretical (Alloc)', color='skyblue', alpha=0.8)
        rects2 = ax.bar(x + width/2, actual_heights, width, label='Actual (Physics)', color='salmon', alpha=0.8)
        
        # Add values on top
        ax.bar_label(rects1, padding=3, fmt='%.1f')
        ax.bar_label(rects2, labels=actual_labels, padding=3, fmt='%s', color='red', fontweight='bold')
        
        ax.set_ylabel('Makespan (Seconds)')
        ax.set_title('Tournament Results: Theoretical vs Physics Execution')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        fig.tight_layout()
        plt.savefig(self.output_dir / f"makespan_comparison_{self.timestamp}.png", dpi=150)
        plt.close()

    def _plot_planner_performance(self):
        names = [r['name'] for r in self.results]
        p_times = [r['p_time'] for r in self.results]
        iters = [r['iterations'] for r in self.results]
        backtracks = [r['backtracks'] for r in self.results]
        
        # Calculate IPS Labels
        bar_labels = []
        for r, t, i, b in zip(self.results, p_times, iters, backtracks):
            ips = i / t if t > 0.001 else 0
            status = "" if r['valid'] else "\n(FAILED)"
            bar_labels.append(f"{t:.2f}s\n({int(ips)} it/s)\nBT: {b}{status}")

        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # --- Primary Axis: Stacked Planning Time ---
        color_cycle = plt.cm.tab20.colors 
        
        max_robots = 0
        for r in self.results:
            max_robots = max(max_robots, len(r['robot_times']))
            
        bottoms = np.zeros(len(self.results))
        
        for r_idx in range(max_robots):
            row_times = []
            for res in self.results:
                if r_idx < len(res['robot_times']):
                    row_times.append(res['robot_times'][r_idx])
                else:
                    row_times.append(0.0)
            
            c = color_cycle[r_idx % len(color_cycle)]
            ax1.bar(names, row_times, bottom=bottoms, color=c, edgecolor='white', width=0.6, label=f"Robot {r_idx+1}" if r_idx < 5 else "") 
            bottoms += np.array(row_times)

        ax1.set_ylabel('Planning Time (s)', color='tab:blue')
        ax1.set_xlabel('Candidate')
        
        # Add labels to bars
        # For stacked bars, ax.bar_label is tricky. We manually annotate using the 'bottoms' (which is now total height)
        for i, (label, height) in enumerate(zip(bar_labels, bottoms)):
            ax1.text(i, height + 0.05, label, ha='center', va='bottom', fontsize=9)
        
        # Secondary Axis: Cumulative Iterations
        ax2 = ax1.twinx()
        ax2.set_ylabel('Cumulative Iterations', color='tab:green')
        
        for x_idx, res in enumerate(self.results):
            cum_iters = np.cumsum(res['robot_iters'])
            x_vals = [x_idx] * len(cum_iters)
            ax2.plot(x_vals, cum_iters, marker='_', color='tab:green', linestyle='None', markersize=10, alpha=0.5)
            ax2.plot(x_vals, cum_iters, color='tab:green', linestyle=':', alpha=0.3)
            if len(cum_iters) > 0:
                ax2.plot(x_idx, cum_iters[-1], marker='D', color='tab:green', markersize=6)

        ax2.set_ylim(bottom=0)
        
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper left', ncol=2, fontsize='small')

        plt.title('Planner Efficiency: Per-Robot Time (Stacked) & Cumulative Iterations')
        fig.tight_layout()
        plt.grid(True, alpha=0.2)
        plt.savefig(self.output_dir / f"planner_efficiency_{self.timestamp}.png", dpi=150)
        plt.close()

    def _plot_stability_metrics(self):
        # Scatter plot: Theo Cost vs Actual Cost
        valid_res = [r for r in self.results if r['valid']]
        invalid_res = [r for r in self.results if not r['valid']]
        
        plt.figure(figsize=(10, 10))
        
        # Plot Valid
        if valid_res:
            theo = [r['theo'] for r in valid_res]
            actual = [r['actual'] for r in valid_res]
            names = [r['name'] for r in valid_res]
            colors = np.random.rand(len(valid_res))
            plt.scatter(theo, actual, c=colors, s=200, cmap='viridis', alpha=0.8, label='Valid')
            
            for i, txt in enumerate(names):
                plt.annotate(txt, (theo[i], actual[i]), xytext=(5, 5), textcoords='offset points')
                
            # Diagonal line
            max_val = max(max(theo), max(actual)) + 5
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Ideal')

        # Plot Invalid (as separate markers at the bottom or separate list)
        if invalid_res:
            # We plot them at y=0 but with distinct marker
            theo_fail = [r['theo'] for r in invalid_res]
            names_fail = [r['name'] for r in invalid_res]
            # Use a fixed Y-value relative to the plot scale, or just 0
            plt.scatter(theo_fail, [0]*len(theo_fail), marker='x', color='red', s=200, label='Failed/Timeout')
            for i, txt in enumerate(names_fail):
                plt.annotate(txt, (theo_fail[i], 0), xytext=(5, 5), textcoords='offset points', color='red')

        plt.xlabel('Theoretical Cost (Allocator)')
        plt.ylabel('Actual Makespan (Physics)')
        plt.title('Reality Gap Analysis (Failed runs at Y=0)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)
        
        # Ensure 0 is visible
        plt.ylim(bottom=-1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"reality_gap_{self.timestamp}.png", dpi=150)
        plt.close()
