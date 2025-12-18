import math
import numpy as np
import time
from decimal import Decimal
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from shapely.geometry import Polygon

from dg_commons.sim.models.diff_drive import DiffDriveModel, DiffDriveState, DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons import apply_SE2_to_shapely_geo
import numpy as np

from .planner_viz import PlannerDebugger


def SE2_from_xytheta(xytheta):
    x, y, theta = xytheta
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])


@dataclass
class PlanPoint:
    x: float
    y: float
    theta: float
    t: float
    v: float
    w: float
    target_idx: int = 0
    accumulated_dist: float = 0.0


class ExactSpaceTimePlanner:
    def __init__(
        self, static_obstacles: List[Polygon], dt: float = 0.1, margin: float = 0.01, use_stagnation_logic: bool = True
    ):
        self.static_obstacles = static_obstacles
        self.dt = dt
        self.decimal_dt = Decimal(str(dt))
        self.margin = margin
        self.reservations: Dict[int, List[Polygon]] = {}
        self.debugger = PlannerDebugger()
        self.use_stagnation_logic = use_stagnation_logic
        self.MAX_ITERS = 10000 # [NEW]

    def plan_prioritized(self, robots_sequence, initial_states, waypoints_dict, geometries, params, time_limit: float = 60.0, best_known_makespan: float = float('inf')):
        final_plans = {}
        self.reservations.clear()
        total_start = time.time()
        deadline = total_start + time_limit

        start_footprints = {}
        for r_name in robots_sequence:
            s = initial_states[r_name]
            vg = geometries[r_name]
            footprint_poly = vg.outline_as_polygon.buffer(self.margin)
            tform = SE2_from_xytheta([s.x, s.y, s.psi])
            start_footprints[r_name] = apply_SE2_to_shapely_geo(footprint_poly, tform)

        for i, robot_name in enumerate(robots_sequence):
            # Global Timeout Check
            if time.time() > deadline:
                print(f"Global Timeout reached before planning {robot_name}. Aborting.")
                break
                
            # print(f"Planning physics execution for {robot_name}...")
            r_start = time.time()
            # self.debugger.start_robot(robot_name) # [DISABLED] Speed

            # [MODIFIED] Lower priority robots are GHOSTS. We do not treat them as static obstacles.
            # They must yield to us.
            temp_static_obstacles = []

            start_state = initial_states[robot_name]
            model = DiffDriveModel(x0=start_state, vg=geometries[robot_name], vp=params[robot_name])
            targets = waypoints_dict.get(robot_name, [])

            trajectory = self._plan_single_robot_backtracking(
                model, 
                targets, 
                extra_static_obstacles=temp_static_obstacles,
                deadline=deadline,
                pruning_threshold=best_known_makespan + 1.0 # [NEW] Pruning limit
            )
            
            if trajectory is None:
                # print(f"Planning for {robot_name} failed (Timeout/Pruned). Stopping global plan.")
                break
                
            final_plans[robot_name] = trajectory
            
            dur = max(0.0, time.time() - r_start)
            # self.debugger.record_planning_time(dur) # [NEW]
            # print(f"  -> Planned {robot_name} in {dur:.2f}s")

            footprint = model.vg.outline_as_polygon.buffer(self.margin)
            last_time_idx = 0
            final_poly = None

            for pt in trajectory:
                time_idx = int(round(pt.t / self.dt))
                last_time_idx = time_idx
                tform = SE2_from_xytheta([pt.x, pt.y, pt.theta])
                current_poly = apply_SE2_to_shapely_geo(footprint, tform)
                final_poly = current_poly

                if time_idx not in self.reservations:
                    self.reservations[time_idx] = []
                self.reservations[time_idx].append(current_poly)

            if final_poly:
                PARKING_HORIZON_STEPS = 2000
                for k in range(1, PARKING_HORIZON_STEPS):
                    future_idx = last_time_idx + k
                    if future_idx not in self.reservations:
                        self.reservations[future_idx] = []
                    self.reservations[future_idx].append(final_poly)

        # print(f"Total Exact Planning Time: {time.time() - total_start:.2f}s")
        # self.debugger.plot_summary(self.static_obstacles) # [DISABLED] Speed
        return final_plans

    def _plan_single_robot_backtracking(self, model, targets, extra_static_obstacles, deadline=None, pruning_threshold=float('inf')):
        trajectory: List[PlanPoint] = []
        current_time = 0.0
        current_dist = 0.0 # [NEW]
        initial_state_copy = DiffDriveState(x=model._state.x, y=model._state.y, psi=model._state.psi)
        
        # Log Initial State
        self.debugger.record_iteration(0, 0.0, 0, 0.0, 0.0, 0.0)

        # Physics Constants
        r = model.vg.wheelradius
        L = model.vg.wheelbase
        w_motor_max = model.vp.omega_limits[1]
        v_max = r * w_motor_max
        w_max = (2 * r * w_motor_max) / L

        def get_cmds_inverse(v_des, w_des):
            omega_r = (v_des / r) + (w_des * L / (2 * r))
            omega_l = (v_des / r) - (w_des * L / (2 * r))
            return DiffDriveCommands(omega_l=omega_l, omega_r=omega_r)

        def is_safe(next_poly, time_idx):
            # for obs in self.static_obstacles:
            #     if obs.intersects(next_poly):
            #         return False
            # for obs in extra_static_obstacles:
            #     if obs.intersects(next_poly):
            #         return False
            if time_idx in self.reservations:
                for reserved_poly in self.reservations[time_idx]:
                    if reserved_poly.intersects(next_poly):
                        return False
            return True

        target_idx = 0
        forced_wait_steps = 0
        
        # --- ESCALATION STATE ---
        max_progress_idx = 0
        consecutive_backtrack_count = 0 # [MODIFIED]
        
        # Tuning Parameters
        L2_THRESHOLD = 3 # [MODIFIED]
        L3_THRESHOLD = 3 # [MODIFIED]
        L4_THRESHOLD = 3 # [NEW]
        L5_THRESHOLD = 3 # [NEW]
        
        iter_count = 0
        # MAX_ITERS = 50000

        while target_idx < len(targets):
            iter_count += 1
            # [MODIFIED] Moved logging to end of loop for accurate "result of iteration"
            
            # Checks
            if iter_count > self.MAX_ITERS:
                print(f"WARNING: Max iterations reached. Stopping.")
                return None # Fail
            
            if deadline and time.time() > deadline:
                print(f"Timeout inside planning loop.")
                return None # Fail
            
            # [NEW] Pruning Check: If current sim time exceeds best known + margin
            if current_time > pruning_threshold:
                print(f"Pruned: Current time {current_time:.2f}s > Threshold {pruning_threshold:.2f}s")
                return None # Fail (Not Optimal)
            
            # 1. Progress Check
            if target_idx > max_progress_idx:
                max_progress_idx = target_idx
                consecutive_backtrack_count = 0 # Reset escalation
            
            # Note: We no longer increment a 'struggle_count' here.
            # We ONLY count explicit backtracks.

            # --- PLANNING LOGIC ---
            safe = False
            cmd_v, cmd_w = 0.0, 0.0
            
            if forced_wait_steps > 0:
                # [WAIT SUBSTITUTION MODE]
                # We are replaying history with WAITS instead of Moves
                cmd_v, cmd_w = 0.0, 0.0
                forced_wait_steps -= 1
                
                cmd = get_cmds_inverse(0.0, 0.0)
                next_model = DiffDriveModel(x0=model._state, vg=model.vg, vp=model.vp)
                next_model.update(cmd, self.decimal_dt)
                next_poly = next_model.get_footprint()
                next_time_idx = int(round((current_time + self.dt) / self.dt))
                safe = is_safe(next_poly, next_time_idx)
                
            else:
                # [NORMAL GREEDY MODE]
                tx, ty = targets[target_idx]
                dx = tx - model._state.x
                dy = ty - model._state.y
                dist = math.hypot(dx, dy)

                if dist < 1e-3:
                    target_idx += 1
                    # Log state (no move)
                    self.debugger.record_iteration(iter_count, current_time, target_idx, current_dist, 0.0, 0.0)
                    continue

                heading_fwd = math.atan2(dy, dx)
                heading_rev = math.atan2(-dy, -dx)
                err_fwd = (heading_fwd - model._state.psi + math.pi) % (2 * math.pi) - math.pi
                err_rev = (heading_rev - model._state.psi + math.pi) % (2 * math.pi) - math.pi

                move_dir = 1.0
                target_psi = heading_fwd
                if abs(err_rev) < abs(err_fwd):
                    move_dir = -1.0
                    target_psi = heading_rev

                d_psi = (target_psi - model._state.psi + math.pi) % (2 * math.pi) - math.pi

                if abs(d_psi) > 1e-4:
                    max_step_rot = w_max * self.dt
                    if abs(d_psi) <= max_step_rot:
                        cmd_w = d_psi / self.dt
                    else:
                        cmd_w = np.sign(d_psi) * w_max
                else:
                    max_step_dist = v_max * self.dt
                    if dist <= max_step_dist:
                        cmd_v_mag = dist / self.dt
                    else:
                        cmd_v_mag = v_max
                    cmd_v = cmd_v_mag * move_dir

                cmd = get_cmds_inverse(cmd_v, cmd_w)
                next_model = DiffDriveModel(x0=model._state, vg=model.vg, vp=model.vp)
                next_model.update(cmd, self.decimal_dt)
                next_poly = next_model.get_footprint()
                next_time_idx = int(round((current_time + self.dt) / self.dt))
                safe = is_safe(next_poly, next_time_idx)

            # --- EXECUTION ---
            if safe:
                model.update(cmd, self.decimal_dt)
                current_time += self.dt
                # Update Distance (Approximate arc length)
                # If rotating in place, dist doesn't change much, or we can use 0 for pure rotation.
                # Here we use linear velocity contribution.
                current_dist += abs(cmd_v) * self.dt
                
                s = model._state
                trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, cmd_v, cmd_w, target_idx, current_dist))

                # [VISUALIZATION] Record Waits
                is_waiting = abs(cmd_v) < 1e-6 and abs(cmd_w) < 1e-6
                if is_waiting:
                    self.debugger.record_wait(iter_count, current_time)

                # Log SUCCESS
                self.debugger.record_iteration(iter_count, current_time, target_idx, current_dist, cmd_v, cmd_w)

                # Note: We do NOT reset last_progress_iter here.
                # It is only reset if current_time > max_reached_time.
            else:
                # FAILURE HANDLING
                self.debugger.record_collision(iter_count, current_time, next_poly.centroid.x, next_poly.centroid.y)
                # Cancel forced wait if we crash while waiting
                forced_wait_steps = 0

                # 1. Try One Single Greedy Wait
                wait_success = False
                if forced_wait_steps == 0:
                    already_waiting = abs(cmd_v) < 1e-6 and abs(cmd_w) < 1e-6
                    if not already_waiting:
                        wait_model = DiffDriveModel(x0=model._state, vg=model.vg, vp=model.vp)
                        wait_model.update(DiffDriveCommands(0, 0), self.decimal_dt)
                        wait_poly = wait_model.get_footprint()
                        wait_time_idx = int(round((current_time + self.dt) / self.dt))

                        if is_safe(wait_poly, wait_time_idx):
                            model.update(DiffDriveCommands(0, 0), self.decimal_dt)
                            current_time += self.dt
                            # Wait adds 0 distance
                            s = model._state
                            trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, 0.0, 0.0, target_idx, current_dist))
                            wait_success = True
                            
                            # Log Wait Success
                            self.debugger.record_iteration(iter_count, current_time, target_idx, current_dist, 0.0, 0.0)

                if wait_success:
                    continue

                # 2. ESCALATING BACKTRACK LOGIC
                consecutive_backtrack_count += 1 # [MODIFIED] Increment ONLY when we actually backtrack
                
                # Determine "Pop Magnitude" (Moves, not Steps)
                moves_to_pop = 1
                
                # Logic: L2=3 means "After 3 backtracks, switch to Medium". 
                # So backtracks #1, #2, #3 are Small. #4 is Medium.
                # OR does user mean "3 Small, then 3 Medium"?
                # "Trigger medium ... after a fixed count of used small backtracks"
                # So:
                # Count 1, 2, 3 -> Small (Pop 1)
                # Count 4, 5, 6 -> Medium (Pop 2)
                # Count 7, 8, 9 -> Large (Pop 5)
                # Count 10, 11, 12 -> Huge (Pop 10)
                # Count 13+ -> Massive (Pop 15)
                
                if consecutive_backtrack_count > (L2_THRESHOLD + L3_THRESHOLD + L4_THRESHOLD + L5_THRESHOLD):
                    moves_to_pop = 15
                elif consecutive_backtrack_count > (L2_THRESHOLD + L3_THRESHOLD + L4_THRESHOLD):
                    moves_to_pop = 10
                elif consecutive_backtrack_count > (L2_THRESHOLD + L3_THRESHOLD):
                    moves_to_pop = 5
                elif consecutive_backtrack_count > L2_THRESHOLD:
                    moves_to_pop = 2
                
                popped_moves_count = 0
                skipped_old_waits = 0
                
                if len(trajectory) == 0:
                    # Cannot backtrack from start - just wait here
                    model.update(DiffDriveCommands(0,0), self.decimal_dt)
                    current_time += self.dt
                    # Start Wait
                    s = model._state
                    trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, 0.0, 0.0, target_idx, current_dist))
                    self.debugger.record_iteration(iter_count, current_time, target_idx, current_dist, 0.0, 0.0)
                    continue

                # Pop Loop: Remove X moves, skipping past waits
                while len(trajectory) > 0 and moves_to_pop > 0:
                    prev_pt = trajectory.pop()
                    
                    is_wait_step = (abs(prev_pt.v) < 1e-6 and abs(prev_pt.w) < 1e-6)
                    
                    if is_wait_step:
                        skipped_old_waits += 1
                    else:
                        popped_moves_count += 1
                        moves_to_pop -= 1

                # Restore
                if len(trajectory) > 0:
                    restore_pt = trajectory[-1]
                    model._state = DiffDriveState(x=restore_pt.x, y=restore_pt.y, psi=restore_pt.theta)
                    current_time = restore_pt.t
                    current_dist = restore_pt.accumulated_dist
                    target_idx = restore_pt.target_idx
                    self.debugger.record_backtrack(iter_count, prev_pt.t, current_time, popped_moves_count)
                    
                    # Log Backtrack (Time Drops, Dist Drops)
                    # We log the NEW state (restored state)
                    self.debugger.record_iteration(iter_count, current_time, target_idx, current_dist, 0.0, 0.0)
                    
                    # FORMULA: Old Waits + Rewound Time (No extra +1 needed)
                    forced_wait_steps = skipped_old_waits + popped_moves_count
                else:
                    model._state = DiffDriveState(x=initial_state_copy.x, y=initial_state_copy.y, psi=initial_state_copy.psi)
                    current_time = 0.0
                    current_dist = 0.0
                    target_idx = 0 
                    forced_wait_steps = skipped_old_waits + popped_moves_count
                    
                    self.debugger.record_iteration(iter_count, current_time, target_idx, current_dist, 0.0, 0.0)

        return trajectory
