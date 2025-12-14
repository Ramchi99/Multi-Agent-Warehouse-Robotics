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

class ExactSpaceTimePlanner:
    def __init__(self, static_obstacles: List[Polygon], dt: float = 0.1, margin: float = 0.1, use_stagnation_logic: bool = True):
        self.static_obstacles = static_obstacles
        self.dt = dt
        self.decimal_dt = Decimal(str(dt))
        self.margin = margin
        self.reservations: Dict[int, List[Polygon]] = {}
        self.debugger = PlannerDebugger()
        self.use_stagnation_logic = use_stagnation_logic

    def plan_prioritized(self, robots_sequence, initial_states, waypoints_dict, geometries, params):
        final_plans = {}
        self.reservations.clear()
        total_start = time.time()

        start_footprints = {}
        for r_name in robots_sequence:
            s = initial_states[r_name]
            vg = geometries[r_name]
            footprint_poly = vg.outline_as_polygon.buffer(self.margin)
            tform = SE2_from_xytheta([s.x, s.y, s.psi])
            start_footprints[r_name] = apply_SE2_to_shapely_geo(footprint_poly, tform)

        for i, robot_name in enumerate(robots_sequence):
            print(f"Planning physics execution for {robot_name}...")
            r_start = time.time()
            self.debugger.start_robot(robot_name)

            # [MODIFIED] Lower priority robots are GHOSTS. We do not treat them as static obstacles.
            # They must yield to us.
            temp_static_obstacles = []

            start_state = initial_states[robot_name]
            model = DiffDriveModel(x0=start_state, vg=geometries[robot_name], vp=params[robot_name])
            targets = waypoints_dict.get(robot_name, [])

            trajectory = self._plan_single_robot_backtracking(model, targets, extra_static_obstacles=temp_static_obstacles)
            final_plans[robot_name] = trajectory
            
            print(f"  -> Planned {robot_name} in {time.time() - r_start:.2f}s")

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

        print(f"Total Exact Planning Time: {time.time() - total_start:.2f}s")
        self.debugger.plot_summary(self.static_obstacles)
        return final_plans

    def _plan_single_robot_backtracking(self, model, targets, extra_static_obstacles):
        trajectory: List[PlanPoint] = []
        current_time = 0.0
        initial_state_copy = DiffDriveState(x=model._state.x, y=model._state.y, psi=model._state.psi)

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
            for obs in self.static_obstacles:
                if obs.intersects(next_poly): return False
            for obs in extra_static_obstacles:
                if obs.intersects(next_poly): return False
            if time_idx in self.reservations:
                for reserved_poly in self.reservations[time_idx]:
                    if reserved_poly.intersects(next_poly): return False
            return True

        target_idx = 0
        forced_wait_steps = 0
        
        # --- ADAPTIVE STAGNATION CONFIG ---
        max_reached_time = 0.0
        last_progress_iter = 0
        
        # User Setting: Trigger quickly (100 iterations)
        STAGNATION_THRESHOLD = 20 
        
        # Counter for how many times we have stagnated consecutively
        # (without reaching a new record time)
        stagnation_counter = 0 
        
        iter_count = 0
        MAX_ITERS = 5000

        while target_idx < len(targets):
            iter_count += 1
            if iter_count % 5 == 0:
                 self.debugger.record_iteration(iter_count, current_time, target_idx)

            if iter_count > MAX_ITERS:
                print(f"WARNING: Max iterations reached. Stopping.")
                break

            # 1. Update Global Progress (The "High Score")
            if current_time > max_reached_time:
                max_reached_time = current_time
                last_progress_iter = iter_count
                
                # We broke through the deadlock! Reset the severity level.
                if stagnation_counter > 0:
                    # print(f"  [Recovered] Broke through barrier after {stagnation_counter} jumps.")
                    stagnation_counter = 0
            
            # 2. Check for Stagnation
            is_stagnant = False
            if self.use_stagnation_logic:
                is_stagnant = (iter_count - last_progress_iter) > STAGNATION_THRESHOLD

            # --- PLANNING LOGIC ---
            safe = False
            cmd_v, cmd_w = 0.0, 0.0
            
            if is_stagnant:
                # Force failure to trigger the Jump Logic below
                safe = False
            
            elif forced_wait_steps > 0:
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
                    if abs(d_psi) <= max_step_rot: cmd_w = d_psi / self.dt
                    else: cmd_w = np.sign(d_psi) * w_max
                else:
                    max_step_dist = v_max * self.dt
                    if dist <= max_step_dist: cmd_v_mag = dist / self.dt
                    else: cmd_v_mag = v_max
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
                s = model._state
                trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, cmd_v, cmd_w, target_idx))
                
                # [VISUALIZATION] Record Waits
                if iter_count % 5 == 0:
                    is_waiting = (abs(cmd_v) < 1e-6 and abs(cmd_w) < 1e-6)
                    if is_waiting:
                        self.debugger.record_wait(iter_count, current_time)
                
                # Note: We do NOT reset last_progress_iter here. 
                # It is only reset if current_time > max_reached_time.
            else:
                # FAILURE HANDLING
                if not is_stagnant:
                    self.debugger.record_collision(next_poly.centroid.x, next_poly.centroid.y)
                    # Cancel forced wait if we crash while waiting
                    forced_wait_steps = 0 

                # 1. Try One Single Greedy Wait (Only if NOT Stagnant)
                wait_success = False
                if not is_stagnant and forced_wait_steps == 0:
                    already_waiting = (abs(cmd_v) < 1e-6 and abs(cmd_w) < 1e-6)
                    if not already_waiting:
                        wait_model = DiffDriveModel(x0=model._state, vg=model.vg, vp=model.vp)
                        wait_model.update(DiffDriveCommands(0,0), self.decimal_dt)
                        wait_poly = wait_model.get_footprint()
                        wait_time_idx = int(round((current_time + self.dt) / self.dt))
                        
                        if is_safe(wait_poly, wait_time_idx):
                            model.update(DiffDriveCommands(0,0), self.decimal_dt)
                            current_time += self.dt
                            s = model._state
                            trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, 0.0, 0.0, target_idx))
                            wait_success = True
                
                if wait_success:
                    continue
                
                # 2. BACKTRACK LOGIC
                
                # Determine Jump Size based on Escalation
                pop_count = 1
                
                if is_stagnant:
                    stagnation_counter += 1
                    
                    # [ESCALATION LOGIC]
                    if stagnation_counter <= 2:
                        jump_steps = 5   # Level 1: 0.5s (Try twice)
                    elif stagnation_counter <= 4:
                        jump_steps = 10  # Level 2: 1.0s (Try twice)
                    else:
                        jump_steps = 20  # Level 3: 2.0s (Panic mode)
                    
                    print(f"  [Stagnation L{stagnation_counter}] Jump {jump_steps} steps & Wait.")
                    self.debugger.record_stagnation(iter_count, current_time)
                    
                    pop_count = jump_steps
                    forced_wait_steps = jump_steps # "Wait Substitution"
                    
                    # Reset watchdog
                    last_progress_iter = iter_count
                    # We do NOT reset max_reached_time, because we want to know if this jump
                    # actually helped us beat the previous record.

                # Check if we can actually pop that many
                if len(trajectory) < pop_count:
                    pop_count = len(trajectory) # Pop everything if needed

                # Pop Loop
                if len(trajectory) == 0:
                    # Cannot backtrack from start
                    model.update(DiffDriveCommands(0,0), self.decimal_dt)
                    current_time += self.dt
                    s = model._state
                    trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, 0.0, 0.0, target_idx))
                    continue

                skipped_wait_steps = 0
                while len(trajectory) > 0 and pop_count > 0:
                    prev_pt = trajectory.pop()
                    pop_count -= 1
                    
                    was_wait = (abs(prev_pt.v) < 1e-6 and abs(prev_pt.w) < 1e-6)
                    if was_wait:
                        skipped_wait_steps += 1
                        pop_count += 1 
                    else:
                        if pop_count == 0:
                            break

                # Restore
                if len(trajectory) > 0:
                    restore_pt = trajectory[-1]
                    model._state = DiffDriveState(x=restore_pt.x, y=restore_pt.y, psi=restore_pt.theta)
                    current_time = restore_pt.t
                    target_idx = restore_pt.target_idx
                    self.debugger.record_backtrack(iter_count, prev_pt.t, current_time)
                    forced_wait_steps = max(forced_wait_steps, skipped_wait_steps + 1)
                else:
                    model._state = DiffDriveState(x=initial_state_copy.x, y=initial_state_copy.y, psi=initial_state_copy.psi)
                    current_time = 0.0
                    target_idx = 0 
                    forced_wait_steps = max(forced_wait_steps, skipped_wait_steps + 1)

        return trajectory
