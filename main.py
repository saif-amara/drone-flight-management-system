"""
Drone Autopilot Simulation
==========================
6-DOF Quadrotor with Cascade PID Autopilot + Trajectory Following
Author: Generated for ISSAT Sousse — Systèmes Embarqués
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

from drone_model     import DroneModel
from pid_controller  import DroneAutopilot
from trajectory      import TrajectoryGenerator, mission_square, mission_helix, mission_figure8
from visualizer      import DroneVisualizer


# ── Simulation parameters ─────────────────────────────────────────────────────
DT          = 0.01   # time step (s)
T_MAX       = 60.0   # max simulation time (s)
LOG_EVERY   = 5      # log every N steps (reduces memory)

MISSIONS = {
    'square':   mission_square(altitude=4.0, side=5.0),
    'helix':    mission_helix(altitude_start=2.0, altitude_end=8.0, radius=3.0, turns=2),
    'figure8':  mission_figure8(altitude=4.0, scale=3.5),
}


def run_simulation(mission_name='square'):
    print(f"\n{'='*50}")
    print(f"  Drone Autopilot Simulation — Mission: {mission_name.upper()}")
    print(f"{'='*50}\n")

    waypoints = MISSIONS[mission_name]

    # Init
    drone      = DroneModel()
    autopilot  = DroneAutopilot()
    trajectory = TrajectoryGenerator(waypoints, cruise_speed=1.5, acceptance_radius=0.4)

    drone.reset()
    autopilot.reset()
    trajectory.reset()

    log = []
    t   = 0.0
    step = 0

    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Duration:  up to {T_MAX}s  |  dt = {DT}s\n")
    print("  Simulating", end='', flush=True)

    while t < T_MAX:
        # Get target from trajectory
        target_wp = trajectory.get_target(drone.position)
        target    = [target_wp[0], target_wp[1], target_wp[2], target_wp[3]]

        # Compute control
        forces = autopilot.compute(drone.state, target, DT)

        # Step dynamics
        state = drone.step(forces, DT)

        # Log
        if step % LOG_EVERY == 0:
            log.append({
                't':        t,
                'state':    state.copy(),
                'target':   target,
                'forces':   forces.copy(),
                'progress': trajectory.progress(),
            })

        t    += DT
        step += 1

        if step % 500 == 0:
            print('.', end='', flush=True)

        if trajectory.completed:
            print(f'\n\n  ✓ Mission complete at t = {t:.1f}s')
            break
    else:
        print(f'\n\n  ✗ Mission timeout at t = {T_MAX}s')

    print(f'  Total log frames: {len(log)}\n')

    # Final position
    final = log[-1]['state']
    print(f"  Final position : x={final[0]:.2f}  y={final[1]:.2f}  z={final[2]:.2f}")
    print(f"  Final angles   : φ={np.degrees(final[3]):.1f}°  "
          f"θ={np.degrees(final[4]):.1f}°  ψ={np.degrees(final[5]):.1f}°\n")

    return log, waypoints


def main():
    parser = argparse.ArgumentParser(description='Drone Autopilot Simulation')
    parser.add_argument('--mission', choices=['square', 'helix', 'figure8'],
                        default='square', help='Mission type (default: square)')
    parser.add_argument('--save', action='store_true',
                        help='Save animation as GIF')
    args = parser.parse_args()

    log, waypoints = run_simulation(args.mission)

    print("  Launching 3D animation...")
    vis  = DroneVisualizer(log, np.array(waypoints)[:, :3])
    anim = vis.animate(interval=25)

    if args.save:
        fname = f'drone_{args.mission}.gif'
        print(f"  Saving animation to {fname} ...")
        anim.save(fname, writer='pillow', fps=20)
        print(f"  Saved: {fname}")

    plt.show()


if __name__ == '__main__':
    main()
