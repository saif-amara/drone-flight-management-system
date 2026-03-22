"""
Drone Autopilot Simulation — FMS Edition
==========================================
Features:
  - 6-DOF quadrotor dynamics (RK4)
  - Cascade PID autopilot
  - Flight Management System (FMS):
      * Waypoint editor
      * Battery monitor + RTH on critical battery
      * Geofence enforcement
      * Obstacle avoidance (potential field)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

from drone_model    import DroneModel
from pid_controller import DroneAutopilot
from trajectory     import TrajectoryGenerator, mission_square, mission_helix, mission_figure8
from visualizer     import DroneVisualizer
from fms            import FlightManagementSystem
from obstacles      import ObstacleAvoidance, scenario_urban, scenario_forest, scenario_single


DT        = 0.01
T_MAX     = 90.0
LOG_EVERY = 5

MISSIONS = {
    'square':  mission_square(altitude=4.0, side=5.0),
    'helix':   mission_helix(altitude_start=2.0, altitude_end=8.0, radius=3.0, turns=2),
    'figure8': mission_figure8(altitude=4.0, scale=3.5),
}

OBSTACLE_SCENARIOS = {
    'urban':  scenario_urban,
    'forest': scenario_forest,
    'single': scenario_single,
    'none':   None,
}


def run_simulation(mission_name='square', obstacle_scenario='none',
                   battery_capacity=50.0, geofence_radius=15.0,
                   trigger_rth=False):

    print(f"\n{'='*55}")
    print(f"  Drone FMS Simulation")
    print(f"  Mission  : {mission_name.upper()}")
    print(f"  Obstacles: {obstacle_scenario}")
    print(f"  Battery  : {battery_capacity} Wh")
    print(f"  Geofence : {geofence_radius} m radius")
    print(f"{'='*55}\n")

    waypoints = MISSIONS[mission_name]

    drone     = DroneModel()
    autopilot = DroneAutopilot()
    traj      = TrajectoryGenerator(waypoints, cruise_speed=1.5, acceptance_radius=0.4)

    fms = FlightManagementSystem(
        waypoints        = waypoints,
        battery_capacity = battery_capacity,
        geofence_radius  = geofence_radius,
        geofence_max_alt = 18.0,
        rth_altitude     = 6.0,
    )

    avoidance = ObstacleAvoidance(influence_radius=2.5, repulsion_gain=3.0)
    if obstacle_scenario != 'none' and OBSTACLE_SCENARIOS[obstacle_scenario]:
        OBSTACLE_SCENARIOS[obstacle_scenario](avoidance)
        print(f"  Obstacles loaded: {len(avoidance.obstacles)}\n")

    drone.reset()
    autopilot.reset()
    traj.reset()
    fms.reset(home_position=[0.0, 0.0, 0.0])

    fms.editor.print_plan()

    log  = []
    t    = 0.0
    step = 0
    rth_demo_triggered = False

    print("  Simulating", end='', flush=True)

    while t < T_MAX:
        pos = drone.position

        if trigger_rth and t > 20.0 and not rth_demo_triggered:
            print(f"\n  [DEMO] Manual RTH triggered at t={t:.1f}s")
            fms.rth.activate()
            rth_demo_triggered = True

        fms.update(pos, forces[0] if 'forces' in dir() else 9.81, DT)

        target  = fms.get_target(traj, pos)
        target  = avoidance.modify_target(pos, target)
        forces  = autopilot.compute(drone.state, target, DT)
        state   = drone.step(forces, DT)

        if step % LOG_EVERY == 0:
            status = fms.status()
            log.append({
                't':           t,
                'state':       state.copy(),
                'target':      target,
                'forces':      forces.copy(),
                'progress':    traj.progress(),
                'battery':     status['battery_pct'],
                'rth_active':  status['rth_active'],
                'geofence_ok': status['geofence_ok'],
            })

        t    += DT
        step += 1

        if step % 500 == 0:
            print('.', end='', flush=True)

        if fms.rth.is_landed:
            print(f'\n\n  [FMS] RTH complete — landed at t={t:.1f}s')
            break
        if traj.completed and not fms.rth.active:
            print(f'\n\n  Mission complete at t={t:.1f}s')
            break
    else:
        print(f'\n\n  Timeout at t={T_MAX}s')

    s = fms.status()
    print(f"\n  Battery remaining : {s['battery_pct']:.1f}%  [{s['battery_status']}]")
    print(f"  Geofence OK       : {s['geofence_ok']}")
    print(f"  RTH state         : {s['rth_state']}")
    final = log[-1]['state']
    print(f"  Final position    : x={final[0]:.2f}  y={final[1]:.2f}  z={final[2]:.2f}\n")

    return log, waypoints, avoidance


def main():
    parser = argparse.ArgumentParser(description='Drone FMS Simulation')
    parser.add_argument('--mission',   choices=['square','helix','figure8'], default='square')
    parser.add_argument('--obstacles', choices=['none','single','urban','forest'], default='none')
    parser.add_argument('--battery',   type=float, default=50.0)
    parser.add_argument('--geofence',  type=float, default=15.0)
    parser.add_argument('--rth',       action='store_true')
    parser.add_argument('--save',      action='store_true')
    args = parser.parse_args()

    log, waypoints, avoidance = run_simulation(
        mission_name      = args.mission,
        obstacle_scenario = args.obstacles,
        battery_capacity  = args.battery,
        geofence_radius   = args.geofence,
        trigger_rth       = args.rth,
    )

    print("  Launching 3D animation...")
    vis  = DroneVisualizer(log, np.array(waypoints)[:, :3], avoidance=avoidance)
    anim = vis.animate(interval=25)

    if args.save:
        fname = f'drone_{args.mission}.gif'
        anim.save(fname, writer='pillow', fps=20)
        print(f"  Saved: {fname}")

    plt.show()


if __name__ == '__main__':
    main()
