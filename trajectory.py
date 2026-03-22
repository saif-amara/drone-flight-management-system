import numpy as np

class TrajectoryGenerator:
    """
    Waypoint-based trajectory generator.
    Drone follows a list of 3D waypoints with a given cruise speed.
    """

    def __init__(self, waypoints, cruise_speed=1.5, acceptance_radius=0.3):
        """
        waypoints      : list of [x, y, z] or [x, y, z, yaw]
        cruise_speed   : m/s
        acceptance_radius : distance to switch to next waypoint
        """
        self.waypoints = []
        for wp in waypoints:
            if len(wp) == 3:
                self.waypoints.append([wp[0], wp[1], wp[2], 0.0])
            else:
                self.waypoints.append(list(wp))

        self.cruise_speed = cruise_speed
        self.acceptance_radius = acceptance_radius
        self.current_idx = 0
        self.completed = False

    def reset(self):
        self.current_idx = 0
        self.completed = False

    def get_target(self, position):
        """Returns current target [x, y, z, yaw]"""
        if self.completed:
            return self.waypoints[-1]

        target = self.waypoints[self.current_idx]
        dist = np.linalg.norm(np.array(position) - np.array(target[:3]))

        if dist < self.acceptance_radius:
            if self.current_idx < len(self.waypoints) - 1:
                self.current_idx += 1
            else:
                self.completed = True

        return self.waypoints[self.current_idx]

    def progress(self):
        return self.current_idx / max(len(self.waypoints) - 1, 1)


# ── Predefined missions ────────────────────────────────────────────────────────

def mission_square(altitude=3.0, side=4.0):
    """Takeoff → square pattern → land"""
    return [
        [0.0,    0.0,    0.3],          # takeoff
        [0.0,    0.0,    altitude],     # climb
        [side,   0.0,    altitude],
        [side,   side,   altitude],
        [0.0,    side,   altitude],
        [0.0,    0.0,    altitude],     # return
        [0.0,    0.0,    0.3],          # land
    ]

def mission_helix(altitude_start=2.0, altitude_end=8.0, radius=3.0, turns=2, n_points=40):
    """Helical ascent"""
    wps = [[0.0, 0.0, 0.5]]
    wps.append([0.0, 0.0, altitude_start])
    for i in range(n_points):
        angle = turns * 2 * np.pi * i / n_points
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = altitude_start + (altitude_end - altitude_start) * i / n_points
        wps.append([x, y, z])
    wps.append([0.0, 0.0, altitude_end])
    wps.append([0.0, 0.0, 0.5])
    return wps

def mission_figure8(altitude=4.0, scale=3.0, n_points=50):
    """Figure-8 pattern"""
    wps = [[0.0, 0.0, 0.5], [0.0, 0.0, altitude]]
    for i in range(n_points):
        t = 2 * np.pi * i / n_points
        x = scale * np.sin(t)
        y = scale * np.sin(t) * np.cos(t)
        wps.append([x, y, altitude])
    wps.append([0.0, 0.0, altitude])
    wps.append([0.0, 0.0, 0.5])
    return wps
