"""
Obstacle Avoidance
==================
Simple potential field method:
  - Obstacles repel the drone
  - Target attracts the drone
  - Combined force modifies the waypoint target
"""

import numpy as np


class Obstacle:
    def __init__(self, x, y, z, radius):
        self.position = np.array([x, y, z])
        self.radius   = radius


class ObstacleAvoidance:
    """
    Artificial Potential Field obstacle avoidance.
    Modifies the target position to steer away from obstacles.
    """

    def __init__(self, influence_radius=2.5, repulsion_gain=3.0):
        self.obstacles        = []
        self.influence_radius = influence_radius   # metres around obstacle
        self.repulsion_gain   = repulsion_gain
        self.active           = True

    def add_obstacle(self, x, y, z, radius=0.5):
        self.obstacles.append(Obstacle(x, y, z, radius))

    def clear(self):
        self.obstacles = []

    def compute_repulsion(self, position):
        """
        Returns a 3D repulsion vector pushing away from all nearby obstacles.
        """
        pos = np.array(position[:3])
        total_repulsion = np.zeros(3)

        for obs in self.obstacles:
            diff = pos - obs.position
            dist = np.linalg.norm(diff) - obs.radius  # distance to surface
            dist = max(dist, 0.05)                     # avoid division by zero

            if dist < self.influence_radius:
                # Repulsion magnitude: stronger when closer
                magnitude = self.repulsion_gain * (
                    1.0 / dist - 1.0 / self.influence_radius
                ) / (dist ** 2)
                direction = diff / (np.linalg.norm(diff) + 1e-6)
                total_repulsion += magnitude * direction

        return total_repulsion

    def modify_target(self, position, target):
        """
        Returns a modified target that avoids obstacles.
        The target is shifted by the repulsion vector.
        """
        if not self.active or not self.obstacles:
            return target

        repulsion = self.compute_repulsion(position)
        t = np.array(target[:3])
        modified = t + repulsion * 0.4   # scale factor

        return [modified[0], modified[1], modified[2], target[3]]

    def is_path_clear(self, position, target, steps=10):
        """
        Check if straight line from position to target is obstacle-free.
        Returns False if any obstacle is too close along the path.
        """
        p = np.array(position[:3])
        t = np.array(target[:3])

        for i in range(steps + 1):
            point = p + (t - p) * i / steps
            for obs in self.obstacles:
                dist = np.linalg.norm(point - obs.position) - obs.radius
                if dist < 0.3:
                    return False
        return True

    def get_obstacles_for_viz(self):
        """Return obstacle data for visualization."""
        return [(obs.position, obs.radius) for obs in self.obstacles]


# ── Predefined obstacle scenarios ─────────────────────────────────────────────

def scenario_urban(avoidance):
    """Buildings / urban environment."""
    avoidance.add_obstacle(3.0,  1.0, 2.0, radius=0.6)
    avoidance.add_obstacle(-2.0, 3.0, 1.5, radius=0.5)
    avoidance.add_obstacle(4.0, -2.0, 3.0, radius=0.7)
    avoidance.add_obstacle(1.0,  5.0, 2.0, radius=0.4)

def scenario_forest(avoidance):
    """Trees."""
    for x, y in [(2,2), (-2,3), (3,-1), (-1,-3), (4,4), (-3,-2)]:
        avoidance.add_obstacle(x, y, 2.0, radius=0.4)

def scenario_single(avoidance):
    """One obstacle in the middle of the path."""
    avoidance.add_obstacle(2.5, 2.5, 4.0, radius=0.8)
