"""
Flight Management System (FMS)
================================
Features:
  - Battery monitor (consumption model)
  - Return to Home (RTH)
  - Geofence enforcement
  - Waypoint editor (add/remove/insert)
"""

import numpy as np


# ── Battery Model ─────────────────────────────────────────────────────────────

class BatteryMonitor:
    """
    Simple energy consumption model.
    Consumption depends on thrust force and time.
    """

    def __init__(self, capacity_wh=50.0, voltage=11.1):
        self.capacity_wh   = capacity_wh       # total energy (Wh)
        self.voltage       = voltage            # battery voltage (V)
        self.energy_used   = 0.0               # Wh consumed
        self.hover_power   = 80.0              # Watts at hover thrust
        self.mass          = 1.0
        self.g             = 9.81

    def reset(self):
        self.energy_used = 0.0

    def update(self, thrust, dt):
        """Update energy consumption based on thrust."""
        hover_thrust = self.mass * self.g
        power = self.hover_power * (thrust / (hover_thrust + 1e-6)) ** 1.5
        self.energy_used += power * dt / 3600.0   # convert W*s to Wh

    @property
    def level(self):
        """Battery level 0.0 → 1.0"""
        return max(0.0, 1.0 - self.energy_used / self.capacity_wh)

    @property
    def percent(self):
        return self.level * 100.0

    @property
    def is_critical(self):
        return self.level < 0.15

    @property
    def is_low(self):
        return self.level < 0.30

    @property
    def is_dead(self):
        return self.level <= 0.0

    def status(self):
        if self.is_dead:     return "DEAD"
        if self.is_critical: return "CRITICAL"
        if self.is_low:      return "LOW"
        return "OK"


# ── Geofence ──────────────────────────────────────────────────────────────────

class Geofence:
    """
    Cylindrical geofence: max radius from origin + altitude limits.
    If drone exits the fence → velocity is clamped back inside.
    """

    def __init__(self, max_radius=15.0, min_alt=0.0, max_alt=20.0):
        self.max_radius = max_radius
        self.min_alt    = min_alt
        self.max_alt    = max_alt
        self.violated   = False

    def check(self, position):
        """Returns True if position is INSIDE the fence."""
        x, y, z = position
        r = np.sqrt(x**2 + y**2)
        inside = (r <= self.max_radius) and (self.min_alt <= z <= self.max_alt)
        self.violated = not inside
        return inside

    def clamp_target(self, target):
        """Clamp a waypoint target to stay inside the fence."""
        x, y, z, yaw = target
        r = np.sqrt(x**2 + y**2)
        if r > self.max_radius:
            scale = self.max_radius / (r + 1e-6)
            x *= scale
            y *= scale
        z = np.clip(z, self.min_alt, self.max_alt)
        return [x, y, z, yaw]

    def boundary_points(self, n=64):
        """Return XY circle points for visualization."""
        angles = np.linspace(0, 2 * np.pi, n)
        xs = self.max_radius * np.cos(angles)
        ys = self.max_radius * np.sin(angles)
        return xs, ys


# ── Waypoint Editor ───────────────────────────────────────────────────────────

class WaypointEditor:
    """
    Interactive waypoint editor.
    Each waypoint: [x, y, z, yaw]
    """

    def __init__(self, waypoints=None):
        self.waypoints = []
        if waypoints:
            for wp in waypoints:
                self.add(wp)

    def add(self, wp, index=None):
        """Append or insert a waypoint."""
        if len(wp) == 3:
            wp = list(wp) + [0.0]
        wp = list(wp[:4])
        if index is None:
            self.waypoints.append(wp)
        else:
            self.waypoints.insert(index, wp)

    def remove(self, index):
        if 0 <= index < len(self.waypoints):
            self.waypoints.pop(index)

    def update(self, index, wp):
        if 0 <= index < len(self.waypoints):
            if len(wp) == 3:
                wp = list(wp) + [0.0]
            self.waypoints[index] = list(wp[:4])

    def clear(self):
        self.waypoints = []

    def total_distance(self):
        if len(self.waypoints) < 2:
            return 0.0
        pts = np.array([wp[:3] for wp in self.waypoints])
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def print_plan(self):
        print("\n  Flight Plan:")
        print(f"  {'#':<4} {'X':>7} {'Y':>7} {'Z':>7} {'Yaw':>7}")
        print("  " + "-" * 36)
        for i, wp in enumerate(self.waypoints):
            print(f"  {i:<4} {wp[0]:>7.2f} {wp[1]:>7.2f} {wp[2]:>7.2f} {wp[3]:>7.1f}")
        print(f"\n  Total distance: {self.total_distance():.1f} m")
        print(f"  Waypoints     : {len(self.waypoints)}\n")

    def get(self):
        return [wp[:] for wp in self.waypoints]


# ── Return to Home ────────────────────────────────────────────────────────────

class ReturnToHome:
    """
    RTH sequence:
      1. Climb to safe altitude
      2. Fly to home XY
      3. Descend and land
    """

    RTH_IDLE     = "IDLE"
    RTH_CLIMB    = "CLIMB"
    RTH_TRANSIT  = "TRANSIT"
    RTH_DESCEND  = "DESCEND"
    RTH_LANDED   = "LANDED"

    def __init__(self, safe_altitude=6.0, land_speed=0.5):
        self.home          = [0.0, 0.0, 0.0]
        self.safe_altitude = safe_altitude
        self.land_speed    = land_speed
        self.state         = self.RTH_IDLE
        self.active        = False

    def set_home(self, position):
        self.home = list(position[:3])

    def activate(self):
        if not self.active:
            self.active = True
            self.state  = self.RTH_CLIMB
            print("  [RTH] Activated — returning to home")

    def deactivate(self):
        self.active = False
        self.state  = self.RTH_IDLE

    def get_target(self, position):
        """
        Returns [x, y, z, yaw] target based on RTH state.
        Call this every step when RTH is active.
        """
        x, y, z = position

        if self.state == self.RTH_CLIMB:
            target = [x, y, self.safe_altitude, 0.0]
            if abs(z - self.safe_altitude) < 0.4:
                self.state = self.RTH_TRANSIT
                print("  [RTH] Altitude reached — transiting home")

        elif self.state == self.RTH_TRANSIT:
            target = [self.home[0], self.home[1], self.safe_altitude, 0.0]
            dx = abs(x - self.home[0])
            dy = abs(y - self.home[1])
            if dx < 0.4 and dy < 0.4:
                self.state = self.RTH_DESCEND
                print("  [RTH] Over home — descending")

        elif self.state == self.RTH_DESCEND:
            target = [self.home[0], self.home[1], 0.3, 0.0]
            if z < 0.5:
                self.state  = self.RTH_LANDED
                self.active = False
                print("  [RTH] Landed successfully")

        elif self.state == self.RTH_LANDED:
            target = [self.home[0], self.home[1], 0.0, 0.0]

        else:
            target = list(position) + [0.0]

        return target

    @property
    def is_landed(self):
        return self.state == self.RTH_LANDED


# ── FMS (combines everything) ─────────────────────────────────────────────────

class FlightManagementSystem:
    """
    Top-level FMS that integrates:
      - WaypointEditor
      - BatteryMonitor
      - Geofence
      - ReturnToHome
    """

    def __init__(self, waypoints=None,
                 battery_capacity=50.0,
                 geofence_radius=15.0,
                 geofence_max_alt=20.0,
                 rth_altitude=6.0):

        self.editor   = WaypointEditor(waypoints)
        self.battery  = BatteryMonitor(capacity_wh=battery_capacity)
        self.geofence = Geofence(max_radius=geofence_radius,
                                 max_alt=geofence_max_alt)
        self.rth      = ReturnToHome(safe_altitude=rth_altitude)

        self._rth_triggered = False

    def reset(self, home_position=None):
        self.battery.reset()
        self.rth.deactivate()
        self._rth_triggered = False
        if home_position is not None:
            self.rth.set_home(home_position)

    def update(self, position, thrust, dt):
        """Call every simulation step."""
        self.battery.update(thrust, dt)

        if not self._rth_triggered:
            if self.battery.is_critical:
                print(f"  [FMS] Battery critical ({self.battery.percent:.0f}%) — RTH triggered!")
                self.rth.activate()
                self._rth_triggered = True

        if not self.geofence.check(position):
            if not self._rth_triggered:
                print(f"  [FMS] Geofence violated at {position} — RTH triggered!")
                self.rth.activate()
                self._rth_triggered = True

    def get_target(self, trajectory, position):
        """
        Returns the active target for the autopilot.
        RTH overrides normal trajectory when active.
        """
        if self.rth.active:
            raw = self.rth.get_target(position)
        else:
            raw = trajectory.get_target(position)

        return self.geofence.clamp_target(raw)

    def status(self):
        return {
            'battery_pct': self.battery.percent,
            'battery_status': self.battery.status(),
            'geofence_ok': not self.geofence.violated,
            'rth_active': self.rth.active,
            'rth_state': self.rth.state,
        }
