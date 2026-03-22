import numpy as np

class DroneModel:
    """
    6-DOF Quadrotor Dynamics Model
    State: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
    """

    def __init__(self):
        # Physical parameters
        self.m = 1.0          # mass (kg)
        self.g = 9.81         # gravity (m/s^2)
        self.Ixx = 0.0196     # moment of inertia x (kg.m^2)
        self.Iyy = 0.0196     # moment of inertia y (kg.m^2)
        self.Izz = 0.0264     # moment of inertia z (kg.m^2)
        self.l = 0.23         # arm length (m)
        self.k = 2.98e-6      # thrust coefficient
        self.b = 1.14e-7      # drag coefficient
        self.kd = 0.25        # linear drag

        # State vector [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
        self.state = np.zeros(12)

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=float)
        else:
            self.state = np.zeros(12)

    def derivatives(self, state, forces):
        """
        Compute state derivatives given current state and control forces.
        forces = [F_total, tau_phi, tau_theta, tau_psi]
        """
        x, y, z, phi, theta, psi, vx, vy, vz, p, q, r = state
        F, tau_phi, tau_theta, tau_psi = forces

        # Rotation matrix (body to world)
        cphi   = np.cos(phi);   sphi   = np.sin(phi)
        ctheta = np.cos(theta); stheta = np.sin(theta)
        cpsi   = np.cos(psi);   spsi   = np.sin(psi)

        # Linear accelerations (world frame)
        ax = (F / self.m) * (cpsi * stheta * cphi + spsi * sphi) - self.kd * vx / self.m
        ay = (F / self.m) * (spsi * stheta * cphi - cpsi * sphi) - self.kd * vy / self.m
        az = (F / self.m) * (ctheta * cphi) - self.g - self.kd * vz / self.m

        # Angular accelerations (body frame)
        dp = (tau_phi  + (self.Iyy - self.Izz) * q * r) / self.Ixx
        dq = (tau_theta + (self.Izz - self.Ixx) * p * r) / self.Iyy
        dr = (tau_psi   + (self.Ixx - self.Iyy) * p * q) / self.Izz

        # Euler angle rates
        dphi   = p + (q * sphi + r * cphi) * np.tan(theta)
        dtheta = q * cphi - r * sphi
        dpsi   = (q * sphi + r * cphi) / (ctheta + 1e-6)

        return np.array([vx, vy, vz, dphi, dtheta, dpsi, ax, ay, az, dp, dq, dr])

    def step(self, forces, dt):
        """RK4 integration step"""
        k1 = self.derivatives(self.state, forces)
        k2 = self.derivatives(self.state + 0.5 * dt * k1, forces)
        k3 = self.derivatives(self.state + 0.5 * dt * k2, forces)
        k4 = self.derivatives(self.state + dt * k3, forces)

        self.state = self.state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Keep angles in [-pi, pi]
        self.state[3:6] = np.arctan2(np.sin(self.state[3:6]), np.cos(self.state[3:6]))
        return self.state.copy()

    @property
    def position(self):
        return self.state[0:3]

    @property
    def angles(self):
        return self.state[3:6]  # phi, theta, psi

    @property
    def velocity(self):
        return self.state[6:9]

    @property
    def angular_velocity(self):
        return self.state[9:12]
