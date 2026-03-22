import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, output_min=-np.inf, output_max=np.inf):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max

        self._integral = 0.0
        self._prev_error = 0.0
        self._first_step = True

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._first_step = True

    def compute(self, setpoint, measurement, dt):
        error = setpoint - measurement

        # Proportional
        P = self.kp * error

        # Integral (with anti-windup)
        self._integral += error * dt
        I = self.ki * self._integral

        # Derivative (avoid derivative kick on first step)
        if self._first_step:
            D = 0.0
            self._first_step = False
        else:
            D = self.kd * (error - self._prev_error) / (dt + 1e-9)

        self._prev_error = error

        output = P + I + D
        return float(np.clip(output, self.output_min, self.output_max))


class DroneAutopilot:
    """
    Cascade PID Autopilot:
    - Outer loop: position → desired angles
    - Inner loop: angles → torques
    - Altitude loop: z → thrust
    """

    def __init__(self):
        # --- Altitude (Z) ---
        self.pid_z   = PIDController(kp=8.0,  ki=0.5, kd=5.0,
                                     output_min=0.0, output_max=30.0)

        # --- Position XY → desired angles ---
        self.pid_x   = PIDController(kp=0.5,  ki=0.01, kd=0.4,
                                     output_min=-0.3, output_max=0.3)
        self.pid_y   = PIDController(kp=0.5,  ki=0.01, kd=0.4,
                                     output_min=-0.3, output_max=0.3)

        # --- Attitude (angles) ---
        self.pid_phi   = PIDController(kp=6.0, ki=0.1, kd=3.0,
                                       output_min=-5.0, output_max=5.0)
        self.pid_theta = PIDController(kp=6.0, ki=0.1, kd=3.0,
                                       output_min=-5.0, output_max=5.0)
        self.pid_psi   = PIDController(kp=4.0, ki=0.05, kd=2.0,
                                       output_min=-3.0, output_max=3.0)

        self.m  = 1.0
        self.g  = 9.81

    def reset(self):
        for pid in [self.pid_z, self.pid_x, self.pid_y,
                    self.pid_phi, self.pid_theta, self.pid_psi]:
            pid.reset()

    def compute(self, state, target, dt):
        """
        state  : drone state [x,y,z, phi,theta,psi, vx,vy,vz, p,q,r]
        target : [x_d, y_d, z_d, psi_d]
        returns: forces [F, tau_phi, tau_theta, tau_psi]
        """
        x, y, z = state[0], state[1], state[2]
        phi, theta, psi = state[3], state[4], state[5]

        x_d, y_d, z_d, psi_d = target

        # Thrust from altitude controller
        F = self.pid_z.compute(z_d, z, dt) + self.m * self.g

        # Desired angles from position error (rotated to body frame)
        dx_error = self.pid_x.compute(x_d, x, dt)
        dy_error = self.pid_y.compute(y_d, y, dt)

        # Rotate position error to yaw frame
        phi_d   =  dx_error * np.sin(psi) - dy_error * np.cos(psi)
        theta_d =  dx_error * np.cos(psi) + dy_error * np.sin(psi)

        # Attitude torques
        tau_phi   = self.pid_phi.compute(phi_d,   phi,   dt)
        tau_theta = self.pid_theta.compute(theta_d, theta, dt)
        tau_psi   = self.pid_psi.compute(psi_d,   psi,   dt)

        return np.array([F, tau_phi, tau_theta, tau_psi])
