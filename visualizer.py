import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches


class DroneVisualizer:
    def __init__(self, trajectory_data, waypoints, arm_length=0.23):
        self.data = trajectory_data   # list of state dicts
        self.waypoints = np.array(waypoints)
        self.arm_length = arm_length
        self.fig = plt.figure(figsize=(14, 9), facecolor='#0d0d1a')
        self._setup_axes()

    def _setup_axes(self):
        # 3D main view
        self.ax3d = self.fig.add_axes([0.0, 0.25, 0.65, 0.75],
                                       projection='3d', facecolor='#0d0d1a')
        # Side plots
        self.ax_alt  = self.fig.add_axes([0.67, 0.68, 0.31, 0.25], facecolor='#111122')
        self.ax_att  = self.fig.add_axes([0.67, 0.38, 0.31, 0.25], facecolor='#111122')
        self.ax_vel  = self.fig.add_axes([0.67, 0.08, 0.31, 0.25], facecolor='#111122')
        # Progress bar area
        self.ax_prog = self.fig.add_axes([0.05, 0.08, 0.58, 0.12], facecolor='#111122')

        for ax in [self.ax_alt, self.ax_att, self.ax_vel, self.ax_prog]:
            ax.tick_params(colors='#aaaacc', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333366')

        self.fig.patch.set_facecolor('#0d0d1a')

    def _draw_drone_body(self, ax, pos, angles, color='#00ffcc'):
        x, y, z = pos
        phi, theta, psi = angles
        L = self.arm_length

        # Rotation matrix
        R = self._rotation_matrix(phi, theta, psi)

        # Arm endpoints (X shape)
        arms = [
            np.array([ L,  0, 0]),
            np.array([-L,  0, 0]),
            np.array([ 0,  L, 0]),
            np.array([ 0, -L, 0]),
        ]
        arm_colors = ['#ff4444', '#ff4444', '#4444ff', '#4444ff']
        motor_colors = ['#ffaa00', '#ffaa00', '#00aaff', '#00aaff']

        artists = []
        for arm, ac, mc in zip(arms, arm_colors, motor_colors):
            tip = R @ arm + np.array([x, y, z])
            line, = ax.plot([x, tip[0]], [y, tip[1]], [z, tip[2]],
                            '-', color=ac, linewidth=2.5, zorder=5)
            dot = ax.scatter(*tip, color=mc, s=40, zorder=6)
            artists += [line, dot]

        # Center body
        body = ax.scatter(x, y, z, color=color, s=60, zorder=7)
        artists.append(body)
        return artists

    @staticmethod
    def _rotation_matrix(phi, theta, psi):
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi),  np.cos(psi), 0],
                        [0,            0,           1]])
        Ry = np.array([[ np.cos(theta), 0, np.sin(theta)],
                        [0,             1, 0            ],
                        [-np.sin(theta), 0, np.cos(theta)]])
        Rx = np.array([[1, 0,          0         ],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi),  np.cos(phi)]])
        return Rz @ Ry @ Rx

    def animate(self, interval=30, trail_length=80):
        states    = np.array([d['state']    for d in self.data])
        targets   = np.array([d['target']   for d in self.data])
        times     = np.array([d['t']        for d in self.data])
        progress  = np.array([d['progress'] for d in self.data])

        positions = states[:, 0:3]
        angles    = states[:, 3:6]
        velocities= states[:, 6:9]

        # Axis limits
        pad = 1.5
        xl = (positions[:,0].min()-pad, positions[:,0].max()+pad)
        yl = (positions[:,1].min()-pad, positions[:,1].max()+pad)
        zl = (max(0, positions[:,2].min()-pad), positions[:,2].max()+pad)

        ax = self.ax3d

        # Static: waypoints
        ax.scatter(self.waypoints[:,0], self.waypoints[:,1], self.waypoints[:,2],
                   color='#ffff00', s=60, marker='*', zorder=3, label='Waypoints')
        # Desired path
        ax.plot(targets[:,0], targets[:,1], targets[:,2],
                '--', color='#555588', linewidth=1, alpha=0.6)

        ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
        ax.set_xlabel('X (m)', color='#8888bb', fontsize=8)
        ax.set_ylabel('Y (m)', color='#8888bb', fontsize=8)
        ax.set_zlabel('Z (m)', color='#8888bb', fontsize=8)
        ax.tick_params(colors='#666699', labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#222244')
        ax.yaxis.pane.set_edgecolor('#222244')
        ax.zaxis.pane.set_edgecolor('#222244')
        ax.grid(True, color='#1a1a33', linewidth=0.5)
        title = ax.set_title('Drone Autopilot Simulation', color='#00ffcc',
                              fontsize=11, fontweight='bold', pad=10)

        # Trail
        trail_line, = ax.plot([], [], [], '-', color='#00ffcc', linewidth=1.2,
                               alpha=0.7, zorder=4)

        # Side plots setup
        t_arr = times

        def setup_side(ax2, ylabel, color):
            ax2.set_xlim(t_arr[0], t_arr[-1])
            ax2.set_ylabel(ylabel, color=color, fontsize=7)
            ax2.set_xlabel('t (s)', color='#666699', fontsize=7)

        setup_side(self.ax_alt, 'Altitude (m)', '#00ffcc')
        setup_side(self.ax_att, 'Angles (rad)',  '#ff8844')
        setup_side(self.ax_vel, 'Speed (m/s)',   '#44aaff')

        alt_line,   = self.ax_alt.plot([], [], color='#00ffcc', linewidth=1.2)
        alt_ref,    = self.ax_alt.plot([], [], '--', color='#ffff00', linewidth=0.8, alpha=0.7)
        phi_line,   = self.ax_att.plot([], [], color='#ff4444', linewidth=1.0, label='φ')
        theta_line, = self.ax_att.plot([], [], color='#44ff44', linewidth=1.0, label='θ')
        psi_line,   = self.ax_att.plot([], [], color='#4444ff', linewidth=1.0, label='ψ')
        vx_line,    = self.ax_vel.plot([], [], color='#ff8844', linewidth=1.0, label='vx')
        vy_line,    = self.ax_vel.plot([], [], color='#44ffcc', linewidth=1.0, label='vy')
        vz_line,    = self.ax_vel.plot([], [], color='#aa88ff', linewidth=1.0, label='vz')

        self.ax_att.legend(fontsize=6, facecolor='#111122', labelcolor='white',
                           loc='upper right', framealpha=0.5)
        self.ax_vel.legend(fontsize=6, facecolor='#111122', labelcolor='white',
                           loc='upper right', framealpha=0.5)

        # Progress bar
        self.ax_prog.set_xlim(0, 1); self.ax_prog.set_ylim(0, 1)
        self.ax_prog.set_yticks([])
        prog_bg = self.ax_prog.barh(0.3, 1.0, height=0.4, color='#222244')
        prog_bar = self.ax_prog.barh(0.3, 0.0, height=0.4, color='#00ffcc')
        prog_text = self.ax_prog.text(0.5, 0.75, 'Mission: 0%',
                                       ha='center', va='center',
                                       color='#aaccff', fontsize=8,
                                       transform=self.ax_prog.transAxes)

        drone_artists = []

        def update(frame):
            nonlocal drone_artists
            for a in drone_artists:
                a.remove()
            drone_artists = []

            i = frame

            # Trail
            start = max(0, i - trail_length)
            trail_line.set_data(positions[start:i+1, 0], positions[start:i+1, 1])
            trail_line.set_3d_properties(positions[start:i+1, 2])

            # Drone body
            drone_artists = self._draw_drone_body(
                ax, positions[i], angles[i])

            # Side plots
            alt_line.set_data(t_arr[:i], positions[:i, 2])
            alt_ref.set_data(t_arr[:i],  targets[:i, 2])
            self.ax_alt.set_ylim(zl)

            phi_line.set_data(t_arr[:i],   angles[:i, 0])
            theta_line.set_data(t_arr[:i], angles[:i, 1])
            psi_line.set_data(t_arr[:i],   angles[:i, 2])
            self.ax_att.relim(); self.ax_att.autoscale_view()

            vx_line.set_data(t_arr[:i], velocities[:i, 0])
            vy_line.set_data(t_arr[:i], velocities[:i, 1])
            vz_line.set_data(t_arr[:i], velocities[:i, 2])
            self.ax_vel.relim(); self.ax_vel.autoscale_view()

            # Progress
            p = float(progress[i])
            prog_bar[0].set_width(p)
            prog_text.set_text(f'Mission Progress: {p*100:.0f}%  |  t = {t_arr[i]:.1f}s')

            title.set_text(f'Drone Autopilot Simulation  |  Alt: {positions[i,2]:.1f}m  '
                           f'|  φ={np.degrees(angles[i,0]):.1f}°  '
                           f'θ={np.degrees(angles[i,1]):.1f}°')

            return [trail_line] + drone_artists + [alt_line, alt_ref,
                    phi_line, theta_line, psi_line,
                    vx_line, vy_line, vz_line]

        n_frames = len(self.data)
        anim = animation.FuncAnimation(
            self.fig, update,
            frames=range(0, n_frames, max(1, n_frames // 300)),
            interval=interval, blit=False)

        plt.tight_layout()
        return anim
