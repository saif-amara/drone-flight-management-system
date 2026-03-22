import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


class DroneVisualizer:
    def __init__(self, trajectory_data, waypoints, arm_length=0.23, avoidance=None):
        self.data       = trajectory_data
        self.waypoints  = np.array(waypoints)
        self.arm_length = arm_length
        self.avoidance  = avoidance
        self.fig        = plt.figure(figsize=(15, 9), facecolor='#0d0d1a')
        self._setup_axes()

    def _setup_axes(self):
        self.ax3d  = self.fig.add_axes([0.00, 0.22, 0.62, 0.78], projection='3d', facecolor='#0d0d1a')
        self.ax_alt  = self.fig.add_axes([0.65, 0.68, 0.33, 0.25], facecolor='#111122')
        self.ax_att  = self.fig.add_axes([0.65, 0.38, 0.33, 0.25], facecolor='#111122')
        self.ax_bat  = self.fig.add_axes([0.65, 0.08, 0.33, 0.25], facecolor='#111122')
        self.ax_hud  = self.fig.add_axes([0.02, 0.02, 0.58, 0.18], facecolor='#111122')

        for ax in [self.ax_alt, self.ax_att, self.ax_bat, self.ax_hud]:
            ax.tick_params(colors='#aaaacc', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333366')

    @staticmethod
    def _rotation_matrix(phi, theta, psi):
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi),  np.cos(psi), 0],
                        [0, 0, 1]])
        Ry = np.array([[ np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
        Rx = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi),  np.cos(phi)]])
        return Rz @ Ry @ Rx

    def _draw_drone_body(self, ax, pos, angles):
        x, y, z = pos
        phi, theta, psi = angles
        L = self.arm_length
        R = self._rotation_matrix(phi, theta, psi)
        arms = [np.array([L,0,0]), np.array([-L,0,0]),
                np.array([0,L,0]), np.array([0,-L,0])]
        arm_colors   = ['#ff4444','#ff4444','#4444ff','#4444ff']
        motor_colors = ['#ffaa00','#ffaa00','#00aaff','#00aaff']
        artists = []
        for arm, ac, mc in zip(arms, arm_colors, motor_colors):
            tip = R @ arm + np.array([x, y, z])
            ln, = ax.plot([x, tip[0]], [y, tip[1]], [z, tip[2]], '-', color=ac, linewidth=2.5, zorder=5)
            dot = ax.scatter(*tip, color=mc, s=40, zorder=6)
            artists += [ln, dot]
        body = ax.scatter(x, y, z, color='#00ffcc', s=60, zorder=7)
        artists.append(body)
        return artists

    def animate(self, interval=30, trail_length=80):
        states    = np.array([d['state']    for d in self.data])
        targets   = np.array([d['target']   for d in self.data])
        times     = np.array([d['t']        for d in self.data])
        progress  = np.array([d['progress'] for d in self.data])
        battery   = np.array([d['battery']  for d in self.data])
        rth_flags = np.array([d['rth_active'] for d in self.data])

        positions  = states[:, 0:3]
        angles     = states[:, 3:6]
        velocities = states[:, 6:9]

        pad = 1.5
        xl = (positions[:,0].min()-pad, positions[:,0].max()+pad)
        yl = (positions[:,1].min()-pad, positions[:,1].max()+pad)
        zl = (0, positions[:,2].max()+pad)

        ax = self.ax3d

        # Waypoints
        ax.scatter(self.waypoints[:,0], self.waypoints[:,1], self.waypoints[:,2],
                   color='#ffff00', s=60, marker='*', zorder=3)

        # Desired path
        ax.plot(targets[:,0], targets[:,1], targets[:,2],
                '--', color='#555588', linewidth=1, alpha=0.5)

        # Geofence circle (if available from data)
        theta_c = np.linspace(0, 2*np.pi, 64)
        # Try to get geofence radius from first log entry
        gf_r = 15.0
        gf_xs = gf_r * np.cos(theta_c)
        gf_ys = gf_r * np.sin(theta_c)
        gf_zs = np.zeros(64)
        ax.plot(gf_xs, gf_ys, gf_zs, '--', color='#ff4444', linewidth=1, alpha=0.5, label='Geofence')

        # Obstacles
        if self.avoidance:
            for obs_pos, obs_r in self.avoidance.get_obstacles_for_viz():
                u = np.linspace(0, 2*np.pi, 16)
                v = np.linspace(0, np.pi, 8)
                ox = obs_pos[0] + obs_r * np.outer(np.cos(u), np.sin(v))
                oy = obs_pos[1] + obs_r * np.outer(np.sin(u), np.sin(v))
                oz = obs_pos[2] + obs_r * np.outer(np.ones(16), np.cos(v))
                ax.plot_surface(ox, oy, oz, color='#ff6600', alpha=0.4)

        ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
        ax.set_xlabel('X (m)', color='#8888bb', fontsize=8)
        ax.set_ylabel('Y (m)', color='#8888bb', fontsize=8)
        ax.set_zlabel('Z (m)', color='#8888bb', fontsize=8)
        ax.tick_params(colors='#666699', labelsize=7)
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#222244')
        ax.yaxis.pane.set_edgecolor('#222244')
        ax.zaxis.pane.set_edgecolor('#222244')
        ax.grid(True, color='#1a1a33', linewidth=0.5)

        title = ax.set_title('', color='#00ffcc', fontsize=10, fontweight='bold')

        trail_line, = ax.plot([], [], [], '-', color='#00ffcc', linewidth=1.2, alpha=0.7, zorder=4)

        # Side plots
        self.ax_alt.set_xlim(times[0], times[-1]); self.ax_alt.set_ylim(zl)
        self.ax_alt.set_ylabel('Alt (m)', color='#00ffcc', fontsize=7)
        self.ax_alt.set_xlabel('t (s)', color='#666699', fontsize=7)

        self.ax_att.set_xlim(times[0], times[-1])
        self.ax_att.set_ylabel('Angles (rad)', color='#ff8844', fontsize=7)
        self.ax_att.set_xlabel('t (s)', color='#666699', fontsize=7)

        self.ax_bat.set_xlim(times[0], times[-1]); self.ax_bat.set_ylim(0, 105)
        self.ax_bat.set_ylabel('Battery (%)', color='#44ff88', fontsize=7)
        self.ax_bat.set_xlabel('t (s)', color='#666699', fontsize=7)
        self.ax_bat.axhline(30, color='#ffaa00', linewidth=0.8, linestyle='--', alpha=0.7)
        self.ax_bat.axhline(15, color='#ff4444', linewidth=0.8, linestyle='--', alpha=0.7)

        alt_line,   = self.ax_alt.plot([], [], color='#00ffcc', linewidth=1.2)
        alt_ref,    = self.ax_alt.plot([], [], '--', color='#ffff00', linewidth=0.8, alpha=0.7)
        phi_line,   = self.ax_att.plot([], [], color='#ff4444', linewidth=1.0, label='φ')
        theta_line, = self.ax_att.plot([], [], color='#44ff44', linewidth=1.0, label='θ')
        bat_line,   = self.ax_bat.plot([], [], color='#44ff88', linewidth=1.5)

        self.ax_att.legend(fontsize=6, facecolor='#111122', labelcolor='white', loc='upper right', framealpha=0.5)

        # HUD
        self.ax_hud.set_xlim(0, 1); self.ax_hud.set_ylim(0, 1)
        self.ax_hud.set_xticks([]); self.ax_hud.set_yticks([])
        prog_bg  = self.ax_hud.barh(0.25, 1.0, height=0.3, color='#222244', left=0.15)
        prog_bar = self.ax_hud.barh(0.25, 0.0, height=0.3, color='#00ffcc', left=0.15)
        hud_text = self.ax_hud.text(0.5, 0.72, '', ha='center', va='center',
                                     color='#aaccff', fontsize=8,
                                     transform=self.ax_hud.transAxes)
        rth_text = self.ax_hud.text(0.5, 0.15, '', ha='center', va='center',
                                     color='#ff4444', fontsize=9, fontweight='bold',
                                     transform=self.ax_hud.transAxes)

        drone_artists = []

        def update(frame):
            nonlocal drone_artists
            for a in drone_artists:
                a.remove()
            drone_artists = []

            i = frame
            start = max(0, i - trail_length)

            trail_line.set_data(positions[start:i+1, 0], positions[start:i+1, 1])
            trail_line.set_3d_properties(positions[start:i+1, 2])
            drone_artists = self._draw_drone_body(ax, positions[i], angles[i])

            alt_line.set_data(times[:i], positions[:i, 2])
            alt_ref.set_data(times[:i], targets[:i, 2])

            phi_line.set_data(times[:i],   angles[:i, 0])
            theta_line.set_data(times[:i], angles[:i, 1])
            self.ax_att.relim(); self.ax_att.autoscale_view()

            bat_line.set_data(times[:i], battery[:i])
            bat_color = '#ff4444' if battery[i] < 15 else '#ffaa00' if battery[i] < 30 else '#44ff88'
            bat_line.set_color(bat_color)

            p = float(progress[i])
            prog_bar[0].set_width(p * 0.85)
            hud_text.set_text(
                f'Mission: {p*100:.0f}%  |  t={times[i]:.1f}s  |  '
                f'Alt={positions[i,2]:.1f}m  |  Bat={battery[i]:.0f}%'
            )

            rth_text.set_text('RTH ACTIVE' if rth_flags[i] else '')

            title.set_text(
                f'Drone FMS  |  Alt:{positions[i,2]:.1f}m  '
                f'Bat:{battery[i]:.0f}%  '
                f'{"RTH" if rth_flags[i] else "AUTO"}'
            )

            return [trail_line] + drone_artists + [alt_line, alt_ref,
                    phi_line, theta_line, bat_line]

        n = len(self.data)
        anim = animation.FuncAnimation(
            self.fig, update,
            frames=range(0, n, max(1, n // 300)),
            interval=interval, blit=False)

        plt.tight_layout()
        return anim
