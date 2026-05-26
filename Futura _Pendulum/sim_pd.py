import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

model = mujoco.MjModel.from_xml_path("furuta.xml")
data = mujoco.MjData(model)

data.qpos[0] = 0
data.qpos[1] = np.pi -0.3

data.qvel[0] = 0
data.qvel[1] = 0

mujoco.mj_forward(model, data)

kp_theta = 57.5
kd_theta = 11

kp_alpha = -2.3
kd_alpha = -2

F_MAX = 20.0

log_time = []
log_theta = []
log_theta_dot = []

log_x = []
log_x_dot = []

log_force = []

def setup_camera(viewer):

    viewer.cam.distance = 5.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -10

    viewer.cam.lookat[:] = [0, 0, 0.5]


with mujoco.viewer.launch_passive(model, data) as viewer:

    setup_camera(viewer)

    while viewer.is_running():
        alpha = data.qpos[0]
        theta = data.qpos[1]

        alpha_dot = data.qvel[0]
        theta_dot = data.qvel[1]

        e_theta = theta - np.pi
        e_alpha = alpha
        
        force = (
            - kp_theta * e_theta
            - kd_theta * theta_dot
            - kp_alpha * e_alpha
            - kd_alpha * alpha_dot
        )

        force = np.clip(force, -F_MAX, F_MAX)

        data.ctrl[0] = force

        mujoco.mj_step(model, data)

        viewer.sync()


        log_time.append(data.time)

        log_theta.append(np.degrees(theta))
        log_theta_dot.append(np.degrees(theta_dot))

        log_x.append(np.degrees(alpha))
        log_x_dot.append(np.degrees(alpha_dot))

        log_force.append(force)

        time.sleep(model.opt.timestep)

# ============================================================
# CONVERT LOGS TO NUMPY
# ============================================================

t = np.array(log_time)

theta_vals = np.array(log_theta)
theta_dot_vals = np.array(log_theta_dot)

x_vals = np.array(log_x)
x_dot_vals = np.array(log_x_dot)

force_vals = np.array(log_force)

# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

fig.suptitle(
    "Furuta Pendulum PD Control",
    fontsize=16,
    fontweight="bold"
)

# ------------------------------------------------------------
# POLE ANGLE
# ------------------------------------------------------------

axes[0].plot(t, theta_vals, linewidth=2)

axes[0].axhline(180, linestyle="--")

axes[0].set_ylabel("Theta (deg)")
axes[0].set_title("Pole Angle")
axes[0].grid(True)

# ------------------------------------------------------------
# ARM POSITION
# ------------------------------------------------------------

axes[1].plot(t, x_vals, linewidth=2)

axes[1].axhline(0, linestyle="--")

axes[1].set_ylabel("Alpha (deg)")
axes[1].set_title("Arm Angle")
axes[1].grid(True)

# ------------------------------------------------------------
# ANGULAR VELOCITY
# ------------------------------------------------------------

axes[2].plot(t, theta_dot_vals, linewidth=2)

axes[2].axhline(0, linestyle="--")

axes[2].set_ylabel("Theta Dot (deg/s)")
axes[2].set_title("Pole Angular Velocity")
axes[2].grid(True)

# ------------------------------------------------------------
# CONTROL TORQUE
# ------------------------------------------------------------

axes[3].plot(t, force_vals, linewidth=2)

axes[3].axhline(0, linestyle="--")

axes[3].set_ylabel("Torque (Nm)")
axes[3].set_xlabel("Time (s)")
axes[3].set_title("Control Torque")
axes[3].grid(True)

plt.tight_layout()

plt.show()
