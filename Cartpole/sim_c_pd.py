import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

model = mujoco.MjModel.from_xml_path("cartpole_min.xml")
data = mujoco.MjData(model)

data.qpos[0] = 0.0
data.qpos[1] = 0.2

data.qvel[0] = 0.0
data.qvel[1] = 0.0

mujoco.mj_forward(model, data)

kp_theta = 35
kd_theta = 5

kp_x = 0.5
kd_x = 0.2

F_MAX = 25.0

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
        x = data.qpos[0]
        theta = data.qpos[1]

        x_dot = data.qvel[0]
        theta_dot = data.qvel[1]

        if abs(theta) > np.radians(70):

            print("\nPole fell down!")
            

        force = (
            - kp_theta * theta
            - kd_theta * theta_dot
            - kp_x * x
            - kd_x * x_dot
        )


        if abs(x) > 1.3:

            force *= 0.6


        if x > 1.45:

            force -= 10
        elif x < -1.45:

            force += 10

        # force = np.clip(force, -F_MAX, F_MAX)

        data.ctrl[0] = force

        mujoco.mj_step(model, data)

        viewer.sync()


        log_time.append(data.time)

        log_theta.append(np.degrees(theta))
        log_theta_dot.append(np.degrees(theta_dot))

        log_x.append(x)
        log_x_dot.append(x_dot)

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
    "Cartpole PD Control with Track Limits",
    fontsize=16,
    fontweight="bold"
)

# ------------------------------------------------------------
# POLE ANGLE
# ------------------------------------------------------------

axes[0].plot(t, theta_vals, linewidth=2)

axes[0].axhline(0, linestyle="--")

axes[0].set_ylabel("Theta (deg)")
axes[0].set_title("Pole Angle")
axes[0].grid(True)

# ------------------------------------------------------------
# CART POSITION
# ------------------------------------------------------------

axes[1].plot(t, x_vals, linewidth=2)

axes[1].axhline(0, linestyle="--")
axes[1].axhline(1.5, linestyle=":")
axes[1].axhline(-1.5, linestyle=":")

axes[1].set_ylabel("Cart Position (m)")
axes[1].set_title("Cart Motion")
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
# CONTROL FORCE
# ------------------------------------------------------------

axes[3].plot(t, force_vals, linewidth=2)

axes[3].axhline(0, linestyle="--")

axes[3].set_ylabel("Force (N)")
axes[3].set_xlabel("Time (s)")
axes[3].set_title("Control Force")
axes[3].grid(True)

plt.tight_layout()

plt.show()