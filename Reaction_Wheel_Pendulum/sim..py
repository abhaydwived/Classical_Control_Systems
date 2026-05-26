import time

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("rwp.xml")
data = mujoco.MjData(model)

# Start near upright (pi = straight up from hanging-down zero)
data.qpos[0] = 0.3
data.qpos[1] = 0.0
data.qvel[0] = 0.0
data.qvel[1] = 0.0

mujoco.mj_forward(model, data)

# Tuned gains for upright balance
kp_theta = 40
kd_theta = 6.
kw = 0.*2
kp_alpha = 0.*0.5

alpha = 0.2
beta = 0.5

prev_torque = 0
filtered_theta_dot = 0

TORQUE_MAX = 150.0

log_time, log_theta, log_theta_dot, log_wheel_speed, log_torque = [], [], [], [], []

def setup_camera(viewer):
    viewer.cam.distance = 3.0
    viewer.cam.azimuth  = 90
    viewer.cam.elevation = -10
    viewer.cam.lookat[:] = [0, 0, 1]

with mujoco.viewer.launch_passive(model, data) as viewer:
    setup_camera(viewer)
    while viewer.is_running():
        theta       = data.qpos[0]
        theta_dot   = data.qvel[0]
        wheel_speed = data.qvel[1]
        wheel_angle = np.tanh(0.3 * data.qpos[1])

        # Error from upright equilibrium (pi radians)
        theta_error = 0. - theta #- np.pi

        # print(np.degrees(data.qpos[0]))

        # Wrap angle error to [-pi, pi]
        theta_error = (theta_error + np.pi) % (2 * np.pi) - np.pi
        filtered_theta_dot = (
            beta * theta_dot +
            (1 - beta) * filtered_theta_dot
        )

        if abs(theta_error) < np.radians(1):
            theta_error = 0

        torque = (
            kp_theta * theta
            +kd_theta * theta_dot
           # -kw * wheel_speed
            #-kp_alpha * wheel_angle
        )

        torque = np.clip(torque, -TORQUE_MAX, TORQUE_MAX)

        # torque = (
        #     alpha * torque +
        #     (1 - alpha) * prev_torque
        # )

        # prev_torque = torque

        data.ctrl[0] = torque   # wheel motor only
        # data.ctrl[1] = 0      # stick motor unused

        mujoco.mj_step(model, data)
        viewer.sync()

        log_time.append(data.time)
        log_theta.append(np.degrees(theta))
        log_theta_dot.append(np.degrees(theta_dot))
        log_wheel_speed.append(np.degrees(wheel_speed))
        log_torque.append(torque)
        time.sleep(model.opt.timestep)

# ============================================================
# CONVERT TO NUMPY
# ============================================================

t = np.array(log_time)

theta_vals = np.array(log_theta)
theta_dot_vals = np.array(log_theta_dot)

wheel_speed_vals = np.array(log_wheel_speed)

torque_vals = np.array(log_torque)

# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(11, 12),
    sharex=True
)

fig.suptitle(
    "Reaction Wheel Pendulum PD Control",
    fontsize=16,
    fontweight="bold"
)

# ============================================================
# PENDULUM ANGLE
# ============================================================

axes[0].plot(t, theta_vals, linewidth=2)

axes[0].axhline(0, linestyle="--")

axes[0].set_ylabel("Theta (deg)")
axes[0].set_title("Pendulum Angle")

axes[0].grid(True)

# ============================================================
# PENDULUM ANGULAR VELOCITY
# ============================================================

axes[1].plot(t, theta_dot_vals, linewidth=2)

axes[1].axhline(0, linestyle="--")

axes[1].set_ylabel("Theta Dot (deg/s)")
axes[1].set_title("Pendulum Angular Velocity")

axes[1].grid(True)

# ============================================================
# WHEEL SPEED
# ============================================================

axes[2].plot(t, wheel_speed_vals, linewidth=2)

axes[2].axhline(0, linestyle="--")

axes[2].set_ylabel("Wheel Speed (deg/s)")
axes[2].set_title("Reaction Wheel Speed")

axes[2].grid(True)

# ============================================================
# CONTROL TORQUE
# ============================================================

axes[3].plot(t, torque_vals, linewidth=2)

axes[3].axhline(0, linestyle="--")

axes[3].set_ylabel("Torque")
axes[3].set_xlabel("Time (s)")

axes[3].set_title("Motor Torque")

axes[3].grid(True)

# ============================================================
# SHOW
# ============================================================

plt.tight_layout()

plt.show()