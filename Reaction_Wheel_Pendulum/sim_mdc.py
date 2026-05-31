import time

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

# LOAD MODEL

model = mujoco.MjModel.from_xml_path("rwp.xml")
data = mujoco.MjData(model)

# INITIAL CONDITIONS

# Upright balancing starts near theta = 0
data.qpos[0] = 0.3
data.qpos[1] = 0.0

data.qvel[0] = 0.0
data.qvel[1] = 0.0

mujoco.mj_forward(model, data)

# PHYSICAL PARAMETERS

# You should later extract these from XML accurately

m = 1.8

l = 0.42

Ip = 0.46

g = 9.81

# Effective pendulum inertia
M = Ip + m * l**2

# CONTROLLER GAINS

kp_theta = 20.0
kd_theta = 7.0

# Wheel damping
kw = 0

# TORQUE LIMIT

TORQUE_MAX = 150.0

# LOGGING

log_time = []

log_theta = []
log_theta_dot = []

log_wheel_speed = []

log_torque = []

# CAMERA

def setup_camera(viewer):

    viewer.cam.distance = 3.0

    viewer.cam.azimuth = 90

    viewer.cam.elevation = -10

    viewer.cam.lookat[:] = [0, 0, 1]


with mujoco.viewer.launch_passive(model, data) as viewer:

    setup_camera(viewer)

    while viewer.is_running():

        theta = data.qpos[0]

        theta_dot = data.qvel[0]

        wheel_speed = data.qvel[1]

        # Upright equilibrium at theta = 0

        theta_error = theta

        # Small deadband

        if abs(theta_error) < np.radians(1):

            theta_error = 0

        # DESIRED ANGULAR ACCELERATION

        theta_ddot_des = (
            -kp_theta * theta_error
            -kd_theta * theta_dot
        )

        # MODEL TERMS

        gravity_term = m * g * l * np.sin(theta)
        
        torque = (
    
            -M * theta_ddot_des
            -gravity_term
            # -kw * wheel_speed
        )

        # TORQUE LIMIT

        torque = np.clip(
            torque,
            -TORQUE_MAX,
            TORQUE_MAX
        )

        data.ctrl[0] = torque

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)


        log_time.append(data.time)

        log_theta.append(
            np.degrees(theta)
        )

        log_theta_dot.append(
            np.degrees(theta_dot)
        )

        log_wheel_speed.append(
            np.degrees(wheel_speed)
        )

        log_torque.append(
            torque
        )

# CONVERT TO NUMPY

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
    "Reaction Wheel Pendulum Model-Based Control",
    fontsize=16,
    fontweight="bold"
)

# ============================================================
# PENDULUM ANGLE
# ============================================================

axes[0].plot(
    t,
    theta_vals,
    linewidth=2
)

axes[0].axhline(
    0,
    linestyle="--"
)

axes[0].set_ylabel(
    "Theta (deg)"
)

axes[0].set_title(
    "Pendulum Angle"
)

axes[0].grid(True)

# ============================================================
# ANGULAR VELOCITY
# ============================================================

axes[1].plot(
    t,
    theta_dot_vals,
    linewidth=2
)

axes[1].axhline(
    0,
    linestyle="--"
)

axes[1].set_ylabel(
    "Theta Dot (deg/s)"
)

axes[1].set_title(
    "Pendulum Angular Velocity"
)

axes[1].grid(True)

# ============================================================
# WHEEL SPEED
# ============================================================

axes[2].plot(
    t,
    wheel_speed_vals,
    linewidth=2
)

axes[2].axhline(
    0,
    linestyle="--"
)

axes[2].set_ylabel(
    "Wheel Speed (deg/s)"
)

axes[2].set_title(
    "Reaction Wheel Speed"
)

axes[2].grid(True)

# ============================================================
# CONTROL TORQUE
# ============================================================

axes[3].plot(
    t,
    torque_vals,
    linewidth=2
)

axes[3].axhline(
    0,
    linestyle="--"
)

axes[3].set_ylabel(
    "Torque"
)

axes[3].set_xlabel(
    "Time (s)"
)

axes[3].set_title(
    "Control Torque"
)

axes[3].grid(True)

# ============================================================
# SHOW
# ============================================================

plt.tight_layout()

plt.show()