import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path("cartpole_min.xml")
data = mujoco.MjData(model)

# ============================================================
# INITIAL CONDITIONS
# ============================================================

data.qpos[0] = 0.0
data.qpos[1] = 0.3

data.qvel[0] = 0.0
data.qvel[1] = 0.0

mujoco.mj_forward(model, data)

# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

M = 1.0      # cart mass
m = 0.25      # pole + bob mass

l = 0.35     # COM distance

g = 9.81

# ============================================================
# CONTROLLER GAINS
# ============================================================

kp_theta = 35.0
kd_theta = 8.0

kp_x = 2.0
kd_x = 3.0

# ============================================================
# FORCE LIMIT
# ============================================================

F_MAX = 40.0

# ============================================================
# FILTER
# ============================================================

beta = 0.5

filtered_theta_dot = 0.0

# ============================================================
# LOGGING
# ============================================================

log_time = []

log_theta = []
log_theta_dot = []

log_x = []
log_x_dot = []

log_force = []

# ============================================================
# CAMERA
# ============================================================

def setup_camera(viewer):

    viewer.cam.distance = 5.0

    viewer.cam.azimuth = 90

    viewer.cam.elevation = -10

    viewer.cam.lookat[:] = [0, 0, 0.5]

# ============================================================
# SIMULATION
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    setup_camera(viewer)

    while viewer.is_running():

        # ====================================================
        # STATES
        # ====================================================

        x = data.qpos[0]

        theta = data.qpos[1]

        x_dot = data.qvel[0]

        theta_dot = data.qvel[1]

        # ====================================================
        # FILTERED DERIVATIVE
        # ====================================================

        filtered_theta_dot = (
            beta * theta_dot
            + (1 - beta) * filtered_theta_dot
        )

        theta_dot_used = filtered_theta_dot

        # ====================================================
        # FALL DETECTION
        # ====================================================

        if abs(theta) > np.radians(70):

            print("\nPole fell!")

        # ====================================================
        # DESIRED ANGULAR ACCELERATION
        # ====================================================

        theta_ddot_des = (
            -kp_theta * theta
            -kd_theta * theta_dot_used
        )

        # ====================================================
        # MODEL-BASED FORCE
        # ====================================================

        gravity_term = (
            m * g * np.sin(theta)
        )

        centripetal_term = (
            m * l * theta_dot**2 * np.sin(theta)
        )

        coupling_term = (
            m * l * theta_ddot_des * np.cos(theta)
        )

        # ====================================================
        # POSITION STABILIZATION
        # ====================================================

        position_term = (
            -kp_x * x
            -kd_x * x_dot
        )

        # ====================================================
        # FINAL CONTROL FORCE
        # ====================================================

        force = (
            coupling_term
            + gravity_term
            - centripetal_term
            + position_term
        )

        # ====================================================
        # TRACK EDGE SAFETY
        # ====================================================

        if x > 1.35:

            force -= 10

        elif x < -1.35:

            force += 10

        # ====================================================
        # FORCE LIMIT
        # ====================================================

        force = np.clip(
            force,
            -F_MAX,
            F_MAX
        )

        # ====================================================
        # APPLY CONTROL
        # ====================================================

        data.ctrl[0] = force

        # ====================================================
        # STEP
        # ====================================================

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)

        # ====================================================
        # LOGGING
        # ====================================================

        log_time.append(data.time)

        log_theta.append(
            np.degrees(theta)
        )

        log_theta_dot.append(
            np.degrees(theta_dot)
        )

        log_x.append(x)

        log_x_dot.append(x_dot)

        log_force.append(force)

# ============================================================
# NUMPY
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

fig, axes = plt.subplots(
    4,
    1,
    figsize=(11, 12),
    sharex=True
)

fig.suptitle(
    "Cartpole Model-Based Control",
    fontsize=16,
    fontweight="bold"
)

# ============================================================
# POLE ANGLE
# ============================================================

axes[0].plot(t, theta_vals, linewidth=2)

axes[0].axhline(0, linestyle="--")

axes[0].set_ylabel("Theta (deg)")

axes[0].set_title("Pole Angle")

axes[0].grid(True)

# ============================================================
# CART POSITION
# ============================================================

axes[1].plot(t, x_vals, linewidth=2)

axes[1].axhline(0, linestyle="--")

axes[1].axhline(1.5, linestyle=":")

axes[1].axhline(-1.5, linestyle=":")

axes[1].set_ylabel("Cart Position")

axes[1].set_title("Cart Motion")

axes[1].grid(True)

# ============================================================
# ANGULAR VELOCITY
# ============================================================

axes[2].plot(t, theta_dot_vals, linewidth=2)

axes[2].axhline(0, linestyle="--")

axes[2].set_ylabel("Theta Dot")

axes[2].set_title("Pole Angular Velocity")

axes[2].grid(True)

# ============================================================
# CONTROL FORCE
# ============================================================

axes[3].plot(t, force_vals, linewidth=2)

axes[3].axhline(0, linestyle="--")

axes[3].set_ylabel("Force")

axes[3].set_xlabel("Time (s)")

axes[3].set_title("Control Force")

axes[3].grid(True)

plt.tight_layout()

plt.show()