import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

XML_FILE = "double_pedulum.xml"

TIMESTEPS = [0.001, 0.002, 0.005, 0.01]

INTEGRATORS = [
    "Euler",
    "RK4",
    "implicit",
    "implicitfast"
]

SIM_TIME = 50.0

# ============================================================
# ENERGY FUNCTION
# ============================================================

def compute_energy(q1, q2, q1dot, q2dot,
                   m1=1.0, m2=1.0,
                   l1=1.0, l2=1.0,
                   g=9.81):

    # ------------------------
    # Positions
    # ------------------------

    x1 = l1 * np.sin(q1)
    y1 = -l1 * np.cos(q1)

    x2 = x1 + l2 * np.sin(q1 + q2)
    y2 = y1 - l2 * np.cos(q1 + q2)

    # ------------------------
    # Velocities
    # ------------------------

    vx1 = l1 * q1dot * np.cos(q1)
    vy1 = l1 * q1dot * np.sin(q1)

    vx2 = vx1 + l2 * (q1dot + q2dot) * np.cos(q1 + q2)
    vy2 = vy1 + l2 * (q1dot + q2dot) * np.sin(q1 + q2)

    # ------------------------
    # Kinetic Energy
    # ------------------------

    T1 = 0.5 * m1 * (vx1**2 + vy1**2)
    T2 = 0.5 * m2 * (vx2**2 + vy2**2)

    T = T1 + T2

    # ------------------------
    # Potential Energy
    # ------------------------

    V1 = m1 * g * y1
    V2 = m2 * g * y2

    V = V1 + V2

    # ------------------------
    # Total Energy
    # ------------------------

    E = T + V

    return E


# ============================================================
# RUN SIMULATION
# ============================================================

results = {}

for integrator in INTEGRATORS:

    results[integrator] = {}

    for dt in TIMESTEPS:

        print(f"Running: {integrator} | dt={dt}")

        # ------------------------------------------------
        # Load model
        # ------------------------------------------------

        model = mujoco.MjModel.from_xml_path("double_pedulum.xml")

        # timestep
        model.opt.timestep = dt

        # integrator
        if integrator == "Euler":
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

        elif integrator == "RK4":
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4

        elif integrator == "implicit":
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT

        elif integrator == "implicitfast":
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        data = mujoco.MjData(model)

        # ------------------------------------------------
        # Initial condition
        # ------------------------------------------------

        data.qpos[:] = [1.0, 0.5]
        data.qvel[:] = [0.0, 0.0]

        # passive simulation
        data.ctrl[:] = [0.0]

        # ------------------------------------------------
        # Storage
        # ------------------------------------------------

        times = []
        q1s = []
        q2s = []
        energies = []

        # ------------------------------------------------
        # Simulation loop
        # ------------------------------------------------

        steps = int(SIM_TIME / dt)

        for step in range(steps):

            mujoco.mj_step(model, data)

            q1 = data.qpos[0]
            q2 = data.qpos[1]

            q1dot = data.qvel[0]
            q2dot = data.qvel[1]

            energy = compute_energy(
                q1, q2,
                q1dot, q2dot
            )

            times.append(step * dt)
            q1s.append(q1)
            q2s.append(q2)
            energies.append(energy)

        # ------------------------------------------------
        # Store results
        # ------------------------------------------------

        results[integrator][dt] = {
            "time": np.array(times),
            "q1": np.array(q1s),
            "q2": np.array(q2s),
            "energy": np.array(energies)
        }

# ============================================================
# PLOT 1
# ANGLE vs TIME
# ============================================================

for integrator in INTEGRATORS:

    plt.figure(figsize=(10, 5))

    for dt in TIMESTEPS:

        t = results[integrator][dt]["time"]
        q1 = results[integrator][dt]["q1"]

        plt.plot(t, q1, label=f"dt={dt}")

    plt.title(f"{integrator} : q1 vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("q1 (rad)")
    plt.legend()
    plt.grid()

# ============================================================
# PLOT 2
# ENERGY vs TIME
# ============================================================

for integrator in INTEGRATORS:

    plt.figure(figsize=(10, 5))

    for dt in TIMESTEPS:

        t = results[integrator][dt]["time"]
        E = results[integrator][dt]["energy"]

        plt.plot(t, E, label=f"dt={dt}")

    plt.title(f"{integrator} : Energy vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid()

# ============================================================
# PLOT 3
# ENERGY ERROR
# ============================================================

for integrator in INTEGRATORS:

    plt.figure(figsize=(10, 5))

    for dt in TIMESTEPS:

        t = results[integrator][dt]["time"]
        E = results[integrator][dt]["energy"]

        E0 = E[0]

        error = E - E0

        plt.plot(t, error, label=f"dt={dt}")

    plt.title(f"{integrator} : Energy Drift")
    plt.xlabel("Time (s)")
    plt.ylabel("E(t) - E(0)")
    plt.legend()
    plt.grid()

plt.show()