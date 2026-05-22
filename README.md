# Classical Control Systems in MuJoCo

Welcome to the **Classical Control Systems** repository! This project is a beginner-friendly playground for exploring dynamic physical systems and their control mechanisms using **Python** and the industry-standard **MuJoCo** (Multi-Joint dynamics with Contact) physics engine.

MuJoCo is renowned for its speed, accuracy, and contact handling, making it the choice platform for robotics research (including teams at Google DeepMind). This repository demonstrates how to simulate and control classical robotic benchmarks—such as the **Cartpole**, the **Double Pendulum**, the **Furuta Pendulum**, and the **Reaction Wheel Pendulum**—with simple, intuitive proportional-derivative (PD) controllers.

---

## Quick Start Guide

Getting started is simple. You only need a modern Python installation (Python 3.8+) and a few standard libraries.

### 1. Installation
Open your terminal and run the following command to install the required packages:

```bash
pip install mujoco numpy matplotlib
```

> [!NOTE]
> The passive visualizer window will open automatically when you run the scripts. You can use your mouse to orbit (left-click + drag), pan (right-click + drag), and zoom (scroll wheel).

---

## Repository Structure

Below is an overview of the directories and files in this workspace:

```text
Classical_Control_Systems/
├── Cartpole/
│   ├── cartpole.xml             # Detailed MJCF (XML) model of the cartpole
│   ├── cartpole_min.xml         # Minimized MJCF model (cleaner structure)
│   └── sim_c_pd.py              # PD-control simulation script
├── Double_pendulum/
│   ├── double_pedulum.xml       # MJCF model of the passive double pendulum
│   ├── simulate.py              # Integrator comparison script (Euler vs. RK4 vs. Implicit)
│   └── Euler/ RK4/ implicit/    # Pre-rendered simulation analysis plots
├── Futura _Pendulum/         # Furuta (Rotational Inverted) Pendulum
│   ├── furuta.xml               # MJCF model for the Furuta pendulum
│   └── sim_pd.py                # Joint-space PD-balancing simulation script
└── Reaction_Wheel_Pendulum/
    ├── rwp.xml                  # MJCF model for the reaction wheel system
    └── sim..py                  # PD-balancing simulation script
```

---

## System Explanations

Each system illustrates a classic control challenge. Let's break down how they work and the math behind their balance.

---

### 1. Cartpole (Inverted Pendulum on a Cart)
The **Cartpole** is the ultimate benchmark in control theory. It consists of a pole attached by an unactuated hinge joint to a cart, which moves along a linear track.

*   **The Challenge:** The actuator can only apply horizontal force directly to the cart. We must slide the cart left and right to balance the pole upright.
*   **The Physics:**
    *   $x$: Cart position (meters)
    *   $\theta$: Pole angle relative to vertical (radians)

#### Control Strategy
We use a coupled **PD controller** that outputs control force $F$ using both the pole angle error and the cart position error:

$$F = -K_{p,\theta} \cdot \theta - K_{d,\theta} \cdot \dot{\theta} - K_{p,x} \cdot x - K_{d,x} \cdot \dot{x}$$

Where:
*   $-K_{p,\theta} \cdot \theta$ pulls the cart in the direction the pole is falling to "catch" it.
*   $-K_{d,\theta} \cdot \dot{\theta}$ dampens the pole's angular velocity.
*   $-K_{p,x} \cdot x$ gently coaxes the cart back toward the center of the track ($x=0$).
*   $-K_{d,x} \cdot \dot{x}$ prevents the cart from moving too fast.

#### How to Run
```bash
python "Cartpole/sim_c_pd.py"
```

---

### 2. Furuta Pendulum (Rotational Inverted Pendulum)
Invented by Katsuhisa Furuta in 1992, the **Furuta Pendulum** features a horizontal arm rotating in a horizontal plane, powered by a motor. Attached to the end of this arm is a free-spinning pendulum link that rotates in a vertical plane.

*   **The Challenge:** Like the cartpole, this is an **underactuated** system. We can only control the rotation of the base arm ($\alpha$). We must swing the arm back and forth to keep the vertical pendulum ($\theta$) balanced upright.
*   **The Physics:**
    *   $\alpha$: Base arm angle (horizontal rotation)
    *   $\theta$: Pendulum angle (vertical rotation; upright equilibrium is at $\pi$ rad)

#### Control Strategy
Because angles wrap around, we compute the angular error $e_\theta$ using trigonometry to find the shortest path to upright balance:

$$e_\theta = \text{atan2}(\sin(\theta - \pi), \cos(\theta - \pi))$$

The motor torque $\tau$ applied to the horizontal arm is:

$$\tau = K_{p,\theta} \cdot e_\theta + K_{d,\theta} \cdot \dot{\theta} - K_{p,\alpha} \cdot \alpha - K_{d,\alpha} \cdot \dot{\alpha}$$

#### How to Run
```bash
python "Futura _Pendulum/sim_pd.py"
```

---

### 3. Reaction Wheel Pendulum
This system features a pendulum arm attached to a pivot. Mounted at the free tip of the pendulum is a heavy wheel (the reaction wheel) driven by an electric motor.

*   **The Challenge:** The main pendulum joint is completely free and has no motor. Instead, by accelerating or decelerating the reaction wheel, we generate an equal and opposite reaction torque on the pendulum arm itself, allowing us to balance it upright.
*   **The Physics:**
    *   $\theta$: Pendulum arm angle (upright is at $\pi$ rad)
    *   $\omega_w$: Speed of the reaction wheel (rad/s)

#### Control Strategy
The control torque applied to the wheel motor is:

$$\tau = -K_{p,\theta} \cdot e_\theta - K_{d,\theta} \cdot \dot{\theta} - K_{w} \cdot \omega_w$$

> [!TIP]
> Notice the $-K_w \cdot \omega_w$ term. Without it, the reaction wheel would spin faster and faster (saturating the motor) just to counteract constant gravity offsets. This term dampens the wheel speed, eventually bringing the wheel to a halt once balance is achieved!

#### How to Run
```bash
python "Reaction_Wheel_Pendulum/sim..py"
```

---

### 4. Double Pendulum (Numerical Integration Case Study)
Unlike the other folders, `Double_pendulum` does not run active control. Instead, it serves as a fascinating case study in **computational physics and numerical integration methods**.

A double pendulum is a chaotic system. Minor errors in how a computer calculates the physics step-by-step can lead to vastly different trajectories. This script simulates a passive double pendulum over 50 seconds and compares four integrators across four different timesteps ($dt = 0.001\text{s}, 0.002\text{s}, 0.005\text{s}, 0.01\text{s}$):

1.  **Euler (`mjINT_EULER`)**: Explicit Euler integration. First-order accuracy. Known to add energy over time, causing simulated systems to "explode" at larger timesteps.
2.  **RK4 (`mjINT_RK4`)**: 4th-order Runge-Kutta. Highly accurate explicit method, widely used in game engines and graphics because it preserves system energy much better than Euler.
3.  **Implicit (`mjINT_IMPLICIT`)**: MuJoCo's default Newton-based implicit solver. Extremely stable even at huge timesteps, but introduces artificial numerical damping (absorbing energy to keep the math stable).
4.  **Implicit Fast (`mjINT_IMPLICITFAST`)**: A faster approximation of the implicit solver.

#### Energy Conservation and Drift
Since the pendulum is passive, the total energy (Kinetic + Potential) should ideally remain perfectly constant:

$$E_{\text{total}} = T + V = \text{Constant}$$

By plotting the energy error $E(t) - E(0)$, you can visually observe:
*   **Euler** accumulates energy rapidly (positive drift) and becomes unstable.
*   **RK4** maintains beautiful, near-zero energy drift at small timesteps, but can struggle at large timesteps.
*   **Implicit methods** drain energy (negative drift) over time but never crash.

#### How to Run
```bash
python "Double_pendulum/simulate.py"
```

---

## Tuning PD Controllers: A Beginner's Guide

If you want to experiment with changing the gains (like `kp_theta`, `kd_theta`), here is an intuitive way to understand what they do:

| Parameter | Control Term | Intuitive Analogy | Too Low | Too High |
| :--- | :--- | :--- | :--- | :--- |
| **$K_p$** | Proportional | **The Spring:** Pulls the system harder the farther it is from the goal. | Sluggish, never reaches target, droops under gravity. | Violent oscillations, overshoot, unstable behavior. |
| **$K_d$** | Derivative | **The Shock Absorber:** Opposes motion. Acts like friction to slow down moving parts. | System oscillates forever without settling. | Jittery movements, stiff response, locks up motion. |

### The Golden Rule of Tuning
If you are trying to tune a controller from scratch, always use this iterative sequence:
1.  **Set all gains to 0.** The system will fall down helplessly.
2.  **Increase $K_p$ slowly** until the system responds and oscillates back and forth around the target position. It will overshoot and wobble continuously.
3.  **Increase $K_d$ slowly** to act as a brake. This will damp out the oscillations, letting the system settle smoothly at the target.
4.  If the system droops (e.g. balances slightly below upright due to gravity), increase $K_p$ a bit more, then increase $K_d$ to match.

---

## Understanding MuJoCo MJCF Files

Each folder contains an `.xml` file which describes the system's physical properties. If you open these files, you will notice these core blocks:

*   `<worldbody>`: Where we define the bodies, their visual geometries (`<geom type="capsule">`), joints, masses, and positions.
*   `<joint>`: Defines how bodies connect. A `hinge` joint allows rotation (like an elbow), and a `slide` joint allows linear movement (like the cartpole).
*   `<actuator>`: Defines where we can apply forces. In the cartpole, we actuate the slide joint. In the Furuta, we actuate the base vertical joint. In the Reaction Wheel, we actuate the wheel spinning joint.

Have fun exploring and tuning! Modifying these systems is a fantastic way to master the fundamentals of robotics, physics simulation, and classical control theory.
