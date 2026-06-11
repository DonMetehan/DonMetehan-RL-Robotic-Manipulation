# DonMetehan-RL-Robotic-Manipulation
Master's thesis code for RL-based robotic manipulation tasks using the Cross-Entropy Method
# Reinforcement Learning for Sequential Robotic Manipulation Tasks

This repository contains the code and implementation details for my Master's Thesis. The project focuses on solving a complex, multi-stage robotic manipulation task (Converging, Pushing, and Lifting) in a simulated environment (Robosuite / MuJoCo) using the Cross-Entropy Method (CEM) for parameter optimization.

## 🎯 Project Overview
Unlike traditional "pick-and-place" tasks, this project explores **Environmental Contact Exploitation**. The Panda robot must push an object (bread) against a wall and utilize the friction/surface of the wall to slide and lift the object to a specific target coordinate.

The task is divided into three sequential execution phases:
1. **Converge Phase:** Safely approaching the object.
2. **Push Phase:** Pushing the object against the wall while maintaining stable contact dynamics.
3. **Lift Phase:** Exploiting the wall contact to lift the object to the target coordinate.

## ⚙️ Methodology & Control Architecture
Instead of using opaque Deep Neural Networks, this project implements a highly interpretable, deterministic **State-Feedback Linear Policy**:
$u = \theta_1 s + \theta_2$

The control parameters ($\theta_1$ and $\theta_2$) are optimized using the **Cross-Entropy Method (CEM)**. 

### Key Engineering Highlights:
* **Dense Reward Shaping:** Implemented logarithmic distance penalties and dynamic thresholds to encourage millimeter-level precision and prevent reward hacking.
* **Kinematic & Safety Constraints:** Designed strict collision detection and Z-axis (height) penalties to ensure the robot doesn't bypass physical constraints or lose object contact.
* **State-Based Execution:** The system dynamically loads successful parameters from previous phases to execute the full sequence seamlessly.

## 📊 Results & Visualization

**Phase Execution & Distance Minimization:**
*(The graph below demonstrates the continuous reduction of distance across the Converge, Push, and Lift phases.)*

<img width="640" height="480" alt="Rollout 1 Distances" src="https://github.com/user-attachments/assets/bfc2b7c0-7ea2-432e-9258-ef1075f472df" />


**Simulation (Push & Lift Phase):**
*(The robot successfully pushes the object and attempts the wall-supported lift.)*



https://github.com/user-attachments/assets/9ab4857f-0b75-49ac-bdcf-505a10a0d907



## 🚧 Challenges & Future Work
During the development of the **Lift Phase**, a significant control engineering challenge was encountered. Relying solely on a Linear Policy proved to be a bottleneck for the highly non-linear dynamics of lifting an object against a wall. 

**Observations & Planned Improvements:**
* **Policy Limitations:** A simple linear mapping is insufficient for the precise angular manipulation required during the lift.
* **Spatula/Gripper Orientation:** Successful lifting requires the gripper tip to act as a spatula, sliding perfectly under the object at a very specific angle before applying upward force.
* **Next Steps:** I have actively started experimenting with a **6-DoF (Degrees of Freedom) Policy** and introduced a "Pre-Lift" alignment phase to decouple the orientation correction from the upward translation. Transitioning to a more complex non-linear controller or adding orientation-specific dense rewards is the next logical step to perfect the Lift phase.

## 🛠️ Tech Stack
* **Simulation:** Robosuite, MuJoCo
* **Algorithm:** Cross-Entropy Method (CEM), Reinforcement Learning
* **Languages & Libraries:** Python, NumPy, SciPy, Matplotlib
