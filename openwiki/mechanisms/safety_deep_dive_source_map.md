# VLA Safety Deep Dive (Source Map)

This page serves as a comprehensive map linking concepts, mechanisms, and core resources for the entire VLA safety domain. Use this to trace relationships between different topics.

## 🗺️ Core Domains & Concepts
*   **Domain Definition:** Start here to understand what physical hazard means in an embodied agent context. [Safety Domain Concepts](openwiki/domain/safety_concepts.md).
    *   *Key Question:* How does text "profanity" map to a robot collision?
*   **Architectural Flow:** Understand the high-level sequence of checks that must occur during runtime. [VLA Architecture Overview](openwiki/architecture/index.md).

## 🔬 Safety Mechanisms (The "How")

### I. Inference-Time Filtering
These are fast, run-time vetoes applied to candidate actions $\mathbf{a}_{candidate}$. They address immediate dangers.
*   **Mechanism:** [Inference-Time Filtering](openwiki/mechanisms/inference_time_filtering.md).
*   **Use Case Example:** Stopping movement immediately if an object is detected in the path that violates a minimum distance constraint.

### II. Policy Alignment (Training Time)
These methods modify the core policy during training to prevent unsafe behavior from ever becoming viable.
*   **Mechanism:** [Policy Alignment for Safety](openwiki/mechanisms/policy_alignment.md).
*   **Use Case Example:** Ensuring that even when highly rewarded by a complex task goal, the agent never learns to approach joints beyond their rated torque limits.

### III. World Model Prediction (Proactive & Predictive)
The most comprehensive layer; it predicts future risk over time steps $t+1$ to $t+N$.
*   **Mechanism:** [World Model for Predictive Safety](openwiki/mechanisms/world_modeling_safety.md).
*   **Use Case Example:** Detecting that a planned sequence of actions (e.g., grasping an unstable object, moving quickly) will *cause* the object to fall two seconds later.

## 📚 Key Research References
For the deepest dive into the academic work defining these concepts and mechanisms, please refer to:
*   [Foundational Background](VLA-safety/README.md) (This file was used to build the entire wiki.)

## 🔧 Future Agent Guidance
When modifying any component of a VLA system, developers should consider this hierarchy:

1.  **Can I fix it by improving the input data?** $\rightarrow$ Check Domain Concepts.
2.  **Do I need immediate runtime protection?** $\rightarrow$ Use Inference-Time Filters.
3.  **Is my goal just maximizing reward?** $\rightarrow$ Implement Policy Alignment (CMDP).
4.  **Does my task have a multi-step, long-term failure mode?** $\rightarrow$ Integrate World Model prediction.