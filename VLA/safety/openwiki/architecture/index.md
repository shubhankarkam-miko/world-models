# VLA System Architecture Overview

This page provides a high-level view of how various safety components integrate to form a robust Vision-Language-Action (VLA) model pipeline. The architecture is inherently multi-layered, meaning multiple checks and guardrails operate at different points—from planning/trajectory generation through execution monitoring.

## 🌐 Core Pipeline Flow
The VLA system generally follows this flow:

1.  **Input:** Observation ($O$) + Instruction ($\text{I}$).
2.  **Planning & Prediction:** The core model generates a preliminary action sequence or policy $\hat{\pi}$. This stage often uses **Diffusion Planning** techniques (see [VLA/Diffusion-robotics/](file:///VLA/Diffusion-robotics/) reports) to sample likely trajectories.
3.  **Safety Interception (Guardrails):** The predicted output is intercepted by multiple, specialized safety mechanisms. These checks prune unsafe or non-compliant outputs *before* they become concrete actions.
    *   *Key Components:* **Policy Alignment**, **Dual-Loop Defense**, and **Inference Time Filtering**.
4.  **Execution:** A safe action ($A$) is passed to the environment.

## 🧱 Architectural Layers & Concepts

### Vision-Language-Action Safety Architecture (VLESA)
The VLA system safety approach draws heavily from formalization, exemplified by models like VLESA. This architecture emphasizes that safety cannot be a single module but must be integrated into every stage of the pipeline: data collection, model training, planning, and runtime inference.

*   **Source Reference:** Detailed reports on this topic can be found in the `VLA/safety` directory (e.g., [hu2026-vlesa-architecture.html](file:///VLA/safety/hu2026-vlesa-architecture.html)).

### Diffusion Planning Integration
A critical part of the architecture involves using diffusion models to sample possible action spaces, which is more robust than single-point prediction. This allows for *value-guided* planning, where trajectories are scored not just on feasibility, but on predicted safety and utility.

*   **Source Reference:** See reports in `VLA/Diffusion-robotics/` (e.g., [value-guided-diffusion-planning-report.md](file:///VLA/Diffusion-robotics/value-guided-diffusion-planning-report.md)).

## 🔍 Summary of Mechanisms Integration
The architectural diagrams and components are detailed in the dedicated *Mechanisms* section:

*   **[Policy Alignment](#)**: Ensures the model's goal is aligned with human values.
*   **[Dual-Loop Defense](#)**: Provides redundant checks for critical actions.
*   **[Inference Time Filtering](#)**: Acts as a final gatekeeper before action execution.

***Source Map:** This page serves as an index linking the abstract system architecture to the concrete, technical implementations.*