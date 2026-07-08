## 🧠 Safety Mechanisms Overview
This section details the core architectural patterns and mechanisms developed to ensure safety in VLA systems. These components are applied throughout the agent pipeline as defined in the overall architecture.

### Inference-Time Filtering (VLESA)
This layer focuses on fast, real-time checks to veto unsafe actions before execution. It is crucial for immediate hazard avoidance.
*   **Core Concept:** Implementing safety constraints directly into the attention mechanism or action space filtering based on context.
*   **Reference:** See [VLESA Architecture](openwiki/architecture/hu2026-vlesa-architecture.html) for implementation details and flow.

### Policy Alignment (SafeVLA)
This layer focuses on training the VLA model to inherently prefer safe behaviors through constrained optimization during the learning phase.
*   **Core Concept:** Training objectives that penalize unsafe actions, ensuring the policy learned is aligned with safety benchmarks.
*   **Reference:** See [SafeVLA Report](openwiki/testing/zhang2025-safevla-report.md) for methodology and results.

### World Model Prediction (World-Env / SafeDojo)
This layer uses predictive simulation to assess the long-term consequences of actions, addressing potential cascading failures.
*   **Core Concept:** Predicting future states ($\mathbf{s}'$) and costs based on an action $\mathbf{a}$, allowing for proactive risk assessment before execution.
*   **Reference:** See [World-Env Architecture](openwiki/architecture/xiao2026-worldenv-architecture.html) and [SafeDojo Report](openwiki/testing/tang2026-safedojo-report.md) for simulation details.

### Dual-Loop Defense
This describes the layered application of these mechanisms across time, from immediate filtering to long-term world modeling.
*   **Reference:** See [Dual-Loop Defense Architecture](openwiki/mechanisms/dual_loop_defense.md).