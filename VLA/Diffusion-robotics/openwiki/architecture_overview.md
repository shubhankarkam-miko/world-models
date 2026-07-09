# Diffusion Policy Architecture and Components

This page serves as an architectural map for implementing diffusion models in robotics and trajectory planning. The goal is to define the standard components and how they interact to generate actionable, safe, and temporally coherent trajectories.

## Core Model Structure

The typical architecture adapts a general-purpose time-series diffusion framework (like those used in image generation) to sequence data: robot actions $\mathbf{a}_t$.

1.  **Encoder/Conditioning:** The model must accept rich conditioning information $c$, which can include:
    *   Observation state ($\mathbf{o}_t$): Camera views, joint angles.
    *   Goal target ($g$): Final desired end-effector position or state.
    *   Contextual data: Time step embeddings.
2.  **Diffusion Backbone:** This is the core model (e.g., a Transformer or U-Net variant) responsible for learning the noise process $\mathbf{x}_0 \rightarrow \mathbf{x}_T$. It estimates the required noise level and predicts the noise added at each time step, $\epsilon(\mathbf{x}_t, t | c)$.
3.  **Prediction Head:** This maps the denoised representation back to the desired action space or latent trajectory space ($\hat{\mathbf{a}}_0$).

## Key Architectural Variants

*   **Value-Guided Diffusion (Janner et al., 2022):** Incorporates a separate Value Function ($V$) trained via Reinforcement Learning. During inference, this value function guides the sampling process by biasing the trajectory generation toward higher expected returns, making the policy more goal-directed and robust than pure imitation learning.
*   **World-Action Models (WAMs) (NVIDIA):** Represents a shift towards using large, pretrained foundation models that integrate world understanding (e.g., object permanence, physics priors) directly into the action space. The WAM acts as a universal prior for generating *plausible* actions given context and goal state.

## Implementation Considerations
When designing a system, consider:
1.  **Data Modality:** Is it continuous joint space, or discrete high-level commands? This dictates the output dimension of the prediction head.
2.  **Computational Budget:** The trade-off between fidelity (requiring many iterative steps) and real-time performance (demanding distillation/consistency methods like FRMD).

***Source References:***
*   [Chi et al., 2023](https://diffusion_policy_ijrr.pdf): Foundational policy diffusion.
*   [Janner et al., 2022](https://arxiv.org/abs/2205.09991): Early guidance mechanisms.
*   [PAIWorld: A 3D-Consistent World Foundation Model for Robotic Manipulation](https://arxiv.org/html/2606.18375v2): Modern WAM integration.