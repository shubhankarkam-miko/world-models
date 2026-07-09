# Reading List: Diffusion Models for Robotic Control, Trajectory Planning, and World-Action Models

## Purpose

Reading list tracking how diffusion models have transitioned from image/video generation into robotics. Originally adopted to handle highly multimodal human demonstration data (where a robot might choose multiple valid paths to accomplish the same task), diffusion has evolved into a framework for trajectory generation, reactive control, and safety-constrained planning. The list's purpose is to bridge the core architectures that established this paradigm with the latest breakthroughs, specifically focusing on overcoming the inference latency of iterative denoising and the recent shift toward unified World-Action Models (WAMs)

## Foundational Background

- **Diffusion Policy for Robot Learning: What It Is and How to Use It** (Chi et al., 2023) — [Link](https://diffusion-policy.cs.columbia.edu/diffusion_policy_ijrr.pdf) — The foundational bedrock of modern diffusion in robotics. It treats action generation as a conditional denoising process over robot trajectories, demonstrating a profound capability to capture multimodal action distributions and recover from real-world physical perturbations.
- **Planning with Diffusion for Flexible Behavior Synthesis** (Janner et al., 2022) — [Link](https://arxiv.org/abs/2205.09991) — Often recognized via its core framework _Diffuser_, this classic paper formalized trajectory planning as a global diffusion process over full sequence spaces. It introduced the concept of utilizing test-time value guidance or inpainting to steer generated trajectories toward target goals without retraining.
- **Diffusion Models for Robotic Manipulation: A Survey** (Wolf et al., 2025) — [Link](https://arxiv.org/abs/2504.08438) — A comprehensive, high-level structural breakdown mapping how diffusion variants integrate across reinforcement learning, imitation learning, grasp planning, and data augmentation environments.

## Recent Works
### Real-Time Inference & Consistency Distillation

- **FRMD: Fast Robot Motion Diffusion via Trajectory-Level Consistency Distillation** (Frontiers in Robotics and AI, 2026) — [Link](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1751688/full) — Addresses the notorious latency bottleneck of diffusion policies. By enforcing self-consistency along the probability-flow ODE, this framework collapses slow, iterative denoising chains into fast, trajectory-space generations suitable for real-time control frequencies.
- **Muninn: Your Trajectory Diffusion Model But Faster** (arXiv, May 2026) — [Link](https://arxiv.org/html/2605.09999v1) — A clever training-free deployment framework. It evaluates internal, timestep-wise representations to predict when a trajectory has reached a stable structure, caching denoiser outputs to skip redundant generation cycles while preserving plan accuracy.

### Value Guidance & Multi-Agent Motion Planning

- **Generative Multi-Robot Motion Planning via Diffusion Modeling with Multi-Agent Reinforcement Learning Guidance** (arXiv, June 2026) — [Link](https://arxiv.org/html/2606.00933) — Scales trajectory diffusion into multi-agent arenas. Instead of deploying complex joint-space planners, single agents generate individual candidate paths which are then dynamically steered via gradients from a centralized MARL value function to resolve spatial conflicts.
- **Generative Trajectory Planning in Dynamic Environments: A Joint Diffusion and Reinforcement Learning Framework** (OpenReview, 2025/2026) — [Link](https://openreview.net/forum?id=MKM8iEaowV) — Tackles real-time path optimization amidst moving obstacles by dividing global trajectories into shorter-horizon, diffusion-generated "sub-paths". It embeds local collision probability metrics directly into a deep reinforcement learning architecture for highly adaptive execution.

### The Evolution of World-Action Models (WAMs)

- **Pretrained to Imagine, Fine-Tuned to Act: The Rise of World-Action Models** (NVIDIA Developer Blog, 2025/2026) — [Link](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/) — An essential primer analyzing the tectonic shift from pure behavior-cloning diffusion loops to World-Action Models (WAMs) like $\pi_0$ and DreamZero. It details how monolithic and hierarchical video/world diffusion backbones are being reused as foundational physical priors to simulate and execute actions concurrently.
- **PAIWorld: A 3D-Consistent World Foundation Model for Robotic Manipulation** (arXiv, June 2026) — [Link](https://arxiv.org/html/2606.18375v2) — Explores world foundation models built on Diffusion Transformer (DiT) architectures. This paper focuses heavily on maintaining strict multi-view 3D consistency during spatial imagination rollouts, which serves as a highly robust backdrop for training downstream diffusion-based control policies.