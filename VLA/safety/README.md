## Preface

Contains research on safety mechanisms for embodied AI models (VLAs and World Models) that function similarly to text-based moderation or profanity filters. In the LLM space, safety is handled either by external input/output classifiers (filtering out bad tokens) or by internal safety alignment (RLHF/HDPO). In the physical robotics space, "profanity" translates to hazardous physical actions - such as collisiopns, breaking joint constraints, or unsafe object manipulation. Therefore, this list focuses on three analogous approaches: inference-time safety filters (the direct equivalent of an output moderator), safety alignment via contrained training (the equivalent of safety RLHF), and using world models to "hallucinate" and catch unsafe futures before they occur in the real world.

## Foundational Background

- **Vision-Language-Action Safety: Threats, Challenges, Evaluations, and Mechanisms** (arXiv, April 2026) — [Link](li2026-vla-safety-report.md) — This comprehensive survey directly defines the scope of VLA safety and explicitly distinguishes the concrete physical hazards of embodied agents from the abstract text-safety issues of standard LLMs. It is the best starting point for understanding the landscape of training-time and inference-time defenses.
- **VLA-RISK: Benchmarking Vision-Language-Action Models with Physical Robustness** (OpenReview, 2026) — [Link](https://openreview.net/pdf/2b0044c5e9586d1b0dce44c7f3a73dbc43d13da0.pdf) — Just as text models need toxic prompt benchmarks, VLAs need physical vulnerability benchmarks. This paper introduces a testing suite for evaluating how VLAs fail when confronted with confusing or adversarial scenarios across object, action, and space dimensions.

## Recent Works

### Inference-Time Filtering 
- **VLESA: Vision-Language Embodied Safety Agent for Human Activity Monitoring** (arXiv, June 2026) — [Link](hu2026-vlesa-report.md) — This work trains a specific "Safety Q-Filter" on a dataset of safe/unsafe visual actions to judge and intervene in real-time. It is the most direct parallel to a standalone profanity filter, but built for evaluating visual intent and candidate physical actions.
- **Your Model Already Knows: Attention-Guided Safety Filter for Vision-Language-Action Models** (arXiv, June 2026) — [Link](https://arxiv.org/abs/2606.09749) — Instead of using an external heavy vision model to filter outputs, this paper extracts perceptual signals directly from the VLA's internal attention heads to run a real-time Control Barrier Function (CBF), acting as a fast, training-free safety guardrail during execution.

### Policy Alignment
- **SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning** (NeurIPS, 2025) — [Link](zhang2025-safevla-report.md) — Similar to how LLMs undergo safety RLHF to avoid generating toxic text, this paper uses Constrained Markov Decision Processes (CMDPs) to optimize VLAs against elicited safety risks. It explicitly forces the model to learn a safety-performance trade-off for physical manipulation.

### World Models for Safe Imagination
- **SafeDojo: Safe Reinforcement Learning for VLA via Interactive World Model** (arXiv, June 2026) — [Link](tang2026-safedojo-report.md) — This paper uses an action-conditioned world model to let the VLA "dream" its future actions, using a dedicated safety head to predict safety costs from the imagined frames before any real-world damage occurs.
- **World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training** (arXiv, September 2025) — [Link](xiao2026-worldenv-report.md) — Demonstrates how a physically-consistent world simulator can act as a safe virtual environment, providing continuous safety constraint checks and reward signals to fine-tune a VLA policy without the physical risks of real-world trial and error.