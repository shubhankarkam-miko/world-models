# Core Concepts in VLA Safety Domains

This domain page defines the fundamental concepts necessary for understanding why safety mechanisms are required for embodied AI agents (VLAs). It draws parallels between established text-based moderation techniques and their physical manifestation in robotics.

## 🌐 The Problem Domain: Profanity in Physical Space
In natural language processing, "profanity" refers to tokens or phrases deemed offensive or harmful. For a VLA agent, the concept of profanity translates into **Hazardous Actions**. A dangerous action might be defined as:
*   Collision with an obstacle (physical damage).
*   Exceeding joint torque limits (mechanical failure).
*   Manipulating objects in a way that violates ethical or physical constraints.

This fundamental shift requires safety checks to move from analyzing abstract tokens to predicting concrete, physical outcomes.

## ⚖️ The Three Safety Paradigms (Source: /VLA-safety/README.md)
The field of VLA safety addresses risk using three analogous approaches:

### 1. Inference-Time Filtering (The Output Moderator)
*   **Concept:** Intervening *after* the main policy generates an action, but *before* it is executed. It acts as a hard veto mechanism against unsafe outputs.
*   **Analogy:** A standalone profanity filter that checks an output string and blocks it if illegal words are found.
*   **Mechanism:** Requires lightweight, highly performant models (e.g., Safety Q-Filters) that can run at high frequency to vet candidate actions $\mathbf{a}_{candidate}$.
*   **Related Research:** [VLESA](https://arxiv.org/abs/2606.03954), [Attention-Guided Filter](https://arxiv.org/abs/2606.09749).

### 2. Policy Alignment (The Constrained Trainer)
*   **Concept:** Modifying the VLA model's training objective to *learn* safety constraints inherently. The agent is trained not just for maximum reward, but also for minimum risk violation.
*   **Analogy:** Safety alignment methods like RLHF (Reinforcement Learning from Human Feedback), where the system learns what "good" or "safe" behavior looks like from expert demonstrations and explicit rules.
*   **Mechanism:** Uses Constrained Markov Decision Processes (CMDPs) to optimize performance subject to safety cost constraints, forcing a fundamental trade-off between goal completion and risk mitigation.
*   **Related Research:** [SafeVLA](https://openreview.net/forum?id=dt940loCBT).

### 3. World Model Prediction (The Pre-Visualizer)
*   **Concept:** Using a predictive world model to simulate the consequences of an action ($\mathbf{a}$) *before* it is performed, effectively allowing "imagining" dangerous outcomes. This provides safety guarantees for long horizons.
*   **Analogy:** A comprehensive physical simulator that warns you about cascading failures before you touch anything in the real world.
*   **Mechanism:** The model predicts a sequence of future states ($\mathbf{s} \to \mathbf{s}' \to \mathbf{s}''...$) and evaluates the cumulative risk or cost associated with those transitions.
*   **Related Research:** [SafeDojo](https://arxiv.org/abs/2606.20698), [World-Env](https://arxiv.org/abs/2509.24948).

By understanding these distinct approaches, developers can select the right layer of defense based on required latency and necessary safety guarantees.