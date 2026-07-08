# Architecture of a Safe VLA Agent

The system does not have a single source codebase architecture, but rather an architectural pattern composed of multiple safety layers integrated around the core VLA decision loop. The goal is to ensure that even if the primary policy model suggests an unsafe action ($\mathbf{a}_{unsafe}$), it can be caught and overridden by one or more dedicated safety components.

## 🔄 The VLA Decision Loop with Safety Layers

A typical interaction sequence proceeds as follows, with mandatory checks at three key points:

1. **Observation:** Input is gathered (Image + Text Prompt).
2. **Policy Generation ($\mathbf{a}_{candidate}$):** The core VLA model predicts an action candidate.
3. **Safety Filter 1: Inference-Time Check (Pre-Execution)**
    *   *Purpose:* To filter out actions that are physically dangerous, nonsensical, or violate immediate constraints *before* they hit the executor/simulator.
    *   *Mechanism Examples:* VLESA Safety Q-Filter, Attention-Guided CBF.
4. **Safety Filter 2: World Model Check (Predictive)**
    *   *Purpose:* To simulate the effect of $\mathbf{a}_{candidate}$ on a future state ($\mathbf{s}'$) and predict if that future state will lead to failure or excessive cost. This is *proactive*.
    *   *Mechanism Examples:* SafeDojo's action-conditioned world model prediction.
5. **Policy Correction/Execution:** If all safety filters pass, $\mathbf{a}_{candidate}$ proceeds. If they fail, the system enters a safe recovery state (e.g., stopping movement, issuing a warning).

## 🧩 Component Integration Map

| Component | Function | Location in Loop | Core Challenge Addressed | Source Research |
| :--- | :--- | :--- | :--- | :--- |
| **Core VLA Policy** | Generates the intended action $\mathbf{a}$. | Generation (Step 2) | Behavioral planning. | N/A (The subject model) |
| **Inference Filter** | Real-time veto of immediate, unsafe actions. | Pre-Execution (Step 3) | Immediate physical hazards/violations. | [VLESA](https://arxiv.org/abs/2606.03954), [Attention Filter](https://arxiv.org/abs/2606.09749) |
| **World Model** | Simulation of future states and costs using predictive modeling. | Predictive (Step 4) | Long-term planning failure, cascading risk. | [SafeDojo](https://arxiv.org/abs/2606.20698), [World-Env](https://arxiv.org/abs/2509.24948). See [World Modeling Safety](openwiki/mechanisms/world_modeling_safety.md) for details. |
| **Constrained Learner** | Optimizing the policy against safety risks. | Training Phase (Pre-Deployment) | Bias towards unsafe high-reward paths. | SafeVLA (NeurIPS, 2025) |

## 🛠️ Agent Workflow Guidance for Future Developers

When implementing or modifying a VLA agent:
1. **Prioritize Constraint:** Never rely solely on the primary VLA policy. Always implement at least one external safety check mechanism.
2. **Start with Filters (Fast):** For initial prototyping, use Inference-Time Filters as they are computationally cheap and provide immediate risk assessment.
3. **Scale to Prediction (Harder):** For robust deployment, integrate World Model prediction layers, as these require significant computational resources but offer the highest safety guarantee by predicting failure modes.